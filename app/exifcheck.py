"""图片隐私检查 —— 只读解析 PNG / JPEG 中的 EXIF 隐私字段，并可导出隐私副本。

仅提取：
  - 拍摄时间（DateTimeOriginal，缺失时回退 DateTime / DateTimeDigitized）
  - 相机品牌 Make / 型号 Model
  - GPS 经纬度（度分秒 + N/S、E/W 方向标记还原为带符号十进制度）

另提供「导出隐私副本」（build_privacy_copy）：
  - 先按 EXIF Orientation 把像素摆正（ImageOps.exif_transpose 同一实现）；
  - 再以**全新 PNG** 重编码（新对象、空 info，不传入任何 encoderinfo），
    保留原有透明通道（RGBA / LA / P 透明），不写入 EXIF / XMP / ICC /
    tEXt·zTXt·iTXt 等任何元数据；
  - 导出后对生成字节**重新解析 + PNG 原始数据块扫描**双重复查，
    发现任一上述元数据即视为服务端内部错误，不返回副本。

设计原则：
  - 纯内存、只读：不保存图片或任何元数据，不创建水印任务；
  - EXIF 损坏 / 无法解码时绝不抛出阻断性错误，字段显示为“无”，
    并在 warnings 中给出提示（图片本身仍可正常解码检查）；
  - 本模块与水印任务、攻击记录、报告归档完全隔离。
"""

from __future__ import annotations

import base64
import datetime
import hashlib
import io
import math
import struct
from typing import Any

from PIL import Image, ImageOps
from PIL.ExifTags import Base, GPS, IFD

# 仅允许 PNG / JPEG（JPEG 含 jpg / jpe 等扩展名，格式以 Pillow 解码结果为准）
ALLOWED_FORMATS = {"PNG", "JPEG"}

# 拍摄时间候选：(所在 IFD, 标签)，按优先级排列。
# DateTimeOriginal / DateTimeDigitized 在 Exif sub-IFD，DateTime 在 IFD0。
_DATETIME_TAGS = (
    (IFD.Exif, Base.DateTimeOriginal),
    (0, Base.DateTime),
    (IFD.Exif, Base.DateTimeDigitized),
)


class PrivacyCheckError(ValueError):
    """图片本身无法解码或格式不被允许（区别于“EXIF 损坏”）。"""


# ---------------------------------------------------------------- 基础工具

def _safe_text(value: Any) -> str | None:
    """把 EXIF 字符串值清洗为可用文本；空 / 全空白返回 None。"""
    if value is None:
        return None
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")
    text = str(value).replace("\x00", " ").strip()
    return text or None


def _as_float(value: Any) -> float | None:
    """容忍 IFDRational / tuple / 字符串 / 各种数值类型转 float。"""
    if value is None:
        return None
    try:
        f = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if not math.isfinite(f):
        return None
    return f


def _dms_to_decimal(dms: Any, ref: str | None) -> tuple[float | None, str | None]:
    """度分秒三元组 + 方向标记 -> 带符号十进制度。

    返回 (十进制, 警告)；无法解析时返回 (None, 警告或 None)。
    也兼容直接给十进制度数的非标准写法。
    """
    warning = None
    sign = None
    if ref:
        r = str(ref).strip().upper()
        if r in ("S", "W"):
            sign = -1
        elif r in ("N", "E"):
            sign = 1
        else:
            warning = f"无法识别的方向标记 {ref!r}，已按正向处理"
            sign = 1
    else:
        # 方向标记缺失：按 EXIF 约定无法确定半球，缺省按正向
        sign = 1

    if not isinstance(dms, (tuple, list)):
        dec = _as_float(dms)
        return (sign * dec if dec is not None else None), warning

    if len(dms) != 3:
        return None, "GPS 度分秒字段不完整"

    parts = [_as_float(v) for v in dms]
    if any(v is None for v in parts):
        return None, "GPS 度分秒含无法解析的数值"

    degrees, minutes, seconds = parts
    if not (0 <= minutes < 60 and 0 <= seconds < 60 and degrees >= 0):
        return None, "GPS 度分秒数值超出合法范围"

    dec = degrees + minutes / 60.0 + seconds / 3600.0
    dec = sign * dec
    return (round(dec, 7) if dec != 0 else 0.0), warning


def _format_dms(dms: Any, ref: str | None) -> str | None:
    """生成展示用“度° 分′ 秒″ 方向”字符串。"""
    if not isinstance(dms, (tuple, list)) or len(dms) != 3:
        return None
    parts = [_as_float(v) for v in dms]
    if any(v is None for v in parts):
        return None
    return f"{parts[0]:g}° {parts[1]:g}′ {parts[2]:g}″ {ref or ''}".strip()


def _parse_exif_datetime(text: str | None) -> str | None:
    """EXIF 'YYYY:MM:DD HH:MM:SS' -> ISO 8601 'YYYY-MM-DDTHH:MM:SS'。

    时区在 EXIF 中通常缺省，这里只做格式归一化，不臆测时区。
    """
    if not text:
        return None
    candidate = text.strip().replace("\x00", "")
    for fmt in ("%Y:%m:%d %H:%M:%S", "%Y-%m-%d %H:%M:%S",
                "%Y:%m:%d %H:%M", "%Y-%m-%dT%H:%M:%S"):
        try:
            return datetime.datetime.strptime(candidate, fmt).isoformat()
        except ValueError:
            continue
    return None


def open_supported_image(data: bytes) -> Image.Image:
    """打开并校验 PNG/JPEG 图片。

    损坏 / 无法解码 / 格式不支持 -> PrivacyCheckError；
    EXIF 异常在此忽略（由 getexif 的具体解析再决定），像素仍照常解码。
    """
    try:
        im = Image.open(io.BytesIO(data))
        # 提前触发像素解码，拒绝截断 / 无法解码的文件
        im.load()
    except PrivacyCheckError:
        raise
    except Exception as e:
        raise PrivacyCheckError("无法解析上传的图片。") from e

    if im.format not in ALLOWED_FORMATS:
        allowed = "PNG / JPEG"
        raise PrivacyCheckError(
            f"仅支持 {allowed} 图片（当前为 {im.format or '未知格式'}）。")
    return im


def open_rgb_transposed(data: bytes) -> Image.Image:
    """供水印任务使用的安全打开：EXIF 损坏不阻断，旋转失败时退回原图。"""
    try:
        im = Image.open(io.BytesIO(data))
        im.load()
    except Exception as e:
        raise PrivacyCheckError("无法解析上传的图片。") from e
    try:
        im = ImageOps.exif_transpose(im)
    except Exception:
        # EXIF 损坏可能让方向读取抛错：忽略方向，继续用原图
        im = Image.open(io.BytesIO(data))
    return im.convert("RGB")


# ---------------------------------------------------------------- EXIF 提取

def _get_gps_ifd(exif: Any) -> dict | None:
    try:
        return exif.get_ifd(Base.GPSInfo) or {}
    except Exception:
        return None


def _get_exif_ifd(exif: Any) -> dict | None:
    try:
        return exif.get_ifd(IFD.Exif) or {}
    except Exception:
        return None


def _has_raw_exif(im: Image.Image) -> bool:
    """文件中是否实际带有 EXIF 数据段（JPEG APP1 / PNG eXIf）。

    用于区分“本来就没有 EXIF”与“EXIF 损坏导致 getexif() 静默返回空”。
    """
    try:
        raw = im.info.get("exif")
        if isinstance(raw, (bytes, bytearray)) and len(raw) >= 6:
            return True
    except Exception:
        return False
    return False


def _extract_gps(gps_ifd: dict | None, warnings: list) -> dict | None:
    if not gps_ifd:
        return None

    lat_dms = gps_ifd.get(GPS.GPSLatitude)
    lat_ref = gps_ifd.get(GPS.GPSLatitudeRef)
    lon_dms = gps_ifd.get(GPS.GPSLongitude)
    lon_ref = gps_ifd.get(GPS.GPSLongitudeRef)

    has_any = any(v is not None for v in (lat_dms, lat_ref, lon_dms, lon_ref))
    if not has_any:
        return None

    result = {
        "latitude": None,
        "latitude_dms": None,
        "latitude_ref": str(lat_ref).strip() if lat_ref is not None else None,
        "longitude": None,
        "longitude_dms": None,
        "longitude_ref": str(lon_ref).strip() if lon_ref is not None else None,
    }

    partial = False
    if lat_dms is not None:
        lat, warn = _dms_to_decimal(lat_dms,
                                    result["latitude_ref"])
        result["latitude"] = lat
        result["latitude_dms"] = _format_dms(lat_dms, result["latitude_ref"])
        if warn:
            warnings.append(f"纬度：{warn}")
        partial |= lat is None
    else:
        partial = True

    if lon_dms is not None:
        lon, warn = _dms_to_decimal(lon_dms,
                                    result["longitude_ref"])
        result["longitude"] = lon
        result["longitude_dms"] = _format_dms(lon_dms, result["longitude_ref"])
        if warn:
            warnings.append(f"经度：{warn}")
        partial |= lon is None
    else:
        partial = True

    if partial:
        warnings.append("GPS 信息不完整，缺失的经纬度显示为“无”。")
    return result


def analyze_privacy(data: bytes) -> dict:
    """对 PNG/JPEG 字节做隐私检查，返回可直接序列化为 JSON 的结果。"""
    im = open_supported_image(data)

    warnings: list[str] = []
    make = model = None
    datetime_raw = datetime_iso = None
    gps = None
    raw_exif = _has_raw_exif(im)

    try:
        exif = im.getexif()
    except Exception as e:
        # 损坏 / 无法解码的 EXIF（SyntaxError、struct.error 等）
        exif = None
        warnings.append(
            f"EXIF 数据已损坏或无法解码（{type(e).__name__}），已跳过元数据解析，"
            "图片本身检查不受影响。")

    main_ifd = {}
    sub_ifd = None
    if exif is not None:
        try:
            main_ifd = dict(exif)
        except Exception:
            main_ifd = {}
            raw_exif = True
            if not any("损坏" in w for w in warnings):
                warnings.append(
                    "EXIF 数据部分损坏，部分字段无法读取，图片检查不受影响。")
        sub_ifd = _get_exif_ifd(exif)

        # 品牌 / 型号（任何单项异常都不应影响其它字段）
        try:
            make = _safe_text(main_ifd.get(Base.Make))
        except Exception:
            make = None
        try:
            model = _safe_text(main_ifd.get(Base.Model))
        except Exception:
            model = None

        # 拍摄时间：Exif sub-IFD 的 DateTimeOriginal 优先，回退 IFD0 的 DateTime
        for source, tag in _DATETIME_TAGS:
            bucket = main_ifd if source == 0 else sub_ifd
            if not bucket:
                continue
            try:
                raw = _safe_text(bucket.get(tag))
            except Exception:
                raw = None
            if raw:
                datetime_raw = raw
                datetime_iso = _parse_exif_datetime(raw)
                break

        # GPS（位于独立 IFD）
        gps = _extract_gps(_get_gps_ifd(exif), warnings)

    # 文件里带 EXIF 段，但解析结果为空且没有任何字段 -> 视为损坏 / 无法解码
    exif_ok = exif is not None and not any("损坏" in w for w in warnings)
    exif_present = (raw_exif or bool(main_ifd) or bool(sub_ifd)
                    or gps is not None)
    if raw_exif and exif_ok and not (main_ifd or sub_ifd or gps):
        exif_ok = False
        warnings.append(
            "EXIF 数据已损坏或无法解码，未能读取任何元数据，图片本身检查不受影响。")

    return {
        "format": im.format,
        "width": int(im.size[0]),
        "height": int(im.size[1]),
        "exif_present": bool(exif_present),
        "exif_ok": exif_ok,
        "date_time": datetime_raw,
        "date_time_iso": datetime_iso,
        "camera_make": make,
        "camera_model": model,
        "gps": gps,
        "warnings": warnings,
    }


# ---------------------------------------------------------------- 隐私副本导出

class PrivacyExportError(RuntimeError):
    """导出失败：图片无法解析（PrivacyCheckError）或副本复查发现元数据残留。"""


# EXIF Orientation 的中文含义（仅用于响应展示）。摆正操作直接复用
# ImageOps.exif_transpose（其 6 -> ROTATE_270、8 -> ROTATE_90 的映射即
# EXIF 规范的“顺时针 90°/逆时针 90°”）。
ORIENTATION_LABELS = {
    1: "正常（1）",
    2: "水平镜像（2）",
    3: "旋转 180°（3）",
    4: "垂直镜像（4）",
    5: "主对角线翻转（5）",
    6: "顺时针旋转 90°（6）",
    7: "反对角线翻转（7）",
    8: "逆时针旋转 90° / 顺时针 270°（8）",
}

# 副本复查时禁止出现的 PNG 数据块：
#   eXIf = EXIF；iCCP / cHRM / gAMA / sRGB / sBIT = 色彩/ICC 相关；
#   tEXt / zTXt / iTXt = PNG 文本元数据（含 XMP、Comment 等）。
_FORBIDDEN_PNG_CHUNKS = frozenset({
    b"eXIf",
    b"iCCP", b"cHRM", b"gAMA", b"sRGB", b"sBIT",
    b"tEXt", b"zTXt", b"iTXt",
})

_PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def _get_orientation_tag(im: Image.Image) -> int:
    """读取 EXIF Orientation（1–8）；缺失/损坏/越界一律按 1（正常）处理。"""
    try:
        value = int(im.getexif().get(Base.Orientation, 1))
    except Exception:
        return 1
    return value if 1 <= value <= 8 else 1


def _to_clean_mode(im: Image.Image) -> tuple[Image.Image, bool]:
    """归一化为 PNG 可无损表达的模式，并保留透明通道。

    返回 (新图, 是否保留了 alpha 通道)：
      - RGBA / LA 原样保留；
      - 带透明信息的调色板（P）图像转 RGBA（保留逐像素/索引透明）；
      - 其余模式转 RGB（JPEG 的 CMYK/YCbCr 等也由此覆盖）；
      - 位图（1）与灰度（L）直接保留。
    """
    if im.mode in ("RGB", "RGBA", "L", "LA", "1"):
        return im, im.mode in ("RGBA", "LA")
    if im.mode == "P":
        has_transparency = "transparency" in im.info
        if has_transparency or im.im.getpalettemode() == "RGBA":
            return im.convert("RGBA"), True
        return im.convert("RGB"), False
    if im.mode in ("CMYK", "YCbCr", "RGBa", "LAB", "HSV", "I", "I;16", "F"):
        return im.convert("RGB"), False
    return im.convert("RGB"), False


def _strip_to_fresh_image(im: Image.Image) -> Image.Image:
    """用归一化后的原始像素字节构造全新对象 —— 空 info，杜绝一切元数据带入。"""
    try:
        im.load()
    except Exception:
        pass
    fresh = Image.frombytes(im.mode, im.size, im.tobytes())
    # frombytes 得到的对象本就无元数据；显式清空以保证任何 Pillow 版本下都干净。
    fresh.info = {}
    return fresh


def scan_png_metadata(png_data: bytes) -> list[str]:
    """扫描 PNG 字节中的元数据数据块，返回命中的禁止数据块类型名（空=干净）。"""
    if not png_data.startswith(_PNG_MAGIC):
        return ["非 PNG 文件"]
    found: list[str] = []
    pos = len(_PNG_MAGIC)
    while pos + 8 <= len(png_data):
        (length,) = struct.unpack(">I", png_data[pos:pos + 4])
        cid = png_data[pos + 4:pos + 8]
        if len(cid) < 4:
            break
        if cid in _FORBIDDEN_PNG_CHUNKS and cid.decode() not in found:
            found.append(cid.decode())
        if cid == b"IEND":
            break
        pos += 12 + length  # 4 长度 + 4 类型 + 数据 + 4 CRC
        if pos > len(png_data):
            break
    return found


def recheck_clean_png(png_data: bytes) -> dict:
    """对生成的 PNG 字节做「Pillow 重新解析 + 原始数据块扫描」双重复查。

    任何 EXIF / XMP / ICC / PNG 文本（tEXt/zTXt/iTXt）残留都视为复查失败。
    """
    raw_hits = scan_png_metadata(png_data)

    pil_hits: list[str] = []
    fmt = None
    try:
        with Image.open(io.BytesIO(png_data)) as reim:
            reim.load()
            fmt = reim.format
            info = reim.info or {}
            if info.get("exif"):
                pil_hits.append("eXIf")
            try:
                if dict(reim.getexif()):
                    pil_hits.append("EXIF-IFD")
            except Exception:
                pil_hits.append("EXIF-损坏")
            if info.get("icc_profile"):
                pil_hits.append("icc_profile")
            for key in ("XML:com.adobe.xmp", "xmp", "comment"):
                if info.get(key):
                    pil_hits.append(key)
    except Exception as e:
        pil_hits.append(f"重新解析失败({type(e).__name__})")

    clean = fmt == "PNG" and not raw_hits and not pil_hits
    return {
        "clean": clean,
        "format": fmt,
        "raw_chunk_hits": raw_hits,
        "parsed_hits": pil_hits,
    }


def build_privacy_copy(data: bytes) -> dict:
    """把 PNG/JPEG 字节处理为干净的隐私副本 PNG（全程内存，不写盘、不建任务）。

    流程：
      1) open_supported_image 校验格式（仅 PNG/JPEG）并解码像素；
      2) 读取 EXIF Orientation 标签（缺失/损坏按 1 处理），按
         ImageOps.exif_transpose 同一映射摆正像素；
      3) 归一化模式（保留透明通道）后用全新 Image 对象（空 info）重编码 PNG，
         不传任何 encoderinfo —— Pillow 因此只会写 IHDR/IDAT/tRNS/IEND；
      4) 对生成字节重新解析 + PNG 数据块扫描，发现元数据残留即
         抛 PrivacyExportError（不向调用方提供副本）。

    返回：{"source": analyze_privacy 结果, "copy": {...}, "orientation": {...}}
    """
    # 1) 只读检查结果（拍摄时间 / 相机 / GPS）；格式不支持在此直接报错
    source = analyze_privacy(data)

    # 重新打开一次做几何摆正（analyze_privacy 只承诺只读解析，不保证像素已摆正）
    im = open_supported_image(data)
    orient_tag = _get_orientation_tag(im)
    orient_applied = orient_tag not in (0, 1)
    try:
        im = ImageOps.exif_transpose(im)
    except Exception:
        # 与 open_rgb_transposed 一致：摆正失败时退回未旋转的原图
        im = open_supported_image(data)
        orient_applied = False

    # 2) 归一化模式并保留透明通道；3) 全新对象 + 全新 PNG 重编码
    normalized, alpha_preserved = _to_clean_mode(im)
    fresh = _strip_to_fresh_image(normalized)
    buf = io.BytesIO()
    fresh.save(buf, format="PNG", optimize=True)
    png_bytes = buf.getvalue()

    # 4) 导出前复查：重新解析生成文件，有任何元数据残留即拒绝提供下载
    recheck = recheck_clean_png(png_bytes)
    if not recheck["clean"]:
        raise PrivacyExportError(
            "隐私副本复查未通过：生成的 PNG 仍含元数据"
            f"（数据块 {recheck['raw_chunk_hits']} / 解析 {recheck['parsed_hits']}），"
            "已拒绝提供下载。")

    return {
        "source": source,
        "orientation": {
            "tag": orient_tag,
            "label": ORIENTATION_LABELS.get(orient_tag, str(orient_tag)),
            "applied": bool(orient_applied),
        },
        "copy": {
            "format": "PNG",
            "width": int(fresh.size[0]),
            "height": int(fresh.size[1]),
            "mode": fresh.mode,
            "alpha_preserved": bool(alpha_preserved),
            "bytes": len(png_bytes),
            "sha256": hashlib.sha256(png_bytes).hexdigest(),
            "png_base64": base64.b64encode(png_bytes).decode("ascii"),
            "recheck": recheck,
        },
    }


__all__ = [
    "ALLOWED_FORMATS",
    "PrivacyCheckError",
    "PrivacyExportError",
    "ORIENTATION_LABELS",
    "analyze_privacy",
    "build_privacy_copy",
    "open_rgb_transposed",
    "recheck_clean_png",
    "scan_png_metadata",
]
