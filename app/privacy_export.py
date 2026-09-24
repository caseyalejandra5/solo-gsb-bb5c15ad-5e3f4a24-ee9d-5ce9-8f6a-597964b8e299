"""导出隐私副本 —— 摆正像素 + 全新 PNG 重编码 + 导出前复查。

流程（全程仅在内存中进行，不落盘、不创建任务）：

  1. 仅接受 JPEG / PNG，复用 ``exifcheck.open_supported_image`` 严格解码；
  2. 复用 ``exifcheck.analyze_privacy`` 解析原图 EXIF 拍摄时间 / 相机 / GPS
     （供页面展示，字段缺失为 None，损坏 EXIF 只产生警告）；
  3. 读取 EXIF Orientation（1–8），按其语义把像素真正旋转 / 镜像摆正；
     Orientation 缺失或非法时不做几何变换（损坏 EXIF 不阻断导出）；
  4. 规整像素模式（保留透明通道：RGBA / LA，P 带透明索引转 RGBA……），
     通过 ``numpy`` 数组 + ``Image.fromarray`` **重建全新图像对象**，
     从根上丢弃 ``im.info`` 中被 Pillow 在保存 PNG 时自动带入的
     ICC / EXIF / 文本等元数据，再以全新 PNG（optimize）重编码；
  5. **重新解析生成的 PNG 字节**做复查：PNG 字节级 chunk 扫描
     （禁止 eXIf / iCCP / tEXt / zTXt / iTXt）+ Pillow 再解析 +
     重新跑一次 EXIF/GPS 解析 + 像素无损/尺寸/透明通道一致性。
     复查任一硬性项不通过都抛 :class:`PrivacyCopyRecheckError`，
     调用方返回错误且**不提供下载数据**。

与水印任务、攻击记录、报告归档完全隔离；只读隐私检查仍在
``exifcheck.analyze_privacy`` 中，本模块不改变其行为。
"""

from __future__ import annotations

import base64
import hashlib
import io
import struct
from typing import Any

import numpy as np
from PIL import Image
from PIL.ExifTags import Base

from .exifcheck import (
    PrivacyCheckError,           # noqa: F401  （供调用方统一捕获）
    analyze_privacy,
    open_supported_image,
)

# 导出副本同样限制最大边长（方向摆正后的像素尺寸），避免解码 / base64 放大
MAX_EXPORT_DIM = 4096

# PNG 中可能承载本次要清除的元数据的 chunk 类型：
#   eXIf = EXIF（含 GPS / 拍摄时间 / 相机等）
#   iCCP = ICC 色彩配置
#   tEXt / zTXt / iTXt = PNG 文本元数据（XMP 通常也在 iTXt 中）
_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
_PNG_FORBIDDEN_CHUNKS = ("eXIf", "iCCP", "tEXt", "zTXt", "iTXt")

# 可原样保留透明语义的模式
_PRESERVE_MODES = ("1", "L", "LA", "RGB", "RGBA")

# EXIF Orientation -> 像素变换序列（与 Pillow ImageOps 内置映射一致）
try:
    _T = Image.Transpose
    _ORIENTATION_METHODS: dict[int, tuple[Any, ...]] = {
        1: (),
        2: (_T.FLIP_LEFT_RIGHT,),
        3: (_T.ROTATE_180,),
        4: (_T.FLIP_TOP_BOTTOM,),
        5: (_T.FLIP_LEFT_RIGHT, _T.ROTATE_90),
        6: (_T.ROTATE_270,),
        7: (_T.FLIP_LEFT_RIGHT, _T.ROTATE_270),
        8: (_T.ROTATE_90,),
    }
except AttributeError:  # pragma: no cover - 极旧 Pillow 回退
    _ORIENTATION_METHODS = {1: (), 2: (), 3: (), 4: (), 5: (),
                            6: (), 7: (), 8: ()}

_ORIENTATION_LABELS = {
    1: "正常方向（Orientation=1，无需变换）",
    2: "水平镜像（Orientation=2）",
    3: "旋转 180°（Orientation=3）",
    4: "垂直镜像（Orientation=4）",
    5: "水平镜像后顺时针旋转 90°（Orientation=5）",
    6: "顺时针旋转 90°（Orientation=6）",
    7: "水平镜像后逆时针旋转 90°（Orientation=7）",
    8: "逆时针旋转 90°（Orientation=8）",
}


class PrivacyExportError(ValueError):
    """导出前置条件不满足（如摆正后尺寸超限）。"""


class PrivacyCopyRecheckError(RuntimeError):
    """导出副本复查未通过：仍含被禁元数据或重编码结果不一致。"""

    def __init__(self, offenders: list[str], recheck: dict,
                 context: dict | None = None):
        self.offenders = offenders
        self.recheck = recheck
        self.context = context or {}
        super().__init__("；".join(offenders) or "隐私副本复查未通过。")


# ---------------------------------------------------------------- 方向摆正

def read_orientation(im: Image.Image) -> tuple[int | None, str | None]:
    """读取 EXIF Orientation；返回 (1–8 或 None, 警告或 None)。

    EXIF 损坏 / 标签缺失 / 数值非法都不阻断导出：返回 None 并附警告。
    """
    try:
        exif = im.getexif()
        value = exif.get(Base.Orientation)
    except Exception as e:
        return None, f"EXIF Orientation 无法读取（{type(e).__name__}），已按原方向导出。"
    if value is None:
        return None, None
    try:
        value = int(value)
    except (TypeError, ValueError):
        return None, f"EXIF Orientation 值非法（{value!r}），已按原方向导出。"
    if value == 1:
        return 1, None
    if value in _ORIENTATION_METHODS and _ORIENTATION_METHODS[value]:
        return value, None
    return None, f"EXIF Orientation 值越界（{value}，合法范围 1–8），已按原方向导出。"


def apply_orientation(im: Image.Image, orientation: int | None) -> Image.Image:
    """按 EXIF Orientation 把像素真正摆正（变换后的图不再携带语义方向）。"""
    if not orientation or orientation == 1:
        return im
    out = im
    for method in _ORIENTATION_METHODS[orientation]:
        out = out.transpose(method)
    return out


# ---------------------------------------------------------------- 模式规整 / 重编码

def normalize_mode(im: Image.Image) -> Image.Image:
    """规整为可无损保存 PNG、且保留透明语义的模式。

    - L / 1 带 tRNS 透明键值 -> LA（否则 fromarray 重建会丢透明）；
    - RGB / RGBA / LA / L / 1 原样保留（无透明键值时）；
    - P 调板：带透明索引 -> RGBA，否则 -> RGB；
    - 其余（CMYK / RGBa / I / F……）按是否含透明转 RGBA / RGB。

    结果像素经 PNG 编码再解码必须逐像素一致（PNG 无损）。
    """
    # 注意：L/1 必须先检查 tRNS 再走“原样保留”分支
    if im.mode in ("L", "1") and "transparency" in im.info:
        return im.convert("LA")
    if im.mode in _PRESERVE_MODES:
        return im
    if im.mode == "P":
        return im.convert("RGBA" if "transparency" in im.info else "RGB")
    try:
        has_alpha = bool(im.has_transparency_data)
    except Exception:
        has_alpha = "A" in im.getbands()
    return im.convert("RGBA" if has_alpha else "RGB")


def encode_clean_png(im: Image.Image) -> bytes:
    """以全新 PNG 重编码，不带入 EXIF / GPS / XMP / ICC / PNG 文本。

    关键：先 ``np.asarray`` 再 ``Image.fromarray`` 得到**不含任何
    ``info`` 元数据**的新图像对象——Pillow 保存 PNG 时会自动把
    ``info['icc_profile']`` 等写回文件，单纯不向 save() 传参并不够。
    """
    arr = np.asarray(im)
    clean = Image.fromarray(arr, im.mode)
    buf = io.BytesIO()
    clean.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


# ---------------------------------------------------------------- PNG 复查

def _iter_png_chunks(data: bytes) -> list[str]:
    """列出 PNG 字节中的全部 chunk 类型；结构损坏时抛 ValueError。"""
    if not data.startswith(_PNG_SIGNATURE):
        raise ValueError("不是合法的 PNG 签名")
    chunks: list[str] = []
    pos = len(_PNG_SIGNATURE)
    while pos + 8 <= len(data):
        (length,) = struct.unpack(">I", data[pos:pos + 4])
        ctype = data[pos + 4:pos + 8]
        try:
            chunks.append(ctype.decode("ascii"))
        except UnicodeDecodeError as e:
            raise ValueError(f"PNG chunk 类型非法：{ctype!r}") from e
        end = pos + 12 + length
        if end > len(data):
            raise ValueError(f"PNG chunk {ctype!r} 长度超出文件边界")
        if ctype == b"IEND":
            break
        pos = end
    else:
        raise ValueError("PNG 缺少 IEND 结束块")
    return chunks


def recheck_copy(
    png_data: bytes,
    expected_pixels: np.ndarray,
    oriented_size: tuple[int, int],
    source_has_alpha: bool,
) -> dict:
    """重新解析生成文件，逐项复查元数据清除与像素 / 透明通道一致性。

    返回结构化复查结果；任何硬性项失败时 ``passed=False`` 并在
    ``offenders`` 中给出中文原因（调用方据此拒绝下载）。
    """
    offenders: list[str] = []
    checks: dict[str, bool] = {}
    chunks: list[str] = []
    width = height = 0
    mode = ""
    has_alpha = False
    text_keys: list[str] = []
    info_keys: list[str] = []

    # 1) 字节级 chunk 扫描（最权威：即使 Pillow 解析静默忽略也能发现）
    try:
        chunks = _iter_png_chunks(png_data)
        checks["png_structure"] = True
    except ValueError as e:
        checks["png_structure"] = False
        offenders.append(f"生成文件不是结构合法的 PNG（{e}）")
        chunks = []

    forbidden_hit = sorted({c for c in chunks if c in _PNG_FORBIDDEN_CHUNKS})
    checks["no_exif_chunk"] = "eXIf" not in forbidden_hit
    checks["no_icc_chunk"] = "iCCP" not in forbidden_hit
    checks["no_text_chunks"] = not (
        {"tEXt", "zTXt", "iTXt"} & set(forbidden_hit))
    if forbidden_hit:
        offenders.append(
            "生成 PNG 仍含被禁元数据块：" + "、".join(forbidden_hit))

    reopened: Image.Image | None = None
    if checks["png_structure"]:
        try:
            reopened = Image.open(io.BytesIO(png_data))
            reopened.load()
            width, height = reopened.size
            mode = reopened.mode
            has_alpha = bool(reopened.has_transparency_data)
            checks["png_decodable"] = True
        except Exception as e:
            checks["png_decodable"] = False
            offenders.append(f"生成 PNG 无法重新解码（{type(e).__name__}）")

    exif_present = icc_info = xmp_present = text_info = False
    gps_present = False
    privacy: dict | None = None
    if reopened is not None:
        info_keys = sorted(reopened.info.keys())
        # 2) Pillow 再解析：info / text 中不允许出现任何元数据
        exif_info = "exif" in reopened.info
        icc_info = "icc_profile" in reopened.info
        text_keys = sorted(getattr(reopened, "text", {}).keys())
        text_info = bool(text_keys)
        xmp_present = any("xmp" in k.lower() for k in text_keys)
        checks["no_exif_chunk"] = checks["no_exif_chunk"] and not exif_info
        checks["no_icc_chunk"] = checks["no_icc_chunk"] and not icc_info
        checks["no_text_chunks"] = checks["no_text_chunks"] and not text_info
        if exif_info:
            offenders.append("重新解析仍可读到 EXIF（im.info['exif']）。")
        if icc_info:
            offenders.append("重新解析仍可读到 ICC 色彩配置。")
        if text_info:
            offenders.append(
                "重新解析仍可读到 PNG 文本元数据：" + "、".join(text_keys))

        # 3) 对生成文件再跑一次完整隐私解析：EXIF / GPS 必须全无
        try:
            privacy = analyze_privacy(png_data)
            exif_present = bool(privacy["exif_present"])
            gps_present = privacy["gps"] is not None
        except Exception as e:
            offenders.append(f"对生成文件重新执行隐私解析失败（{type(e).__name__}）。")
        checks["exif_gps_absent"] = not exif_present and not gps_present
        if exif_present:
            offenders.append("复查仍在生成文件中检测到 EXIF 数据。")
        if gps_present:
            offenders.append("复查仍在生成文件中检测到 GPS 信息。")
        checks["no_xmp"] = not xmp_present and "iTXt" not in forbidden_hit
        if xmp_present:
            offenders.append("复查仍在生成文件中检测到 XMP 文本块。")

        # 4) 尺寸 / 模式 / 透明通道 / 像素逐值一致性
        dims_ok = (width, height) == oriented_size
        checks["dimensions_match"] = dims_ok
        if not dims_ok:
            offenders.append(
                f"生成文件尺寸 {width}×{height} 与摆正后尺寸 "
                f"{oriented_size[0]}×{oriented_size[1]} 不一致。")
        alpha_ok = has_alpha == source_has_alpha
        checks["alpha_preserved"] = alpha_ok
        if not alpha_ok:
            offenders.append(
                f"透明通道未保留（原图含透明={source_has_alpha}，"
                f"副本含透明={has_alpha}）。")
        try:
            pixels_ok = np.array_equal(np.asarray(reopened), expected_pixels)
        except Exception:
            pixels_ok = False
        checks["pixels_lossless"] = pixels_ok
        if not pixels_ok:
            offenders.append("生成 PNG 重解码后像素与摆正后的原图不一致。")
    else:
        checks.update({
            "png_decodable": False, "exif_gps_absent": False,
            "no_xmp": False, "dimensions_match": False,
            "alpha_preserved": False, "pixels_lossless": False,
        })

    passed = not offenders
    return {
        "passed": passed,
        "format": "PNG" if checks["png_structure"] else None,
        "width": width,
        "height": height,
        "mode": mode,
        "has_alpha": has_alpha,
        "chunks": chunks,
        "info_keys": info_keys,
        "text_keys": text_keys,
        "exif_present": exif_present,
        "gps_present": gps_present,
        "icc_present": (not checks["no_icc_chunk"]),
        "xmp_present": (not checks["no_xmp"]),
        "text_metadata_present": (not checks["no_text_chunks"]),
        "checks": checks,
        "offenders": offenders,
        "reparsed_privacy": {
            "exif_present": (privacy or {}).get("exif_present"),
            "gps": (privacy or {}).get("gps"),
            "date_time": (privacy or {}).get("date_time"),
            "camera_make": (privacy or {}).get("camera_make"),
            "camera_model": (privacy or {}).get("camera_model"),
        } if privacy is not None else None,
    }


# ---------------------------------------------------------------- 入口

def export_privacy_copy(data: bytes) -> dict:
    """对上传的 JPEG/PNG 字节执行完整的隐私副本导出，返回结果字典。

    - 图片无法解码 / 非 PNG·JPEG -> PrivacyCheckError（调用方返 400）；
    - 摆正后超过 MAX_EXPORT_DIM -> PrivacyExportError（400）；
    - 复查不通过 -> PrivacyCopyRecheckError（调用方返 500，不给下载）。
    """
    im = open_supported_image(data)
    source_format = im.format
    source_width, source_height = im.size

    # 原图 EXIF 隐私字段（拍摄时间 / 相机 / GPS），复用只读检查逻辑
    privacy = analyze_privacy(data)
    warnings = list(privacy.get("warnings", []))

    orientation, orient_warn = read_orientation(im)
    if orient_warn:
        warnings.append(orient_warn)
    oriented = apply_orientation(im, orientation)

    normalized = normalize_mode(oriented)
    oriented_size = normalized.size
    if max(oriented_size) > MAX_EXPORT_DIM:
        raise PrivacyExportError(
            f"摆正后图片尺寸 {oriented_size[0]}×{oriented_size[1]} 超过 "
            f"{MAX_EXPORT_DIM}px 上限，请先缩小后再导出。")

    pixels = np.asarray(normalized)
    source_has_alpha = bool(normalized.has_transparency_data)
    png_data = encode_clean_png(normalized)

    context = {
        "source_format": source_format,
        "source_size": {"width": source_width, "height": source_height},
        "original": {
            "width": source_width,
            "height": source_height,
            "exif_present": privacy["exif_present"],
            "exif_ok": privacy["exif_ok"],
            "date_time": privacy["date_time"],
            "date_time_iso": privacy["date_time_iso"],
            "camera_make": privacy["camera_make"],
            "camera_model": privacy["camera_model"],
            "gps": privacy["gps"],
            "warnings": warnings,
        },
        "orientation": {
            "value": orientation,
            "applied": bool(orientation and orientation != 1),
            "label": (_ORIENTATION_LABELS.get(orientation)
                      if orientation else "无 EXIF Orientation，按原方向导出"),
        },
        "copy": {
            "format": "PNG",
            "width": oriented_size[0],
            "height": oriented_size[1],
            "mode": normalized.mode,
            "has_alpha": source_has_alpha,
            "transparency_preserved": source_has_alpha,
            "bytes": len(png_data),
            "chunks": [],
        },
    }

    recheck = recheck_copy(png_data, pixels, oriented_size, source_has_alpha)
    if not recheck["passed"]:
        raise PrivacyCopyRecheckError(recheck["offenders"], recheck, context)

    sha256 = hashlib.sha256(png_data).hexdigest()
    return {
        **context,
        "copy": {**context["copy"],
                 "sha256": sha256,
                 "chunks": recheck["chunks"]},
        "recheck": recheck,
        "download": {"media_type": "image/png",
                     "data_url": "data:image/png;base64,"
                     + base64.b64encode(png_data).decode("ascii")},
        "stored": False,
    }


__all__ = [
    "MAX_EXPORT_DIM",
    "PrivacyCheckError",
    "PrivacyExportError",
    "PrivacyCopyRecheckError",
    "read_orientation",
    "apply_orientation",
    "normalize_mode",
    "encode_clean_png",
    "recheck_copy",
    "export_privacy_copy",
]
