"""图片隐私检查（EXIF）端到端测试。

覆盖：
  - 完整 EXIF（拍摄时间 / 相机品牌型号 / GPS 度分秒还原带符号十进制度）
  - PNG eXIf 与无 EXIF 图片
  - 字段缺失显示为 None（前端显示“无”）
  - 损坏 / 无法解码的 EXIF 不阻断隐私检查
  - 损坏 EXIF 不阻断水印嵌入 / 检测任务
  - 仅接受 PNG/JPEG，不支持的格式与坏图返回 400
  - 检查不写盘、不建任务、不影响归档

运行：
    python -m pytest tests/test_privacy.py -q
或：
    python tests/test_privacy.py
"""
import io
import os
import struct
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ---------------------------------------------------------------- EXIF 构造

SHORT, LONG, RATIONAL, ASCII_TYPE = 3, 4, 5, 2
IFD_EXIF, IFD_GPS = 0x8769, 0x8825


class ExifBuilder:
    """最小可用 Exif APP1 构造器：IFD0 + Exif sub-IFD + GPS IFD（小端 TIFF）。"""

    def __init__(self):
        self.ifds = {0: [], IFD_EXIF: [], IFD_GPS: []}

    def set(self, ifd, tag, typ, value):
        self.ifds[ifd].append((tag, typ, value))

    def _enc(self, typ, value):
        if typ == ASCII_TYPE:
            b = value.encode("ascii", "replace") if isinstance(value, str) else bytes(value)
            if not b.endswith(b"\x00"):
                b += b"\x00"
            return b
        if typ == RATIONAL:
            return b"".join(struct.pack("<II", n, d) for n, d in value)
        if typ == SHORT:
            vals = value if isinstance(value, (list, tuple)) else [value]
            return b"".join(struct.pack("<H", int(v)) for v in vals)
        if typ == LONG:
            vals = value if isinstance(value, (list, tuple)) else [value]
            return b"".join(struct.pack("<I", int(v)) for v in vals)
        raise ValueError(typ)

    def _count(self, typ, value):
        """TIFF 条目计数：ASCII=字节数，RATIONAL/SHORT/LONG=元素个数。"""
        if typ == ASCII_TYPE:
            return len(self._enc(typ, value))
        if typ == RATIONAL:
            return len(value)
        if typ in (SHORT, LONG):
            return len(value) if isinstance(value, (list, tuple)) else 1
        raise ValueError(typ)

    def build_tiff(self):
        if self.ifds[IFD_EXIF]:
            self.ifds[0].append((IFD_EXIF, LONG, 0))
        if self.ifds[IFD_GPS]:
            self.ifds[0].append((IFD_GPS, LONG, 0))
        ents = {k: sorted(v) for k, v in self.ifds.items()}
        order = [0, IFD_EXIF, IFD_GPS]

        def ifd_size(k):
            return 2 + 12 * len(ents[k]) + 4

        # 1) 三个 IFD 连续排布，先算出各自绝对偏移
        off = {0: 8}
        p = 8 + ifd_size(0)
        off[IFD_EXIF], p = p, p + ifd_size(IFD_EXIF)
        off[IFD_GPS], p = p, p + ifd_size(IFD_GPS)
        data_start = p

        # 2) 规划各 IFD 外置数据（>4 字节）在数据区中的偏移
        data_base, cur = {}, data_start
        for k in order:
            data_base[k] = cur
            for tag, typ, value in ents[k]:
                if tag in (IFD_EXIF, IFD_GPS):
                    continue
                if len(self._enc(typ, value)) > 4:
                    cur += len(self._enc(typ, value))
                    if cur % 2:
                        cur += 1

        tiff = b"II\x2a\x00" + struct.pack("<I", 8)
        for k in order:
            body = struct.pack("<H", len(ents[k]))
            dcur = data_base[k]
            for tag, typ, value in ents[k]:
                if tag in (IFD_EXIF, IFD_GPS):
                    body += (struct.pack("<HHI", tag, LONG, 1)
                             + struct.pack("<I", off[tag]))
                    continue
                raw = self._enc(typ, value)
                if len(raw) <= 4:
                    field = raw + b"\x00" * (4 - len(raw))
                else:
                    field = struct.pack("<I", dcur)
                    dcur += len(raw)
                    if dcur % 2:
                        dcur += 1
                body += struct.pack("<HHI", tag, typ, self._count(typ, value)) + field
            body += struct.pack("<I", 0)
            tiff += body
        for k in order:
            for tag, typ, value in ents[k]:
                if tag in (IFD_EXIF, IFD_GPS):
                    continue
                raw = self._enc(typ, value)
                if len(raw) > 4:
                    tiff += raw
                    if len(tiff) % 2:
                        tiff += b"\x00"
        return tiff

    def build_app1(self):
        return b"Exif\x00\x00" + self.build_tiff()


def _full_exif(
    make="AcmeCam",
    model="Model X-100",
    dt="2026:09:23 12:34:56",
    lat=((31, 1), (15, 1), (4500, 100)),
    lat_ref="N",
    lon=((121, 1), (30, 1), (0, 1)),
    lon_ref="E",
):
    b = ExifBuilder()
    if make is not None:
        b.set(0, 0x010F, ASCII_TYPE, make)
    if model is not None:
        b.set(0, 0x0110, ASCII_TYPE, model)
    if dt is not None:
        b.set(IFD_EXIF, 0x9003, ASCII_TYPE, dt)  # DateTimeOriginal
    if lat is not None:
        b.set(IFD_GPS, 0x0002, RATIONAL, lat)
    if lat_ref is not None:
        b.set(IFD_GPS, 0x0001, ASCII_TYPE, lat_ref)
    if lon is not None:
        b.set(IFD_GPS, 0x0004, RATIONAL, lon)
    if lon_ref is not None:
        b.set(IFD_GPS, 0x0003, ASCII_TYPE, lon_ref)
    return b.build_app1()


def _plain_jpeg(size=96) -> bytes:
    arr = np.random.RandomState(1).randint(0, 255, (size, size, 3), np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="JPEG", quality=90)
    return buf.getvalue()


def _inject_app1(jpg: bytes, payload: bytes) -> bytes:
    seg = b"\xff\xe1" + struct.pack(">H", len(payload) + 2) + payload
    return jpg[:2] + seg + jpg[2:]


def _jpeg_with_app1(app1: bytes, size=96) -> bytes:
    return _inject_app1(_plain_jpeg(size), app1)


def _png_with_exif(app1: bytes, size=96) -> bytes:
    arr = np.random.RandomState(2).randint(0, 255, (size, size, 3), np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG", exif=app1[6:])  # 去掉 "Exif\0\0"
    return buf.getvalue()


def _fresh_client():
    tmp = tempfile.mkdtemp(prefix="wm-privacy-")
    os.environ["WM_DATA_DIR"] = tmp
    os.environ["WM_MAX_JOBS"] = "200"
    for mod in ("app.main",):
        sys.modules.pop(mod, None)
    from fastapi.testclient import TestClient
    from app.main import app, DATA_DIR

    return TestClient(app), Path(DATA_DIR)


# ---------------------------------------------------------------- 用例

def test_full_exif_jpeg_dms_conversion():
    client, _ = _fresh_client()
    # 31°15'45"N = 31.2625000 ；121°30'00"E = 121.5
    jpg = _jpeg_with_app1(_full_exif())
    r = client.post(
        "/api/privacy-check",
        files={"image": ("photo.jpg", jpg, "image/jpeg")},
    )
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["format"] == "JPEG"
    assert d["exif_present"] is True
    assert d["date_time"] == "2026:09:23 12:34:56"
    assert d["date_time_iso"] == "2026-09-23T12:34:56"
    assert d["camera_make"] == "AcmeCam"
    assert d["camera_model"] == "Model X-100"
    assert d["gps"]["latitude"] == 31.2625
    assert d["gps"]["longitude"] == 121.5
    assert d["gps"]["latitude_ref"] == "N"
    assert d["gps"]["longitude_ref"] == "E"
    assert "45″" in d["gps"]["latitude_dms"]
    assert d["warnings"] == []
    assert d["stored"] is False


def test_signed_southern_western_hemisphere():
    client, _ = _fresh_client()
    app1 = _full_exif(
        lat=((33, 1), (51, 1), (543, 10)), lat_ref="S",
        lon=((151, 1), (12, 1), (0, 1)), lon_ref="W",
    )
    d = client.post("/api/privacy-check",
                    files={"image": ("sydney.jpg",
                                     _jpeg_with_app1(app1),
                                     "image/jpeg")}).json()
    # 33°51'54.3"S
    assert d["gps"]["latitude"] == round(-(33 + 51 / 60 + 54.3 / 3600), 7)
    assert d["gps"]["longitude"] == -(151 + 12 / 60)
    assert d["gps"]["latitude"] < 0 and d["gps"]["longitude"] < 0


def test_png_exif_supported():
    client, _ = _fresh_client()
    png = _png_with_exif(_full_exif())
    r = client.post("/api/privacy-check",
                    files={"image": ("p.png", png, "image/png")})
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["format"] == "PNG"
    assert d["camera_make"] == "AcmeCam"
    assert d["gps"]["latitude"] == 31.2625


def test_plain_image_all_fields_missing():
    client, _ = _fresh_client()
    d = client.post("/api/privacy-check",
                    files={"image": ("plain.jpg", _plain_jpeg(),
                                     "image/jpeg")}).json()
    assert d["exif_present"] is False
    assert d["date_time"] is None
    assert d["date_time_iso"] is None
    assert d["camera_make"] is None
    assert d["camera_model"] is None
    assert d["gps"] is None
    assert d["warnings"] == []


def test_partial_fields_and_partial_gps():
    client, _ = _fresh_client()
    # 只有品牌，没有型号 / 时间；GPS 只有纬度没有经度
    app1 = _full_exif(model=None, dt=None, lon=None, lon_ref=None)
    d = client.post("/api/privacy-check",
                    files={"image": ("x.jpg",
                                     _jpeg_with_app1(app1),
                                     "image/jpeg")}).json()
    assert d["camera_make"] == "AcmeCam"
    assert d["camera_model"] is None
    assert d["date_time"] is None
    assert d["gps"]["latitude"] == 31.2625
    assert d["gps"]["longitude"] is None
    assert any("GPS 信息不完整" in w for w in d["warnings"])


def test_corrupt_exif_does_not_block_check():
    client, _ = _fresh_client()
    # Exif 头之后是非法 TIFF 头：getexif 会抛 SyntaxError / struct.error
    payload = b"Exif\x00\x00" + b"\xff" * 100
    jpg = _jpeg_with_app1(payload)
    r = client.post("/api/privacy-check",
                    files={"image": ("broken.jpg", jpg, "image/jpeg")})
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["format"] == "JPEG"                 # 图片本身仍成功解码
    assert d["exif_ok"] is False
    assert d["camera_make"] is None
    assert d["gps"] is None
    assert any("损坏" in w or "无法解码" in w for w in d["warnings"])

    # 完全不是 TIFF 的乱码 APP1（部分 Pillow 版本静默返回空，不应报 500）
    payload2 = b"XXXX" + b"\xab" * 60
    r2 = client.post("/api/privacy-check",
                     files={"image": ("broken2.jpg",
                                      _jpeg_with_app1(payload2),
                                      "image/jpeg")})
    assert r2.status_code == 200
    assert r2.json()["format"] == "JPEG"


def _make_wm_png(size=384) -> bytes:
    """与 tests/test_reports 一致的可嵌入 / 可检测测试图。"""
    rng = np.random.RandomState(7)
    y, x = np.mgrid[0:size, 0:size]
    base = (
        128
        + 40 * np.sin(2 * np.pi * x / 130)
        + 30 * np.cos(2 * np.pi * y / 95)
        + rng.normal(0, 12, (size, size))
    ).clip(0, 255).astype(np.uint8)
    rgb = np.stack([base, (base * 0 + 128), (255 - base)], axis=-1)
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="PNG")
    return buf.getvalue()


def _inject_png_exif(png: bytes, exif_payload: bytes) -> bytes:
    """向 PNG 注入一个 eXIf 数据块（payload 可合法也可损坏）。"""
    import zlib
    chunk = (struct.pack(">I", len(exif_payload)) + b"eXIf" + exif_payload
             + struct.pack(">I", zlib.crc32(b"eXIf" + exif_payload)))
    return png[:8] + chunk + png[8:]


def test_corrupt_exif_does_not_block_embed_and_detect():
    client, data_dir = _fresh_client()
    # 损坏的 eXIf 块：Exif 头之后是非法 TIFF，getexif() 会抛错
    bad_png = _inject_png_exif(
        _make_wm_png(), b"Exif\x00\x00" + b"\xff" * 100)

    r = client.post(
        "/api/embed",
        files={"image": ("rot.png", bad_png, "image/png")},
        data={"text": "EXIF不阻断", "strength": "medium"},
    )
    assert r.status_code == 200, r.text
    job_id = r.json()["job_id"]

    # 走标准流程：攻击 -> 检测（无损原图的盲检测本身不在算法设计场景内）
    r = client.post("/api/attack",
                    json={"job_id": job_id, "jpeg_quality": 70,
                          "crop_fraction": 0.75})
    assert r.status_code == 200, r.text
    r = client.post("/api/detect", data={"job_id": job_id})
    assert r.status_code == 200, r.text
    assert r.json()["detected"] is True

    # 独立上传损坏 EXIF 的 PNG 同样不报错、正常建检测任务
    r = client.post("/api/detect",
                    files={"image": ("rot2.png", bad_png, "image/png")})
    assert r.status_code == 200, r.text
    assert r.json()["job_id"] != job_id

    # 隐私检查接口对同一张损坏 EXIF 图片返回 200 与警告
    r = client.post("/api/privacy-check",
                    files={"image": ("rot3.png", bad_png, "image/png")})
    assert r.status_code == 200
    d = r.json()
    assert d["format"] == "PNG"
    assert d["exif_ok"] is False
    assert any("损坏" in w or "无法解码" in w for w in d["warnings"])


def test_unsupported_and_unreadable_images():
    client, _ = _fresh_client()
    # WebP 不允许
    buf = io.BytesIO()
    Image.new("RGB", (32, 32)).save(buf, format="WEBP")
    r = client.post("/api/privacy-check",
                    files={"image": ("a.webp", buf.getvalue(), "image/webp")})
    assert r.status_code == 400
    assert "PNG / JPEG" in r.json()["detail"]

    # 完全无法解码的字节
    r = client.post("/api/privacy-check",
                    files={"image": ("fake.jpg", b"not-an-image" * 8,
                                     "image/jpeg")})
    assert r.status_code == 400

    # 空文件
    r = client.post("/api/privacy-check",
                    files={"image": ("empty.png", b"", "image/png")})
    assert r.status_code == 400


def test_no_persistence_no_job_created():
    client, data_dir = _fresh_client()
    n_before = len(list(data_dir.glob("*"))) if data_dir.exists() else 0
    jpg = _jpeg_with_app1(_full_exif())
    for _ in range(3):
        r = client.post("/api/privacy-check",
                        files={"image": ("p.jpg", jpg, "image/jpeg")})
        assert r.status_code == 200
    n_after = len(list(data_dir.glob("*"))) if data_dir.exists() else 0
    # 隐私检查不创建任务目录、不写任何文件
    assert n_after == n_before
    assert not list(data_dir.glob("*/meta.json"))
    # 归档日志也不受影响
    assert not (data_dir / "archive.log.jsonl").exists()


def test_datetime_fallback_to_datetime_tag():
    client, _ = _fresh_client()
    b = ExifBuilder()
    b.set(0, 0x0132, ASCII_TYPE, "2025:01:02 03:04:05")  # DateTime（非 Original）
    d = client.post("/api/privacy-check",
                    files={"image": ("d.jpg",
                                     _jpeg_with_app1(b.build_app1()),
                                     "image/jpeg")}).json()
    assert d["date_time"] == "2025:01:02 03:04:05"
    assert d["date_time_iso"] == "2025-01-02T03:04:05"


if __name__ == "__main__":
    import inspect
    fns = [v for k, v in sorted(globals().items())
           if k.startswith("test_") and inspect.isfunction(v)]
    for fn in fns:
        fn()
        print(f"[PASS] {fn.__name__}")
    print(f"\n{len(fns)} 个隐私检查测试全部通过 ✔")
