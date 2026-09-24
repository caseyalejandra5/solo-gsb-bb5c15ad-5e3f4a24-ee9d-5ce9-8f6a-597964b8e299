"""隐私副本导出（POST /api/privacy-export）端到端测试。

覆盖：
  - JPEG（完整 EXIF 拍摄时间/相机/GPS）-> 全新 PNG，EXIF/GPS 全部剥离
  - EXIF Orientation（6 = 顺时针 90°）像素被摆正，尺寸高/宽互换
  - PNG eXIf + 文本块（tEXt/iTXt-zTXt）/ ICC（iCCP）不进入副本
  - 透明 PNG（RGBA）的 alpha 通道被保留；不透明 JPEG 不产生多余 alpha
  - 副本字节重新解析复查：Pillow info/getexif 与 PNG 原始数据块扫描均为空
  - 调色板透明（P + tRNS）转 RGBA 保留透明
  - 灰度 / 位图模式可正常导出
  - 仅接受 PNG/JPEG；WebP、坏图、空文件返回 400；超 16MB 返回 413
  - 全程内存：不建任务目录、不写归档，水印任务数据不受影响
  - 损坏 EXIF 不阻断导出
  - 复查扫描器对含 eXIf/iCCP/tEXt/iTXt/zTXt 的 PNG 能如实报警

运行：
    python -m pytest tests/test_privacy_export.py -q
或：
    python tests/test_privacy_export.py
"""
import base64
import io
import os
import struct
import sys
import tempfile
import zlib
from pathlib import Path

import numpy as np
from PIL import Image, PngImagePlugin

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_privacy import (  # noqa: E402
    ExifBuilder,
    _full_exif,
    _inject_app1,
    _jpeg_with_app1,
    _plain_jpeg,
    _png_with_exif,
)


ASCII_TYPE = 2
IFD_EXIF = 0x8769


# ---------------------------------------------------------------- 构造工具

def _fresh_client():
    tmp = tempfile.mkdtemp(prefix="wm-privacy-export-")
    os.environ["WM_DATA_DIR"] = tmp
    os.environ["WM_MAX_JOBS"] = "200"
    sys.modules.pop("app.main", None)
    from fastapi.testclient import TestClient
    from app.main import app, DATA_DIR

    return TestClient(app), Path(DATA_DIR)


def _oriented_jpeg(orientation: int, size: int = 64) -> bytes:
    """带 EXIF Orientation 标签的 JPEG。"""
    b = ExifBuilder()
    # 0x0112 = Orientation，SHORT
    b.set(0, 0x0112, 3, orientation)
    b.set(0, 0x010F, ASCII_TYPE, "AcmeCam")
    b.set(IFD_EXIF, 0x9003, ASCII_TYPE, "2026:09:23 12:34:56")
    return _jpeg_with_app1(b.build_app1(), size=size)


def _rgba_png(size: int = 48) -> bytes:
    rng = np.random.RandomState(11)
    arr = rng.randint(0, 255, (size, size, 4), np.uint8)
    arr[..., 3] = np.where(np.indices((size, size))[0] < size // 2, 0, 255)
    buf = io.BytesIO()
    Image.fromarray(arr, "RGBA").save(buf, format="PNG")
    return buf.getvalue()


def _png_with_text_chunks() -> bytes:
    arr = np.random.RandomState(5).randint(0, 255, (32, 32, 3), np.uint8)
    meta = PngImagePlugin.PngInfo()
    meta.add_text("Title", "a private title")
    meta.add_text("Comment", "secret comment")
    meta.add_itxt("XML:com.adobe.xmp",
                  '<x:xmpmeta xmlns:x="adobe:ns:meta/"><rdf:RDF/></x:xmpmeta>')
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG", pnginfo=meta)
    return buf.getvalue()


def _png_with_icc() -> tuple[bytes, bytes]:
    # 用内置 sRGB profile 构造带 iCCP 数据块的 PNG
    from PIL import ImageCms

    img = Image.new("RGB", (16, 16), (10, 200, 40))
    icc_bytes = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()
    buf = io.BytesIO()
    img.save(buf, format="PNG", icc_profile=icc_bytes)
    return buf.getvalue(), icc_bytes


def _palette_transparent_png() -> bytes:
    img = Image.new("P", (24, 24))
    palette = []
    for i in range(256):
        palette += [i, (255 - i) % 256, (i * 7) % 256]
    img.putpalette(palette)
    img.info["transparency"] = bytes([0] + [255] * 255)  # 索引 0 全透明
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _inject_png_chunk(png: bytes, cid: bytes, payload: bytes,
                      after_idat: bool = False) -> bytes:
    """在 PNG 中插入任意数据块（默认紧跟签名；after_idat=True 放在 IEND 前）。"""
    chunk = (struct.pack(">I", len(payload)) + cid + payload
             + struct.pack(">I", zlib.crc32(cid + payload)))
    if not after_idat:
        return png[:8] + chunk + png[8:]
    # 找到 IEND
    iend = png.rfind(b"IEND")
    insert_at = iend - 4
    return png[:insert_at] + chunk + png[insert_at:]


def _decode_copy(d: dict) -> Image.Image:
    raw = base64.b64decode(d["copy"]["png_base64"])
    im = Image.open(io.BytesIO(raw))
    im.load()
    return im


# ---------------------------------------------------------------- 用例

def test_jpeg_full_exif_exported_as_clean_png():
    client, data_dir = _fresh_client()
    jpg = _jpeg_with_app1(_full_exif())
    r = client.post("/api/privacy-export",
                    files={"image": ("photo.jpg", jpg, "image/jpeg")})
    assert r.status_code == 200, r.text
    d = r.json()

    # 原图识别到的隐私字段照常返回
    src = d["source"]
    assert src["date_time"] == "2026:09:23 12:34:56"
    assert src["camera_make"] == "AcmeCam"
    assert src["camera_model"] == "Model X-100"
    assert src["gps"]["latitude"] == 31.2625

    copy = d["copy"]
    assert copy["format"] == "PNG"
    assert copy["recheck"]["clean"] is True
    assert copy["recheck"]["raw_chunk_hits"] == []
    assert copy["recheck"]["parsed_hits"] == []
    assert copy["alpha_preserved"] is False
    assert len(copy["sha256"]) == 64

    im = _decode_copy(d)
    assert im.format == "PNG"
    assert im.mode in ("RGB", "L")
    assert not im.info.get("exif")
    assert not dict(im.getexif())
    assert "icc_profile" not in im.info
    assert "XML:com.adobe.xmp" not in im.info
    assert "comment" not in im.info

    # base64 字节与 sha256/长度自洽
    raw = base64.b64decode(copy["png_base64"])
    assert len(raw) == copy["bytes"]
    import hashlib
    assert hashlib.sha256(raw).hexdigest() == copy["sha256"]

    # 不建任务、不写归档
    assert not list(data_dir.glob("*/meta.json"))
    assert not (data_dir / "archive.log.jsonl").exists()
    assert d["stored"] is False


def test_exif_orientation_6_pixels_are_transposed():
    client, _ = _fresh_client()

    # 矩形图（40 高 × 80 宽）+ Orientation 6（顺时针 90°）：
    # 摆正后像素尺寸应为 80 高 × 40 宽，且与 ImageOps.exif_transpose 一致
    from PIL import ImageOps

    arr = np.random.RandomState(3).randint(0, 255, (40, 80, 3), np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="JPEG")
    b = ExifBuilder()
    b.set(0, 0x0112, 3, 6)
    b.set(0, 0x010F, ASCII_TYPE, "AcmeCam")
    jpg = _inject_app1(buf.getvalue(), b.build_app1())

    r = client.post("/api/privacy-export",
                    files={"image": ("rot.jpg", jpg, "image/jpeg")})
    assert r.status_code == 200, r.text
    d = r.json()

    assert d["orientation"]["tag"] == 6
    assert d["orientation"]["applied"] is True
    assert "顺时针" in d["orientation"]["label"]
    c = d["copy"]
    assert (c["width"], c["height"]) == (40, 80)
    assert c["recheck"]["clean"] is True

    im = _decode_copy(d)
    ref = ImageOps.exif_transpose(Image.open(io.BytesIO(jpg))).convert("RGB")
    assert np.array_equal(np.array(im), np.array(ref))
    assert not dict(im.getexif())


def test_orientation_normal_not_applied():
    client, _ = _fresh_client()
    r = client.post("/api/privacy-export",
                    files={"image": ("plain.jpg", _plain_jpeg(), "image/jpeg")})
    assert r.status_code == 200
    d = r.json()
    assert d["orientation"]["tag"] == 1
    assert d["orientation"]["applied"] is False


def test_rgba_transparency_preserved():
    client, _ = _fresh_client()
    png = _rgba_png()
    r = client.post("/api/privacy-export",
                    files={"image": ("alpha.png", png, "image/png")})
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["copy"]["alpha_preserved"] is True
    im = _decode_copy(d)
    assert im.mode == "RGBA"
    arr = np.array(im)
    assert arr.shape[2] == 4
    # 上半部透明、下半部不透明被保留
    assert (arr[: arr.shape[0] // 2, :, 3] == 0).all()
    assert (arr[arr.shape[0] // 2:, :, 3] == 255).all()
    assert d["copy"]["recheck"]["clean"] is True


def test_palette_transparency_preserved():
    client, _ = _fresh_client()
    r = client.post("/api/privacy-export",
                    files={"image": ("p.png", _palette_transparent_png(),
                                     "image/png")})
    assert r.status_code == 200, r.text
    d = r.json()
    im = _decode_copy(d)
    assert im.mode == "RGBA"
    assert d["copy"]["alpha_preserved"] is True
    assert np.array(im)[0, 0, 3] == 0


def test_png_text_chunks_stripped():
    client, _ = _fresh_client()
    png = _png_with_text_chunks()
    # 输入确含 tEXt 与 iTXt(XMP)
    im_in = Image.open(io.BytesIO(png))
    assert im_in.info.get("Title") == "a private title"
    assert "XML:com.adobe.xmp" in im_in.info

    r = client.post("/api/privacy-export",
                    files={"image": ("meta.png", png, "image/png")})
    assert r.status_code == 200, r.text
    d = r.json()
    raw = base64.b64decode(d["copy"]["png_base64"])
    assert b"private title" not in raw
    assert b"xmpmeta" not in raw and b"XMP" not in raw
    im = Image.open(io.BytesIO(raw))
    assert im.info.get("Title") is None
    assert "XML:com.adobe.xmp" not in im.info
    assert d["copy"]["recheck"]["clean"] is True


def test_png_exif_chunk_stripped():
    client, _ = _fresh_client()
    png = _png_with_exif(_full_exif())
    r = client.post("/api/privacy-export",
                    files={"image": ("exif.png", png, "image/png")})
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["source"]["camera_make"] == "AcmeCam"
    raw = base64.b64decode(d["copy"]["png_base64"])
    assert b"eXIf" not in raw
    assert b"AcmeCam" not in raw
    assert d["copy"]["recheck"]["clean"] is True


def test_png_icc_profile_stripped():
    client, _ = _fresh_client()
    png, _icc_bytes = _png_with_icc()
    # 输入确含 iCCP 数据块
    assert b"iCCP" in png
    r = client.post("/api/privacy-export",
                    files={"image": ("icc.png", png, "image/png")})
    assert r.status_code == 200, r.text
    d = r.json()
    im = _decode_copy(d)
    assert "icc_profile" not in im.info
    raw = base64.b64decode(d["copy"]["png_base64"])
    assert b"iCCP" not in raw
    assert d["copy"]["recheck"]["clean"] is True


def test_grayscale_and_bitmap_supported():
    client, _ = _fresh_client()
    for mode in ("L", "1"):
        buf = io.BytesIO()
        Image.new(mode, (20, 16), 128 if mode == "L" else 1).save(buf,
                                                                   format="PNG")
        r = client.post("/api/privacy-export",
                        files={"image": (f"{mode}.png", buf.getvalue(),
                                         "image/png")})
        assert r.status_code == 200, r.text
        d = r.json()
        im = _decode_copy(d)
        assert im.mode == mode
        assert d["copy"]["recheck"]["clean"] is True


def test_corrupt_exif_does_not_block_export():
    client, _ = _fresh_client()
    payload = b"Exif\x00\x00" + b"\xff" * 100
    jpg = _jpeg_with_app1(payload)
    r = client.post("/api/privacy-export",
                    files={"image": ("broken.jpg", jpg, "image/jpeg")})
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["copy"]["recheck"]["clean"] is True
    im = _decode_copy(d)
    assert not dict(im.getexif())


def test_unsupported_unreadable_empty_rejected():
    client, _ = _fresh_client()
    buf = io.BytesIO()
    Image.new("RGB", (16, 16)).save(buf, format="WEBP")
    r = client.post("/api/privacy-export",
                    files={"image": ("a.webp", buf.getvalue(), "image/webp")})
    assert r.status_code == 400
    assert "PNG / JPEG" in r.json()["detail"]

    r = client.post("/api/privacy-export",
                    files={"image": ("fake.jpg", b"not-an-image" * 8,
                                     "image/jpeg")})
    assert r.status_code == 400

    r = client.post("/api/privacy-export",
                    files={"image": ("empty.png", b"", "image/png")})
    assert r.status_code == 400


def test_oversized_upload_rejected():
    client, _ = _fresh_client()
    # 超过 16MB 上限（JPEG 允许，仅测体积闸口）
    big = b"\xff\xd8\xff\xe0" + b"\x00" * (16 * 1024 * 1024 + 10)
    r = client.post("/api/privacy-export",
                    files={"image": ("big.jpg", big, "image/jpeg")})
    assert r.status_code == 413


def test_no_jobs_or_archive_created_across_requests():
    client, data_dir = _fresh_client()
    samples = [
        ("a.jpg", _jpeg_with_app1(_full_exif()), "image/jpeg"),
        ("b.png", _rgba_png(), "image/png"),
        ("c.png", _png_with_text_chunks(), "image/png"),
        ("d.jpg", _plain_jpeg(), "image/jpeg"),
    ]
    for name, blob, ctype in samples:
        r = client.post("/api/privacy-export",
                        files={"image": (name, blob, ctype)})
        assert r.status_code == 200, r.text
    assert not list(data_dir.glob("*"))
    assert not (data_dir / "archive.log.jsonl").exists()


def test_existing_privacy_check_still_works():
    client, _ = _fresh_client()
    r = client.post("/api/privacy-check",
                    files={"image": ("p.jpg",
                                     _jpeg_with_app1(_full_exif()),
                                     "image/jpeg")})
    assert r.status_code == 200
    d = r.json()
    assert d["camera_make"] == "AcmeCam"
    assert d["gps"]["longitude"] == 121.5


def test_scan_png_metadata_flags_forbidden_chunks():
    """复查扫描器对含禁止数据块的 PNG 必须如实报警（单元级）。"""
    from app import exifcheck

    # 干净 PNG：无命中
    clean = _rgba_png()
    assert exifcheck.scan_png_metadata(clean) == []
    assert exifcheck.recheck_clean_png(clean)["clean"] is True

    # 注入 eXIf / tEXt / iCCP / iTXt / zTXt
    cases = {
        b"eXIf": b"Exif\x00\x00" + b"\x00" * 20,
        b"tEXt": b"Comment\x00secret",
        b"iCCP": b"ICC Profile\x00\x00" + zlib.compress(b"fake-icc"),
        b"iTXt": b"XML:com.adobe.xmp\x00\x00\x00\x00\x00<x:xmpmeta/>",
    }
    for cid, payload in cases.items():
        injected = _inject_png_chunk(clean, cid, payload)
        hits = exifcheck.scan_png_metadata(injected)
        assert cid.decode() in hits, (cid, hits)
        assert exifcheck.recheck_clean_png(injected)["clean"] is False

    # zTXt：keyword + 压缩方法 + 压缩文本
    ztxt = (b"Comment\x00\x00"
            + zlib.compress(b"compressed secret"))
    injected = _inject_png_chunk(clean, b"zTXt", ztxt)
    assert "zTXt" in exifcheck.scan_png_metadata(injected)

    # 非 PNG
    assert exifcheck.recheck_clean_png(_plain_jpeg())["clean"] is False


def test_recheck_failure_raises_export_error():
    """build_privacy_copy 的复查闸口：扫描器一旦报警即抛错、不返回副本。"""
    from app import exifcheck

    # monkeypatch 扫描器模拟“重编码后仍残留元数据”的极端故障
    orig = exifcheck.scan_png_metadata
    exifcheck.scan_png_metadata = lambda data: ["eXIf"]
    try:
        try:
            exifcheck.build_privacy_copy(_plain_jpeg())
        except exifcheck.PrivacyExportError as e:
            assert "复查" in str(e)
        else:
            raise AssertionError("复查失败时必须抛出 PrivacyExportError")
    finally:
        exifcheck.scan_png_metadata = orig


if __name__ == "__main__":
    import inspect
    fns = [v for k, v in sorted(globals().items())
           if k.startswith("test_") and inspect.isfunction(v)]
    for fn in fns:
        fn()
        print(f"[PASS] {fn.__name__}")
    print(f"\n{len(fns)} 个隐私副本导出测试全部通过 ✔")
