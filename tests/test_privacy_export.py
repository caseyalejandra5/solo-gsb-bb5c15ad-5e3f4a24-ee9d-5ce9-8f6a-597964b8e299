"""导出隐私副本端到端测试。

覆盖：
  - JPEG / PNG 上传后按 EXIF Orientation 摆正像素（1–8 与 Pillow 参照一致）
  - 全新 PNG 重编码：不含 EXIF / GPS / XMP / ICC / PNG 文本元数据
  - 透明通道保留（RGBA / LA / P 带透明索引 / L 带 tRNS）
  - 导出前重新解析生成文件：chunk 扫描 + Pillow 再解析 + 像素无损
  - 原图拍摄时间 / 相机 / GPS 与只读隐私检查结果一致
  - 复查不通过 -> HTTP 500、不提供下载数据
  - 尺寸超限 / 非 PNG·JPEG / 坏图 / 空文件 / 超 16MB
  - 全程内存：不建任务目录、不写任何文件、不影响归档
  - 现有只读 /api/privacy-check 行为不变

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
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps, PngImagePlugin

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_privacy import (  # noqa: E402
    _full_exif, _jpeg_with_app1, _plain_jpeg, _png_with_exif,
)


def _fresh_client():
    tmp = tempfile.mkdtemp(prefix="wm-privacy-export-")
    os.environ["WM_DATA_DIR"] = tmp
    os.environ["WM_MAX_JOBS"] = "200"
    sys.modules.pop("app.main", None)
    from fastapi.testclient import TestClient
    from app.main import app, DATA_DIR
    return TestClient(app), Path(DATA_DIR)


# ---------------------------------------------------------------- 构造工具

def _oriented_jpeg(orientation, size=(120, 160), seed=5):
    """保存带指定 EXIF Orientation 的 JPEG（含时间/相机字段）。"""
    h, w = size
    arr = np.random.RandomState(seed).randint(0, 255, (h, w, 3), np.uint8)
    exif = Image.Exif()
    exif[271] = "AcmeCam"
    exif[272] = "Model X-100"
    exif[36867] = "2026:09:23 12:34:56"
    exif[274] = int(orientation)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="JPEG", quality=92, exif=exif)
    return buf.getvalue()


def _rgba_png(size=(100, 120), with_meta=True, orientation=None):
    h, w = size
    rgba = np.zeros((h, w, 4), np.uint8)
    rgba[..., 0] = 200
    rgba[..., 1] = 110
    yy, xx = np.indices((h, w))
    rgba[..., 3] = np.where(yy < h // 2, 0, 255).astype("uint8")
    buf = io.BytesIO()
    kwargs = {"format": "PNG"}
    if with_meta:
        info = PngImagePlugin.PngInfo()
        info.add_text("Comment", "SECRET-COMMENT")
        info.add_text("Software", "EvilSoft")
        info.add_itxt("XML:com.adobe.xmp", "<xmp>SECRET</xmp>")
        kwargs["pnginfo"] = info
        kwargs["icc_profile"] = b"FAKE-ICC-PROFILE-DATA" * 4
        if orientation is not None:
            exif = Image.Exif()
            exif[274] = int(orientation)
            kwargs["exif"] = exif.tobytes()
    elif orientation is not None:
        exif = Image.Exif()
        exif[274] = int(orientation)
        kwargs["exif"] = exif.tobytes()
    Image.fromarray(rgba).save(buf, **kwargs)
    return buf.getvalue(), rgba


def _png_chunks(data):
    pos, out = 8, []
    while pos < len(data):
        ln = struct.unpack(">I", data[pos:pos + 4])[0]
        out.append(data[pos + 4:pos + 8].decode("latin1"))
        pos += 12 + ln
    return out


def _decode_copy(d):
    return base64.b64decode(d["download"]["data_url"].split(",", 1)[1])


# ---------------------------------------------------------------- 用例

def test_jpeg_orientation_all_eight_match_pillow():
    """8 个方向导出的像素/尺寸必须与 Pillow ImageOps.exif_transpose 一致。"""
    client, _ = _fresh_client()
    h, w = 80, 120
    for orient in range(1, 9):
        jpg = _oriented_jpeg(orient, size=(h, w), seed=orient)
        ref = ImageOps.exif_transpose(Image.open(io.BytesIO(jpg)))
        r = client.post(
            "/api/privacy-export",
            files={"image": (f"o{orient}.jpg", jpg, "image/jpeg")})
        assert r.status_code == 200, r.text
        d = r.json()
        assert d["orientation"]["value"] == orient
        assert d["orientation"]["applied"] is (orient != 1)
        out = Image.open(io.BytesIO(_decode_copy(d)))
        assert out.size == ref.size == (d["copy"]["width"], d["copy"]["height"])
        assert np.array_equal(np.asarray(out), np.asarray(ref))
        assert d["recheck"]["passed"] is True


def test_jpeg_metadata_stripped_and_recheck_passes():
    client, _ = _fresh_client()
    jpg = _jpeg_with_app1(_full_exif())
    # 与只读检查结果对照
    chk = client.post(
        "/api/privacy-check",
        files={"image": ("p.jpg", jpg, "image/jpeg")}).json()
    d = client.post(
        "/api/privacy-export",
        files={"image": ("p.jpg", jpg, "image/jpeg")}).json()
    assert d["original"]["date_time"] == chk["date_time"] == "2026:09:23 12:34:56"
    assert d["original"]["date_time_iso"] == "2026-09-23T12:34:56"
    assert d["original"]["camera_make"] == "AcmeCam"
    assert d["original"]["camera_model"] == "Model X-100"
    assert d["original"]["gps"]["latitude"] == 31.2625
    assert d["original"]["gps"]["longitude"] == 121.5

    raw = _decode_copy(d)
    assert raw.startswith(b"\x89PNG\r\n\x1a\n")
    chunks = _png_chunks(raw)
    assert chunks == ["IHDR", "IDAT", "IEND"]
    rc = d["recheck"]
    assert rc["passed"] is True
    assert all(rc["checks"].values())
    assert rc["exif_present"] is False and rc["gps_present"] is False
    assert rc["icc_present"] is False and rc["xmp_present"] is False
    assert rc["text_metadata_present"] is False
    assert rc["chunks"] == ["IHDR", "IDAT", "IEND"]
    rep = rc["reparsed_privacy"]
    assert rep["exif_present"] is False and rep["gps"] is None
    assert rep["date_time"] is None and rep["camera_make"] is None

    # 字节长度 / sha256 与返回一致
    assert len(raw) == d["copy"]["bytes"]
    import hashlib
    assert hashlib.sha256(raw).hexdigest() == d["copy"]["sha256"]
    assert d["stored"] is False and "job_id" not in d


def test_png_rgba_transparency_preserved_and_meta_stripped():
    client, _ = _fresh_client()
    png, rgba = _rgba_png(orientation=3)
    r = client.post(
        "/api/privacy-export",
        files={"image": ("p.png", png, "image/png")})
    assert r.status_code == 200, r.text
    d = r.json()
    # 180° 摆正
    assert d["orientation"]["value"] == 3
    raw = _decode_copy(d)
    out = Image.open(io.BytesIO(raw))
    out.load()
    assert out.mode == "RGBA"
    assert out.has_transparency_data is True
    assert d["copy"]["mode"] == "RGBA"
    assert d["copy"]["has_alpha"] is True
    expected = np.asarray(Image.fromarray(rgba).rotate(180))
    assert np.array_equal(np.asarray(out), expected)  # 像素逐值无损
    assert _png_chunks(raw) == ["IHDR", "IDAT", "IEND"]
    assert d["recheck"]["passed"] is True


def test_la_and_palette_transparency_modes():
    client, _ = _fresh_client()
    # LA
    la = np.dstack([np.full((24, 32), 130, np.uint8),
                    np.tile(np.arange(32, dtype=np.uint8), (24, 1))])
    buf = io.BytesIO()
    Image.fromarray(la, "LA").save(buf, format="PNG")
    d = client.post("/api/privacy-export",
                    files={"image": ("la.png", buf.getvalue(), "image/png")}).json()
    assert d["copy"]["mode"] == "LA" and d["copy"]["has_alpha"] is True
    out = Image.open(io.BytesIO(_decode_copy(d)))
    assert out.mode == "LA" and np.array_equal(np.asarray(out), la)

    # P 模式带透明索引 -> 必须转 RGBA 而不是丢掉透明
    rgba = np.zeros((16, 16, 4), np.uint8)
    rgba[..., 1] = 200
    rgba[..., 3] = np.where(np.indices((16, 16))[0] < 8, 0, 255).astype("uint8")
    p = Image.fromarray(rgba, "RGBA").convert("P", palette=Image.ADAPTIVE)
    buf2 = io.BytesIO()
    p.save(buf2, format="PNG", transparency=0)
    d2 = client.post("/api/privacy-export",
                     files={"image": ("p.png", buf2.getvalue(), "image/png")}).json()
    assert d2["copy"]["mode"] == "RGBA" and d2["copy"]["has_alpha"] is True
    assert d2["recheck"]["passed"] is True

    # L 带 tRNS 键值 -> LA
    l = Image.new("L", (16, 16), 128)
    buf3 = io.BytesIO()
    l.save(buf3, format="PNG", transparency=0)
    d3 = client.post("/api/privacy-export",
                     files={"image": ("l.png", buf3.getvalue(), "image/png")}).json()
    assert d3["copy"]["mode"] == "LA" and d3["copy"]["has_alpha"] is True
    assert d3["recheck"]["passed"] is True


def test_rgb_no_alpha_stays_rgb_and_png_input():
    client, _ = _fresh_client()
    png = _png_with_exif(_full_exif())
    d = client.post("/api/privacy-export",
                    files={"image": ("p.png", png, "image/png")}).json()
    assert d["source_format"] == "PNG"
    assert d["copy"]["format"] == "PNG"
    assert d["copy"]["mode"] == "RGB"
    assert d["copy"]["has_alpha"] is False
    assert d["recheck"]["passed"] is True


def test_no_orientation_means_passthrough():
    client, _ = _fresh_client()
    d = client.post(
        "/api/privacy-export",
        files={"image": ("plain.jpg", _plain_jpeg(), "image/jpeg")}).json()
    assert d["orientation"]["value"] is None
    assert d["orientation"]["applied"] is False
    assert "无 EXIF Orientation" in d["orientation"]["label"]
    assert d["original"]["date_time"] is None and d["original"]["gps"] is None
    assert d["recheck"]["passed"] is True


def test_corrupt_exif_does_not_block_export():
    client, _ = _fresh_client()
    payload = b"Exif\x00\x00" + b"\xff" * 100
    seg = b"\xff\xe1" + struct.pack(">H", len(payload) + 2) + payload
    jpg = _plain_jpeg()[:2] + seg + _plain_jpeg()[2:]
    r = client.post("/api/privacy-export",
                    files={"image": ("broken.jpg", jpg, "image/jpeg")})
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["orientation"]["value"] is None
    assert any("损坏" in w or "无法" in w for w in d["original"]["warnings"])
    assert d["recheck"]["passed"] is True


def test_recheck_failure_returns_500_and_no_download(monkeypatch=None):
    """编码器若意外泄漏文本元数据：复查失败 -> 500，响应无 download。"""
    client, _ = _fresh_client()
    from app import privacy_export as pe

    def leaky(im):
        info = PngImagePlugin.PngInfo()
        info.add_text("Comment", "LEAK")
        buf = io.BytesIO()
        Image.fromarray(np.asarray(im), im.mode).save(
            buf, format="PNG", pnginfo=info)
        return buf.getvalue()

    orig = pe.encode_clean_png
    pe.encode_clean_png = leaky
    try:
        buf = io.BytesIO()
        Image.fromarray(np.full((32, 32, 3), 90, np.uint8)).save(buf, format="PNG")
        r = client.post("/api/privacy-export",
                        files={"image": ("p.png", buf.getvalue(), "image/png")})
    finally:
        pe.encode_clean_png = orig
    assert r.status_code == 500
    d = r.json()
    assert "download" not in d
    assert d["recheck"]["passed"] is False
    assert any("tEXt" in o or "文本" in o for o in d["offenders"])
    assert d["recheck"]["checks"]["no_text_chunks"] is False
    # 失败响应仍回显原图识别与方向信息，便于页面展示；但明确无下载
    assert d["stored"] is False
    assert "original" in d and "orientation" in d
    assert d["source_format"] == "PNG"


def test_unsupported_bad_empty_oversize():
    client, _ = _fresh_client()
    buf = io.BytesIO()
    Image.new("RGB", (16, 16)).save(buf, format="WEBP")
    r = client.post("/api/privacy-export",
                    files={"image": ("a.webp", buf.getvalue(), "image/webp")})
    assert r.status_code == 400 and "PNG / JPEG" in r.json()["detail"]

    r = client.post("/api/privacy-export",
                    files={"image": ("x.jpg", b"not-an-image" * 8, "image/jpeg")})
    assert r.status_code == 400

    r = client.post("/api/privacy-export",
                    files={"image": ("x.png", b"", "image/png")})
    assert r.status_code == 400

    # 摆正后超过 4096px：构造长边 4097 的 JPEG（JPEG 上限 65535，可直接保存）
    from app import privacy_export as pe
    assert pe.MAX_EXPORT_DIM == 4096
    big = np.full((8, 4097, 3), 100, np.uint8)
    b2 = io.BytesIO()
    Image.fromarray(big).save(b2, format="JPEG")
    r = client.post("/api/privacy-export",
                    files={"image": ("big.jpg", b2.getvalue(), "image/jpeg")})
    assert r.status_code == 400 and "4096" in r.json()["detail"]

    # 超过 16MB 上传限制 -> 413
    huge = io.BytesIO()
    # 用随机噪声生成超大 PNG
    arr = np.random.RandomState(1).randint(0, 255, (4000, 4000, 3), np.uint8)
    Image.fromarray(arr).save(huge, format="PNG", compress_level=1)
    if len(huge.getvalue()) > 16 * 1024 * 1024:
        r = client.post("/api/privacy-export",
                        files={"image": ("huge.png", huge.getvalue(), "image/png")})
        assert r.status_code == 413


def test_no_persistence_jobs_or_archive():
    client, data_dir = _fresh_client()
    jpg = _jpeg_with_app1(_full_exif())
    png, _ = _rgba_png()
    for payload, name, ctype in ((jpg, "a.jpg", "image/jpeg"),
                                 (png, "b.png", "image/png")):
        for _ in range(2):
            r = client.post("/api/privacy-export",
                            files={"image": (name, payload, ctype)})
            assert r.status_code == 200
    assert not list(data_dir.glob("*/meta.json"))
    assert not (data_dir / "archive.log.jsonl").exists()
    # 水印任务文件也不受影响：目录应为空
    assert [p.name for p in data_dir.iterdir()] == []


def test_existing_readonly_privacy_check_unchanged():
    client, _ = _fresh_client()
    d = client.post(
        "/api/privacy-check",
        files={"image": ("p.jpg", _jpeg_with_app1(_full_exif()),
                         "image/jpeg")}).json()
    assert d["date_time"] == "2026:09:23 12:34:56"
    assert d["camera_make"] == "AcmeCam"
    assert d["gps"]["latitude"] == 31.2625
    assert d["stored"] is False


def test_orientation_geometry_swaps_dimensions():
    """Orientation=6/8 摆正后宽高互换。"""
    client, _ = _fresh_client()
    d = client.post(
        "/api/privacy-export",
        files={"image": ("o6.jpg", _oriented_jpeg(6, (100, 200)),
                         "image/jpeg")}).json()
    assert d["source_size"] == {"width": 200, "height": 100}
    assert (d["copy"]["width"], d["copy"]["height"]) == (100, 200)
    assert d["recheck"]["checks"]["dimensions_match"] is True


if __name__ == "__main__":
    import inspect
    fns = [v for k, v in sorted(globals().items())
           if k.startswith("test_") and inspect.isfunction(v)]
    for fn in fns:
        fn()
        print(f"[PASS] {fn.__name__}")
    print(f"\n{len(fns)} 个隐私副本导出测试全部通过 ✔")
