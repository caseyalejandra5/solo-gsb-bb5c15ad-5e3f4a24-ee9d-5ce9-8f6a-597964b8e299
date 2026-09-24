"""传播副本来源定位：SIFT + RANSAC 几何校验纯逻辑 + 端到端 API（只读不落盘）。

运行（需项目依赖）：
    WM_DATA_DIR 由本测试自动指向临时目录；
    python -m pytest tests/test_provenance.py -q
或直接：
    python tests/test_provenance.py
"""
import io
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ---------------------------------------------------------------- 测试图

def _photo_rgb(size=512, seed=3):
    """带多频纹理与类物体圆斑的自然照片风格合成图（SIFT 特征丰富）。"""
    rng = np.random.RandomState(seed)
    y, x = np.mgrid[0:size, 0:size].astype(np.float32)
    img = np.zeros((size, size, 3), np.float32)
    for ch in range(3):
        t = np.zeros_like(x) + 90 + 30 * ch
        for fx, fy, amp, ph in [(1.7, 2.3, 40, 0), (3.1, 1.1, 22, 1.5),
                                (5.3, 4.7, 12, 3.1), (9.2, 7.7, 7, 0.7)]:
            t += amp * np.sin(2 * np.pi * (fx * x / size + fy * y / size) + ph)
        t += 25 * ((x + y) / size)
        img[..., ch] = t
    im = Image.fromarray(img.clip(0, 255).astype(np.uint8))
    d = ImageDraw.Draw(im)
    rr = np.random.RandomState(seed + 7)
    for _ in range(26):
        cx, cy = rr.randint(30, size - 30, 2)
        r = rr.randint(12, 60)
        d.ellipse([cx - r, cy - r, cx + r, cy + r],
                  fill=tuple(int(v) for v in rr.randint(40, 230, 3)))
    im = im.filter(ImageFilter.GaussianBlur(1.2))
    arr = np.asarray(im).astype(np.float32) + rng.normal(0, 6, (size, size, 3))
    return arr.clip(0, 255).astype(np.uint8)


def _png_bytes(rgb):
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="PNG")
    return buf.getvalue()


def _jpeg_bytes(rgb, quality):
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def _scaled(rgb, factor, resample=Image.BILINEAR):
    h, w = rgb.shape[:2]
    return np.array(Image.fromarray(rgb).resize(
        (max(2, int(round(w * factor))), max(2, int(round(h * factor)))),
        resample))


# ---------------------------------------------------------------- 几何纯逻辑

def test_polygon_area_and_quad_helpers():
    from app import provenance as pv

    sq = np.array([[0.0, 0], [10, 0], [10, 10], [0, 10]])
    assert abs(pv._polygon_area(sq) - 100.0) < 1e-9
    assert pv._is_convex(sq)
    ang = pv._quad_angles_deg(sq)
    assert np.allclose(ang, 90.0)

    # 凹四边形（镖形）必须被判为非凸
    dart = np.array([[0.0, 0], [10, 0], [2, 2], [0, 10]])
    assert not pv._is_convex(dart)

    h, v = pv._axis_lengths(sq)
    assert h == 10.0 and v == 10.0


def test_project_points_homography_identity_and_shift():
    from app import provenance as pv

    pts = np.array([[1.0, 2], [3, 4], [5, 6]])
    out = pv._project_points(pts, np.eye(3))
    assert np.allclose(out, pts)

    H = np.array([[1, 0, 7.5], [0, 1, -3.0], [0, 0, 1]], dtype=np.float64)
    out = pv._project_points(pts, H)
    assert np.allclose(out, pts + np.array([7.5, -3.0]))


def test_thresholds_present():
    from app import provenance as pv

    t = pv.thresholds()
    for key in ("min_good_matches", "min_inliers", "min_inlier_ratio",
                "max_rms_px", "min_coverage", "max_coverage"):
        assert key in t
    assert 0 < t["min_inlier_ratio"] <= 1
    assert t["min_coverage"] < t["max_coverage"]


# ---------------------------------------------------------------- 定位算法

def _watermarked(rgb, text="来源定位-测试"):
    from app import dctwm as wm
    return wm.embed_watermark(rgb, text, strength="medium").image


def test_locate_full_jpeg_transcode():
    from app import provenance as pv

    ref = _watermarked(_photo_rgb(seed=3))
    query = np.array(Image.open(io.BytesIO(_jpeg_bytes(ref, 40))).convert("RGB"))
    r = pv.locate_propagation_copy(ref, query)
    assert r["located"] is True, r["reason"]
    pts = np.array([(c["x"], c["y"]) for c in r["quad"]])
    truth = np.array([[0, 0], [512, 0], [512, 512], [0, 512]], float)
    assert np.abs(pts - truth).max() < 3.0
    assert 0.98 <= r["coverage"] <= 1.02
    assert r["inlier_count"] >= 10
    assert r["inlier_ratio"] >= 0.35
    assert r["preview_data_url"].startswith("data:image/jpeg;base64,")
    assert all(r["checks"].values())


def test_locate_crop_scale_jpeg_corners():
    from app import provenance as pv

    ref = _watermarked(_photo_rgb(seed=3))
    x0, y0, x1, y1 = 100, 120, 410, 430
    crop = ref[y0:y1, x0:x1]
    query = _scaled(crop, 0.7)
    query = np.array(Image.open(io.BytesIO(_jpeg_bytes(query, 50))).convert("RGB"))
    r = pv.locate_propagation_copy(ref, query)
    assert r["located"] is True, r["reason"]
    pts = np.array([(c["x"], c["y"]) for c in r["quad"]])
    truth = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], float)
    assert np.abs(pts - truth).max() < 5.0, np.abs(pts - truth).max()
    # 覆盖率 ≈ 裁切面积 / 参照面积
    expect_cov = (x1 - x0) * (y1 - y0) / (512 * 512)
    assert abs(r["coverage"] - expect_cov) < 0.02
    assert [c["name"] for c in r["quad"]] == [
        "top_left", "top_right", "bottom_right", "bottom_left"]


def test_locate_upscale_copy():
    from app import provenance as pv

    ref = _watermarked(_photo_rgb(seed=5))
    crop = ref[80:430, 60:410]
    query = _scaled(crop, 1.8, resample=Image.BICUBIC)
    query = np.array(Image.open(io.BytesIO(_jpeg_bytes(query, 70))).convert("RGB"))
    r = pv.locate_propagation_copy(ref, query)
    assert r["located"] is True, r["reason"]
    pts = np.array([(c["x"], c["y"]) for c in r["quad"]])
    truth = np.array([[60, 80], [410, 80], [410, 430], [60, 430]], float)
    assert np.abs(pts - truth).max() < 5.0


def test_locate_rejects_unrelated_image():
    from app import provenance as pv

    ref = _watermarked(_photo_rgb(seed=3))
    other = _photo_rgb(seed=99)
    other = np.array(Image.open(io.BytesIO(_jpeg_bytes(other, 80))).convert("RGB"))
    r = pv.locate_propagation_copy(ref, other)
    assert r["located"] is False
    assert r["reason"]
    assert r["failed_check"] in set(r["checks"])
    assert not all(r["checks"].values())
    assert r["quad"] is None and r["coverage"] is None
    assert r["preview_data_url"] is None


def test_locate_rejects_flat_image():
    from app import provenance as pv

    ref = _watermarked(_photo_rgb(seed=3))
    flat = np.full((300, 300, 3), 200, np.uint8)
    r = pv.locate_propagation_copy(ref, flat)
    assert r["located"] is False
    assert r["failed_check"] == "good_matches"
    assert r["good_matches"] < pv.MIN_GOOD_MATCHES
    assert r["quad"] is None


def test_locate_rejects_degenerate_subimage_without_fake_position():
    """几何退化 / 匹配不足时任何情况下都不得返回四角坐标。"""
    from app import provenance as pv

    ref = _watermarked(_photo_rgb(seed=3))

    # 极小、重压的副本：无论卡在哪一道门限，都不能给出伪位置
    tiny = ref[200:260, 200:260]
    tiny = _scaled(tiny, 0.5)
    tiny = np.array(Image.open(io.BytesIO(_jpeg_bytes(tiny, 20))).convert("RGB"))
    r = pv.locate_propagation_copy(ref, tiny)
    assert r["located"] is False
    assert r["quad"] is None and r["coverage"] is None
    assert r["preview_data_url"] is None
    assert r["reason"]


# ---------------------------------------------------------------- API 端到端

def _fresh_client():
    tmp = tempfile.mkdtemp(prefix="wm-loc-")
    os.environ["WM_DATA_DIR"] = tmp
    os.environ["WM_MAX_JOBS"] = "200"
    for mod in ("app.main", "app.archive"):
        sys.modules.pop(mod, None)
    from fastapi.testclient import TestClient
    from app.main import app, DATA_DIR  # noqa: E402

    return TestClient(app), Path(DATA_DIR)


def _embed_job(client, text="来源定位-2026", seed=3):
    r = client.post(
        "/api/embed",
        files={"image": ("img.png", _png_bytes(_photo_rgb(seed=seed)), "image/png")},
        data={"text": text, "strength": "medium"},
    )
    assert r.status_code == 200, r.text
    return r.json()["job_id"]


def test_locate_endpoint_success_shape():
    client, _ = _fresh_client()
    job = _embed_job(client)
    wm_bytes = client.get(f"/api/jobs/{job}/watermarked").content
    ref = np.array(Image.open(io.BytesIO(wm_bytes)).convert("RGB"))
    copy = _scaled(ref[120:430, 100:410], 0.7)
    r = client.post(
        f"/api/jobs/{job}/locate",
        files={"image": ("forward.jpg", _jpeg_bytes(copy, 45), "image/jpeg")})
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["located"] is True
    assert data["stored"] is False
    assert data["job_id"] == job
    assert len(data["quad"]) == 4
    assert data["coverage"] is not None and 0 < data["coverage"] <= 1.02
    assert isinstance(data["inlier_count"], int)
    assert isinstance(data["inlier_ratio"], float)
    assert data["rms_px"] >= 0
    assert data["preview_data_url"].startswith("data:image/jpeg;base64,")
    assert {"good_matches", "inlier_count", "inlier_ratio", "quad_shape",
            "quad_bounds", "coverage", "reprojection_rms"} <= set(data["checks"])

    pts = np.array([(c["x"], c["y"]) for c in data["quad"]])
    truth = np.array([[100, 120], [410, 120], [410, 430], [100, 430]], float)
    assert np.abs(pts - truth).max() < 6.0


def test_locate_endpoint_unrelated_is_200_failure():
    client, _ = _fresh_client()
    job = _embed_job(client)
    r = client.post(
        f"/api/jobs/{job}/locate",
        files={"image": ("x.jpg", _jpeg_bytes(_photo_rgb(seed=123), 80),
                         "image/jpeg")})
    assert r.status_code == 200
    data = r.json()
    assert data["located"] is False
    assert data["reason"] and data["failed_check"]
    assert data["quad"] is None and data["coverage"] is None
    assert data["preview_data_url"] is None


def test_locate_does_not_modify_job_or_archive():
    client, data_dir = _fresh_client()
    job = _embed_job(client)

    # 先制造 target.png / 攻击历史 / 一次检测归档
    r = client.post("/api/attack",
                    json={"job_id": job, "jpeg_quality": 70,
                          "crop_fraction": 0.75})
    assert r.status_code == 200
    det = client.post("/api/detect", data={"job_id": job})
    assert det.status_code == 200
    det_id = det.json()["detection_id"]

    job_dir = data_dir / job
    before = {p.relative_to(job_dir).as_posix(): p.stat().st_mtime_ns
              for p in job_dir.rglob("*") if p.is_file()}
    meta_before = (job_dir / "meta.json").read_text("utf-8")
    wm_before = (job_dir / "watermarked.png").read_bytes()

    # 两次定位：一次成功、一次无关图失败
    wm_png = (job_dir / "watermarked").read_bytes() if False else wm_before
    ref = np.array(Image.open(io.BytesIO(wm_before)).convert("RGB"))
    copy = _scaled(ref[90:420, 80:400], 0.8)
    ok = client.post(
        f"/api/jobs/{job}/locate",
        files={"image": ("c.jpg", _jpeg_bytes(copy, 55), "image/jpeg")})
    assert ok.status_code == 200 and ok.json()["located"] is True
    bad = client.post(
        f"/api/jobs/{job}/locate",
        files={"image": ("x.jpg", _jpeg_bytes(_photo_rgb(seed=55), 80),
                         "image/jpeg")})
    assert bad.status_code == 200 and bad.json()["located"] is False

    after = {p.relative_to(job_dir).as_posix(): p.stat().st_mtime_ns
             for p in job_dir.rglob("*") if p.is_file()}
    assert after == before
    assert (job_dir / "meta.json").read_text("utf-8") == meta_before
    assert (job_dir / "watermarked.png").read_bytes() == wm_before

    # 没有新增检测记录 / 标注图 / 归档 / 报告
    det_files = sorted(p.name for p in (job_dir / "detections").iterdir())
    assert det_files == [f"{det_id}.json", f"{det_id}.png",
                         f"{det_id}.source.png"]
    assert not (data_dir / "archive.log.jsonl").exists()
    assert not (data_dir / "reports").exists()


def test_locate_endpoint_errors():
    client, data_dir = _fresh_client()
    job = _embed_job(client)
    copy = _jpeg_bytes(_photo_rgb(), 80)

    # 任务不存在 -> 404
    assert client.post("/api/jobs/000000000000/locate",
                       files={"image": ("x.jpg", copy, "image/jpeg")}
                       ).status_code == 404

    # upload_only 轻量任务（独立检测产生）-> 404
    det = client.post(
        "/api/detect",
        files={"image": ("u.png", _png_bytes(_photo_rgb(seed=77)), "image/png")})
    up_job = det.json()["job_id"]
    assert det.json()["upload_only"] is True
    assert client.post(f"/api/jobs/{up_job}/locate",
                       files={"image": ("x.jpg", copy, "image/jpeg")}
                       ).status_code == 404

    # 空上传 -> 400/422；损坏内容 -> 400；过短边 -> 400
    assert client.post(f"/api/jobs/{job}/locate",
                       files={"image": ("", b"", "application/octet-stream")}
                       ).status_code in (400, 422)
    assert client.post(f"/api/jobs/{job}/locate",
                       files={"image": ("x.png", b"not an image",
                                        "image/png")}).status_code == 400
    tiny = _png_bytes(np.zeros((32, 32, 3), np.uint8))
    assert client.post(f"/api/jobs/{job}/locate",
                       files={"image": ("t.png", tiny, "image/png")}
                       ).status_code == 400


if __name__ == "__main__":
    import inspect
    sys.modules.pop("app.main", None)
    fns = [v for k, v in sorted(globals().items())
           if k.startswith("test_") and inspect.isfunction(v)]
    for fn in fns:
        fn()
        print(f"[PASS] {fn.__name__}")
    print(f"\n{len(fns)} 个传播副本来源定位测试全部通过 ✔")
