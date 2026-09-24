"""自动方向检索测试。

覆盖：
  - 8 个方向的正向变换 / 逆变换回环（含非方形图的尺寸互换）
  - 最佳方向选择：置信度优先、同分固定顺序取舍
  - 冲突判定：有效候选恢复出不同文本
  - 端到端：旋转 / 镜像后的上传图仅在 auto_orient=true 时恢复水印
  - 入选方向标注图逆变换回上传图坐标（与上传图同尺寸，且原始快照不变）
  - 普通检测流程完全不变（不产生 orientation_search 字段）
  - 方向与候选摘要写入检测记录并随报告归档

运行：
    python -m pytest tests/test_orientation.py -q
或：
    python tests/test_orientation.py
"""
import io
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import orientation as ori  # noqa: E402


# ------------------------------------------------------------ 纯函数单测

def test_apply_invert_roundtrip_rect():
    rng = np.random.RandomState(11)
    rgb = rng.randint(0, 256, (200, 300, 3), dtype=np.uint8)
    for spec in ori.ORIENTATIONS:
        view = ori.apply_orientation(rgb, spec)
        back = ori.invert_to_original(view, spec)
        assert back.shape == rgb.shape, (spec["id"], back.shape, rgb.shape)
        assert np.array_equal(back, rgb), spec["id"]


def test_orientation_fixed_order():
    ids = [s["id"] for s in ori.ORIENTATIONS]
    assert ids == [
        "original", "rot90cw", "rot180", "rot270cw",
        "mirror", "mirror_rot90cw", "mirror_rot180", "mirror_rot270cw",
    ]


def test_pick_best_by_confidence_then_order():
    cands = [
        {"index": 0, "detected": True, "confidence": 80.0},
        {"index": 1, "detected": True, "confidence": 90.0},
        {"index": 2, "detected": True, "confidence": 90.0},  # 同分靠后
        {"index": 3, "detected": False, "confidence": 0.0},
    ]
    assert ori.pick_best_index(cands) == 1
    # 全部无效 -> None
    assert ori.pick_best_index([cands[3]]) is None
    # 同分且最高出现在 0/1 -> 取固定顺序 0
    tie = [
        {"index": 0, "detected": True, "confidence": 99.0},
        {"index": 1, "detected": True, "confidence": 99.0},
    ]
    assert ori.pick_best_index(tie) == 0


def test_distinct_texts_and_conflict():
    summaries = [
        {"detected": True, "text": "A"},
        {"detected": True, "text": "B"},
        {"detected": False, "text": ""},
        {"detected": True, "text": "A"},   # 重复不重复计数
    ]
    assert ori.distinct_texts(summaries) == ["A", "B"]
    assert ori.distinct_texts([summaries[0], summaries[3]]) == ["A"]


# ------------------------------------------------------------ 端到端 API

def _fresh_client():
    tmp = tempfile.mkdtemp(prefix="wm-orient-")
    os.environ["WM_DATA_DIR"] = tmp
    os.environ["WM_MAX_JOBS"] = "200"
    for mod in ("app.main", "app.archive"):
        sys.modules.pop(mod, None)
    from fastapi.testclient import TestClient
    from app.main import app, DATA_DIR  # noqa: E402
    return TestClient(app), Path(DATA_DIR)


def _make_textured_png(size=384, seed=7) -> bytes:
    rng = np.random.RandomState(seed)
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


def _png_bytes(arr: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return buf.getvalue()


def _watermarked_png(client, text="方向-2026") -> np.ndarray:
    r = client.post(
        "/api/embed",
        files={"image": ("img.png", _make_textured_png(), "image/png")},
        data={"text": text, "strength": "medium"},
    )
    assert r.status_code == 200, r.text
    wm_path = next(Path(os.environ["WM_DATA_DIR"]).glob("*/watermarked.png"))
    return np.array(Image.open(wm_path).convert("RGB"))


def _rotated_upload(arr: np.ndarray, name: str) -> bytes:
    """模拟传播中的旋转 / 镜像（PIL transpose，正向失真）。"""
    ops = {
        "cw90": Image.Transpose.ROTATE_270,         # PIL 270° = 顺时针 90°
        "r180": Image.Transpose.ROTATE_180,
        "ccw90": Image.Transpose.ROTATE_90,
        "mirror": Image.Transpose.FLIP_LEFT_RIGHT,
        "mirror_cw90": Image.Transpose.TRANSVERSE,  # 镜像后顺时针 90°
        "mirror_ccw90": Image.Transpose.TRANSPOSE,  # 镜像后逆时针 90°
    }
    im = Image.fromarray(arr).transpose(ops[name])
    return _png_bytes(np.array(im))


def test_rotated_upload_normal_mode_fails():
    client, _ = _fresh_client()
    wm_img = _watermarked_png(client)
    r = client.post(
        "/api/detect",
        files={"image": ("up.png", _rotated_upload(wm_img, "cw90"), "image/png")},
        data={"auto_orient": "false"},
    )
    det = r.json()
    assert det["detected"] is False
    assert det["orientation_search"] is None


def test_auto_orient_recovers_all_directions():
    client, data_dir = _fresh_client()
    wm_img = _watermarked_png(client, text="方向-2026")
    expected = {
        "cw90": "rot270cw",
        "r180": "rot180",
        "ccw90": "rot90cw",
        "mirror": "mirror",
        "mirror_cw90": "mirror_rot90cw",
        "mirror_ccw90": "mirror_rot270cw",
    }
    for upload_name, chosen_id in expected.items():
        r = client.post(
            "/api/detect",
            files={"image": ("up.png", _rotated_upload(wm_img, upload_name),
                             "image/png")},
            data={"auto_orient": "true"},
        )
        assert r.status_code == 200, r.text
        det = r.json()
        assert det["detected"] is True, det.get("reason")
        assert det["text"] == "方向-2026", (upload_name, det["text"])
        os_ = det["orientation_search"]
        assert os_["orientations_tried"] == 8
        assert os_["chosen_orientation"] == chosen_id
        assert os_["valid_candidate_count"] >= 1
        assert os_["conflict"] is False
        # 有效候选中只有一个能恢复出文本
        valid = [c for c in os_["candidates"] if c["detected"]]
        assert all(c["text"] == "方向-2026" for c in valid)


def test_upright_upload_single_valid_candidate():
    client, _ = _fresh_client()
    wm_img = _watermarked_png(client)
    r = client.post(
        "/api/detect",
        files={"image": ("up.png", _png_bytes(wm_img), "image/png")},
        data={"auto_orient": "true"},
    )
    os_ = r.json()["orientation_search"]
    assert os_["chosen_orientation"] == "original"
    valid = [c for c in os_["candidates"] if c["detected"]]
    assert len(valid) == 1 and valid[0]["orientation"] == "original"


def test_non_watermarked_upload_no_valid_candidate():
    client, _ = _fresh_client()
    _watermarked_png(client)  # 仅确保有任务目录逻辑，上传的是无水印随机图
    r = client.post(
        "/api/detect",
        files={"image": ("plain.png", _make_textured_png(seed=99), "image/png")},
        data={"auto_orient": "true"},
    )
    det = r.json()
    assert det["detected"] is False
    os_ = det["orientation_search"]
    assert os_["chosen_index"] is None
    assert os_["valid_candidate_count"] == 0
    assert det["marked_det_url"] is None


def test_overlay_back_transformed_and_snapshot_unchanged():
    client, data_dir = _fresh_client()
    wm_img = _watermarked_png(client)
    upload = _rotated_upload(wm_img, "cw90")
    up_arr = np.array(Image.open(io.BytesIO(upload)).convert("RGB"))

    r = client.post(
        "/api/detect",
        files={"image": ("up.png", upload, "image/png")},
        data={"auto_orient": "true"},
    )
    det = r.json()
    assert det["detected"] and det["marked_det_url"]

    job = data_dir / det["job_id"]
    snap = np.array(Image.open(
        job / "detections" / f"{det['detection_id']}.source.png").convert("RGB"))
    marked = np.array(Image.open(
        job / "detections" / f"{det['detection_id']}.png").convert("RGB"))

    # 检测输入快照 = 未经几何变换的上传图
    assert snap.shape == up_arr.shape
    assert np.array_equal(snap, up_arr)
    # 入选方向标注图已逆变换回上传图坐标（与上传图同尺寸）
    assert marked.shape == up_arr.shape


def test_orientation_summary_persisted_and_archived():
    client, data_dir = _fresh_client()
    wm_img = _watermarked_png(client)
    det = client.post(
        "/api/detect",
        files={"image": ("up.png", _rotated_upload(wm_img, "r180"), "image/png")},
        data={"auto_orient": "true"},
    ).json()

    # 检测记录 JSON 中含方向摘要
    rec = json.loads((data_dir / det["job_id"] / "detections"
                      / f"{det['detection_id']}.json").read_text("utf-8"))
    os_ = rec["orientation_search"]
    assert os_["chosen_orientation"] == "rot180"
    assert len(os_["candidates"]) == 8

    # 随报告归档
    rid = client.post(
        f"/api/jobs/{det['job_id']}/reports",
        json={"detection_ids": [det["detection_id"]]}).json()["report_id"]
    report = client.get(f"/api/reports/{rid}").json()
    d0 = report["detections"][0]
    assert d0["orientation_search"]["chosen_orientation"] == "rot180"
    assert client.post(f"/api/reports/{rid}/verify").json()["valid"] is True


def test_job_detection_ignores_auto_orient_flag():
    client, _ = _fresh_client()
    # 标准嵌入 -> 攻击 -> 用 job_id 检测，即便带 auto_orient 也走原流程
    emb = client.post(
        "/api/embed",
        files={"image": ("img.png", _make_textured_png(), "image/png")},
        data={"text": "普通流程", "strength": "medium"},
    ).json()
    client.post("/api/attack",
                json={"job_id": emb["job_id"], "jpeg_quality": 70,
                      "crop_fraction": 0.75})
    det = client.post(
        "/api/detect",
        data={"job_id": emb["job_id"], "auto_orient": "true"}).json()
    assert det["detected"] is True
    assert det["text"] == "普通流程"
    assert det["orientation_search"] is None


if __name__ == "__main__":
    import inspect
    fns = [v for k, v in sorted(globals().items())
           if k.startswith("test_") and inspect.isfunction(v)]
    for fn in fns:
        fn()
        print(f"[PASS] {fn.__name__}")
    print(f"\n{len(fns)} 个方向检索测试全部通过 ✔")
