"""鲁棒性矩阵评测：边界偏序纯逻辑 + 端到端 API（只读不改写）。

运行（需项目依赖）：
    WM_DATA_DIR 由本测试自动指向临时目录；
    python -m pytest tests/test_matrix.py -q
或直接：
    python tests/test_matrix.py
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


def _make_png(size=384, seed=7) -> bytes:
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


# ---------------------------------------------------------------- 边界纯逻辑

def _mk(q, f, ok=True):
    return {"jpeg_quality": q, "crop_fraction": f, "text_match": ok}


def test_boundary_strict_domination():
    from app.matrix import success_boundary

    # 满笛卡尔积且最狠角 (30,0.5) 成功时，它沿两个方向支配其余所有格，
    # 边界只剩最狠角自己。
    cells = [
        _mk(90, 1.0), _mk(50, 0.75), _mk(50, 0.5),
        _mk(30, 0.75), _mk(30, 0.5),
    ]
    assert success_boundary(cells) == {(30, 0.5)}

    # 最狠角失败时，前沿由两个互不可比较的成功格构成：
    # (30,0.75) 质量更严但裁切更轻；(50,0.5) 反之，谁也不支配谁。
    cells = [
        _mk(90, 1.0), _mk(50, 0.75), _mk(50, 0.5),
        _mk(30, 0.75), _mk(30, 0.5, ok=False),
    ]
    b = success_boundary(cells)
    assert b == {(30, 0.75), (50, 0.5)}, b
    assert (90, 1.0) not in b and (50, 0.75) not in b

    # 「至少一项严格更严」才算支配：全等参数格互不支配（实际参数已去重）
    dup = [_mk(50, 0.75), _mk(50, 0.75)]
    assert success_boundary(dup) == {(50, 0.75)}

    # 失败格 / 文本不一致格不参与边界
    mix = [_mk(50, 0.75, ok=True), _mk(30, 0.5, ok=False)]
    assert success_boundary(mix) == {(50, 0.75)}
    assert success_boundary([]) == set()


def test_boundary_does_not_infer_gaps():
    """只看实测成功格：未测的中间组合不会被补出来参与支配。"""
    from app.matrix import success_boundary

    # 只实测了 (50,1.0) 与 (90,0.5) 两个成功格（及其更轻格）。中间缺口
    # (50,0.5) 未测：若“按相邻推断”为成功，它会同时支配这两格、使边界坍缩
    # 为 1 格；本函数不推断，故边界保留两个不可比较的实测格。
    cells = [_mk(90, 1.0), _mk(50, 1.0), _mk(90, 0.5)]
    b = success_boundary(cells)
    assert b == {(50, 1.0), (90, 0.5)}, b

    # 加测 (50,0.5) 成功后，边界按实测结果才坍缩为该格
    assert success_boundary(cells + [_mk(50, 0.5)]) == {(50, 0.5)}


def test_validate_axes():
    from app.matrix import MatrixError, validate_axes

    assert validate_axes([50, 90, 30], [1.0, 0.5]) == ([30, 50, 90], [0.5, 1.0])
    assert validate_axes([100], [1.0]) == ([100], [1.0])

    for bad in ([], [19], [101], [50, 50], [1, 2, 3, 4]):
        try:
            validate_axes(bad, [0.5])
        except MatrixError:
            pass
        else:
            raise AssertionError(f"质量轴应拒绝 {bad}")

    for bad in ([], [0.2], [1.1], [0.5, 0.5], [0.5, 0.6, 0.7, 0.8]):
        try:
            validate_axes([50], bad)
        except MatrixError:
            pass
        else:
            raise AssertionError(f"裁切轴应拒绝 {bad}")


# ---------------------------------------------------------------- API 端到端

def _fresh_client():
    tmp = tempfile.mkdtemp(prefix="wm-matrix-")
    os.environ["WM_DATA_DIR"] = tmp
    os.environ["WM_MAX_JOBS"] = "200"
    for mod in ("app.main", "app.archive"):
        sys.modules.pop(mod, None)
    from fastapi.testclient import TestClient
    from app.main import app, DATA_DIR  # noqa: E402

    return TestClient(app), Path(DATA_DIR)


def _embed_job(client, text="机密-矩阵-2026", strength="medium", seed=11):
    r = client.post(
        "/api/embed",
        files={"image": ("img.png", _make_png(seed=seed), "image/png")},
        data={"text": text, "strength": strength},
    )
    assert r.status_code == 200, r.text
    return r.json()["job_id"]


def test_matrix_endpoint_shape_and_consistency():
    client, data_dir = _fresh_client()
    job = _embed_job(client)

    r = client.post(
        f"/api/jobs/{job}/robustness-matrix",
        json={"jpeg_qualities": [90, 50, 30],
              "crop_fractions": [1.0, 0.75, 0.5]},
    )
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["job_id"] == job
    assert data["jpeg_qualities"] == [30, 50, 90]
    assert data["crop_fractions"] == [0.5, 0.75, 1.0]
    assert len(data["cells"]) == 9
    assert data["counts"]["cells"] == 9
    assert set(data["counts"]) == {"cells", "successful", "failed", "boundary"}
    assert data["counts"]["successful"] + data["counts"]["failed"] == 9

    # 每格字段齐全；成功格必须恢复出与原文一致的文本
    for c in data["cells"]:
        assert {"jpeg_quality", "crop_fraction", "ok", "detected", "text",
                "text_match", "metrics", "width", "height",
                "snapped_fraction_w", "snapped_fraction_h",
                "on_boundary"} <= set(c)
        assert c["ok"] is True
        if c["text_match"]:
            assert c["detected"] and c["text"] == "机密-矩阵-2026"
            assert c["metrics"]["confidence"] >= 0
        else:
            # 不一致要么未检测到，要么恢复出别的文本
            assert (not c["detected"]) or c["text"] != "机密-矩阵-2026"

    # 逐格「先裁切后压缩」：裁切比例对应吸附后的实际边长
    by = {(c["jpeg_quality"], c["crop_fraction"]): c for c in data["cells"]}
    assert by[(50, 1.0)]["width"] == 384
    assert by[(50, 0.5)]["width"] <= round(384 * 0.5 / 8) * 8
    assert by[(50, 0.5)]["width"] >= 128

    # 服务端标记的边界与纯逻辑结果一致
    from app.matrix import success_boundary
    expect = success_boundary(data["cells"])
    got = {(b["jpeg_quality"], b["crop_fraction"]) for b in data["boundary"]}
    assert got == expect
    assert data["counts"]["boundary"] == len(expect)
    for c in data["cells"]:
        assert c["on_boundary"] == (
            (c["jpeg_quality"], c["crop_fraction"]) in expect)


def test_matrix_does_not_modify_job_or_archive():
    client, data_dir = _fresh_client()
    job = _embed_job(client)

    # 先做一次普通攻击 + 检测，留下 target.png / 攻击历史 / 检测归档
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
    target_hash_before = (job_dir / "target.png").read_bytes()
    assert not (data_dir / "archive.log.jsonl").exists()

    # 用与现有攻击不同的参数跑矩阵
    r = client.post(
        f"/api/jobs/{job}/robustness-matrix",
        json={"jpeg_qualities": [95, 35], "crop_fractions": [1.0, 0.5]},
    )
    assert r.status_code == 200, r.text

    # 任务目录文件集合 / mtime 不变；meta、target 内容不变
    after = {p.relative_to(job_dir).as_posix(): p.stat().st_mtime_ns
             for p in job_dir.rglob("*") if p.is_file()}
    assert after == before
    assert (job_dir / "meta.json").read_text("utf-8") == meta_before
    assert (job_dir / "target.png").read_bytes() == target_hash_before

    # 没有新增检测记录 / 标注图 / 归档
    det_files = sorted(p.name for p in (job_dir / "detections").iterdir())
    assert det_files == [f"{det_id}.json", f"{det_id}.png",
                         f"{det_id}.source.png"]
    assert not (data_dir / "archive.log.jsonl").exists()
    assert not (data_dir / "reports").exists()

    # 再跑一次现有攻击/检测，矩阵不影响原流程
    r2 = client.post(
        f"/api/jobs/{job}/robustness-matrix",
        json={"jpeg_qualities": [95, 35], "crop_fractions": [1.0, 0.5]},
    )
    assert r2.status_code == 200
    cells1 = {(c["jpeg_quality"], c["crop_fraction"]): c["text_match"]
              for c in r.json()["cells"]}
    cells2 = {(c["jpeg_quality"], c["crop_fraction"]): c["text_match"]
              for c in r2.json()["cells"]}
    assert cells1 == cells2  # 同一份原图重复评测结果一致


def test_matrix_crop_then_compress_independence():
    """每格都从同一份含水印原图独立出发：极端格之间互不影响。"""
    client, _ = _fresh_client()
    job = _embed_job(client)
    r = client.post(
        f"/api/jobs/{job}/robustness-matrix",
        json={"jpeg_qualities": [100, 25], "crop_fractions": [1.0, 0.3]},
    )
    assert r.status_code == 200, r.text
    data = r.json()
    # 原始无攻击（Q100 + 100%）必须成功，且其图片尺寸未被裁切
    cells = {(c["jpeg_quality"], c["crop_fraction"]): c for c in data["cells"]}
    base = cells[(100, 1.0)]
    assert base["text_match"] is True, base["reason"]
    assert base["width"] == 384 and base["height"] == 384
    # 最狠格尺寸为 30% 吸附（384*0.3=115.2 → 120 → 提升到最小 128）
    hard = cells[(25, 0.3)]
    assert hard["ok"] is True
    assert hard["width"] >= 128


def test_matrix_validation_and_missing_job():
    client, _ = _fresh_client()
    job = _embed_job(client)

    # 空轴
    r = client.post(f"/api/jobs/{job}/robustness-matrix",
                    json={"jpeg_qualities": [], "crop_fractions": [0.5]})
    assert r.status_code == 400
    # 超过 3 个
    r = client.post(f"/api/jobs/{job}/robustness-matrix",
                    json={"jpeg_qualities": [90, 70, 50, 30],
                          "crop_fractions": [0.5]})
    assert r.status_code == 400
    # 越界
    assert client.post(
        f"/api/jobs/{job}/robustness-matrix",
        json={"jpeg_qualities": [15], "crop_fractions": [0.5]}
    ).status_code == 400
    assert client.post(
        f"/api/jobs/{job}/robustness-matrix",
        json={"jpeg_qualities": [50], "crop_fractions": [0.2]}
    ).status_code == 400
    # 重复
    assert client.post(
        f"/api/jobs/{job}/robustness-matrix",
        json={"jpeg_qualities": [50, 50], "crop_fractions": [0.5]}
    ).status_code == 400
    # 缺字段（pydantic 422）
    assert client.post(
        f"/api/jobs/{job}/robustness-matrix",
        json={"jpeg_qualities": [50]}
    ).status_code == 422
    # 任务不存在
    assert client.post(
        f"/api/jobs/000000000000/robustness-matrix",
        json={"jpeg_qualities": [50], "crop_fractions": [0.5]}
    ).status_code == 404


def test_matrix_rejected_for_upload_only_job():
    client, data_dir = _fresh_client()
    emb = client.post(
        "/api/embed",
        files={"image": ("img.png", _make_png(), "image/png")},
        data={"text": "盲检测", "strength": "medium"}).json()
    wm_bytes = (data_dir / emb["job_id"] / "watermarked.png").read_bytes()
    det = client.post("/api/detect",
                      files={"image": ("up.png", wm_bytes, "image/png")})
    assert det.status_code == 200
    up_job = det.json()["job_id"]
    assert det.json()["upload_only"] is True

    r = client.post(
        f"/api/jobs/{up_job}/robustness-matrix",
        json={"jpeg_qualities": [70], "crop_fractions": [0.75]})
    assert r.status_code == 404


if __name__ == "__main__":
    import inspect
    sys.modules.pop("app.main", None)
    fns = [v for k, v in sorted(globals().items())
           if k.startswith("test_") and inspect.isfunction(v)]
    for fn in fns:
        fn()
        print(f"[PASS] {fn.__name__}")
    print(f"\n{len(fns)} 个鲁棒性矩阵测试全部通过 ✔")
