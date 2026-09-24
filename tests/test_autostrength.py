"""自动选强度：逐档评测纯逻辑 + /api/embed-auto 端到端。

覆盖：
  * 参数校验（质量 20~100 / 裁切 30%~100% / PSNR 20~60）；
  * 低→中→高顺序、首个「逐字一致且 PSNR 达标」档胜出；
  * 三档均不过时接口返回 passed=false 且**不创建任务、不写任何文件**；
  * 通过时沿用现有任务保存/下载流程（original/watermarked 可下载、
    meta 记录 auto_selection，攻击/检测/矩阵流程照常）；
  * 手动 /api/embed 行为不变。

运行：
    WM_DATA_DIR 由本测试自动指向临时目录；
    python -m pytest tests/test_autostrength.py -q
或直接：
    python tests/test_autostrength.py
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


def _make_png(size=384, seed=3, smooth=True) -> bytes:
    rng = np.random.RandomState(seed)
    y, x = np.mgrid[0:size, 0:size]
    if smooth:
        base = (
            160
            + 18 * np.sin(2 * np.pi * x / 300)
            + 10 * np.cos(2 * np.pi * y / 260)
            + rng.normal(0, 3, (size, size))
        ).clip(0, 255).astype(np.uint8)
        rgb = np.stack([base, base, base], axis=-1)
    else:
        base = (
            128
            + 40 * np.sin(2 * np.pi * x / 130)
            + 30 * np.cos(2 * np.pi * y / 95)
            + rng.normal(0, 12, (size, size))
        ).clip(0, 255).astype(np.uint8)
        rgb = np.stack([base, base * 0 + 128, 255 - base], axis=-1)
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="PNG")
    return buf.getvalue()


# ---------------------------------------------------------------- 参数校验

def test_validate_params():
    from app.autostrength import (
        AutoStrengthError, MIN_PSNR, MAX_PSNR, validate_params,
    )

    assert validate_params(20, 0.3, 20.0) == (20, 0.3, 20.0)
    assert validate_params(100, 1.0, 60.0) == (100, 1.0, 60.0)
    assert validate_params(50, 0.75, 33.5)[2] == 33.5

    for q in (19, 101, 19.5, True, "50", None):
        try:
            validate_params(q, 0.5, 30.0)
        except AutoStrengthError:
            pass
        else:
            raise AssertionError(f"质量应拒绝 {q!r}")

    for f in (0.29, 1.01, "0.5", True):
        try:
            validate_params(50, f, 30.0)
        except AutoStrengthError:
            pass
        else:
            raise AssertionError(f"裁切比例应拒绝 {f!r}")

    for p in (MIN_PSNR - 0.1, MAX_PSNR + 0.1, "30", True):
        try:
            validate_params(50, 0.5, p)
        except AutoStrengthError:
            pass
        else:
            raise AssertionError(f"PSNR 应拒绝 {p!r}")


# ---------------------------------------------------------------- 逐档选择纯逻辑

def _img():
    return np.array(Image.open(io.BytesIO(_make_png())).convert("RGB"))


def test_evaluate_tiers_easy_picks_low_in_order():
    """温和攻击下三档都能恢复，顺序首个（低）胜出。"""
    from app.autostrength import evaluate_tiers

    chosen, tiers = evaluate_tiers(_img(), "机密-2026", 90, 1.0, 20.0)
    assert chosen == "low"
    assert [t.strength for t in tiers] == ["low", "medium", "high"]
    assert all(t.text_match for t in tiers)
    # 低档 PSNR 最高（强度越弱越不可见）
    psnrs = [t.embed.psnr_db for t in tiers]
    assert psnrs == sorted(psnrs, reverse=True)


def test_evaluate_tiers_picks_medium_when_low_fails():
    """平滑图 Q40 + 裁切 50%：低档恢复失败、中档通过 → 选中档。"""
    from app.autostrength import evaluate_tiers

    chosen, tiers = evaluate_tiers(_img(), "机密-2026", 40, 0.5, 30.0)
    by = {t.strength: t for t in tiers}
    assert by["low"].text_match is False
    assert by["medium"].text_match is True and by["medium"].psnr_ok is True
    assert chosen == "medium"
    # 攻击后尺寸为 50% 吸附（384 → 192）
    assert by["medium"].width == 192 and by["medium"].height == 192


def test_evaluate_tiers_picks_high_on_hard_textured():
    """纹理图 Q30 + 裁切 50%：只有高档恢复 → 选高档。"""
    from app.autostrength import evaluate_tiers

    img = np.array(Image.open(
        io.BytesIO(_make_png(seed=11, smooth=False))).convert("RGB"))
    chosen, tiers = evaluate_tiers(img, "机密-2026", 30, 0.5, 30.0)
    assert chosen == "high"
    assert [t.text_match for t in tiers] == [False, False, True]


def test_evaluate_tiers_all_fail_returns_none():
    """Q20 + 裁切 30% 三档均无法恢复：chosen 为 None。"""
    from app.autostrength import evaluate_tiers

    chosen, tiers = evaluate_tiers(_img(), "机密-2026", 20, 0.3, 30.0)
    assert chosen is None
    assert len(tiers) == 3
    assert all(not t.text_match for t in tiers)


def test_evaluate_tiers_psnr_gate_blocks_all():
    """恢复都成功但 PSNR 下限 60 dB 时三档均不达标 → chosen 为 None。"""
    from app.autostrength import evaluate_tiers

    chosen, tiers = evaluate_tiers(_img(), "机密-2026", 90, 1.0, 60.0)
    assert chosen is None
    assert all(t.text_match and not t.psnr_ok for t in tiers)


def test_evaluate_tiers_uses_same_original():
    """每档都从同一原图出发：各档 PSNR 与直接嵌入的结果一致。"""
    from app import dctwm as wm
    from app.autostrength import evaluate_tiers

    img = _img()
    text = "逐字一致-AB12"
    _, tiers = evaluate_tiers(img, text, 50, 0.75, 20.0)
    for t in tiers:
        direct = wm.embed_watermark(img, text, strength=t.strength)
        assert t.embed.psnr_db == direct.psnr_db
        assert t.embed.image.shape == img.shape


def test_evaluate_tiers_text_verbatim_required():
    """恢复文本必须逐字一致：不同文本不算通过（用 monkeypatch 模拟错文）。"""
    from app import dctwm as wm
    from app import autostrength as mod

    orig = wm.detect_watermark

    def fake_detect(rgb):
        r = orig(rgb)
        if r.detected:                          # 模拟恢复出错文
            r.detected = True
            object.__setattr__(r, "text", "别的文本")
        return r

    wm.detect_watermark = fake_detect
    try:
        chosen, tiers = mod.evaluate_tiers(_img(), "机密-2026", 90, 1.0, 20.0)
    finally:
        wm.detect_watermark = orig
    assert chosen is None
    assert all(t.detect.detected and not t.text_match for t in tiers)


# ---------------------------------------------------------------- API 端到端

def _fresh_client():
    tmp = tempfile.mkdtemp(prefix="wm-auto-")
    os.environ["WM_DATA_DIR"] = tmp
    os.environ["WM_MAX_JOBS"] = "200"
    for mod in ("app.main", "app.archive"):
        sys.modules.pop(mod, None)
    from fastapi.testclient import TestClient
    from app.main import app, DATA_DIR  # noqa: E402

    return TestClient(app), Path(DATA_DIR)


def _post_auto(client, **over):
    data = {"text": "机密-自动-2026", "jpeg_quality": "40",
            "crop_fraction": "0.5", "min_psnr_db": "30"}
    data.update(over)
    return client.post(
        "/api/embed-auto",
        files={"image": ("img.png", _make_png(), "image/png")},
        data=data,
    )


def test_auto_endpoint_success_creates_job_and_artifacts():
    client, data_dir = _fresh_client()
    r = _post_auto(client)
    assert r.status_code == 200, r.text
    j = r.json()
    assert j["passed"] is True
    assert j["mode"] == "auto"
    assert j["selected_strength"] == "medium"
    assert j["job_id"] == j["id"]
    assert j["strength"] == "medium"                  # meta 沿用所选档
    assert len(j["tiers"]) == 3
    low, med, high = j["tiers"]
    assert low["strength"] == "low" and low["text_match"] is False
    assert med["text_match"] is True and med["psnr_ok"] is True
    assert j["selected_tier"]["strength"] == "medium"
    assert j["request"] == {
        "jpeg_quality": 40, "crop_fraction": 0.5, "min_psnr_db": 30.0}

    job_dir = data_dir / j["job_id"]
    assert (job_dir / "original.png").is_file()
    assert (job_dir / "watermarked.png").is_file()
    meta = json.loads((job_dir / "meta.json").read_text("utf-8"))
    assert meta["embed_mode"] == "auto"
    sel = meta["auto_selection"]
    assert sel["selected_strength"] == "medium"
    assert [t["strength"] for t in sel["tiers"]] == ["low", "medium", "high"]

    # 现有下载流程沿用
    d = client.get(f"/api/jobs/{j['job_id']}/watermarked")
    assert d.status_code == 200 and d.headers["content-type"] == "image/png"

    # 后续攻击 / 检测流程照常
    a = client.post("/api/attack",
                    json={"job_id": j["job_id"], "jpeg_quality": 40,
                          "crop_fraction": 0.5})
    assert a.status_code == 200
    det = client.post("/api/detect", data={"job_id": j["job_id"]})
    assert det.status_code == 200
    assert det.json()["text"] == "机密-自动-2026"

    # 矩阵评测同样可用
    mx = client.post(
        f"/api/jobs/{j['job_id']}/robustness-matrix",
        json={"jpeg_qualities": [50], "crop_fractions": [0.75]})
    assert mx.status_code == 200
    assert mx.json()["cells"][0]["text_match"] is True


def test_auto_endpoint_failure_creates_no_job():
    client, data_dir = _fresh_client()
    existing = sorted(p.name for p in data_dir.glob("*"))
    r = _post_auto(client, jpeg_quality="20", crop_fraction="0.3")
    assert r.status_code == 200, r.text
    j = r.json()
    assert j["passed"] is False
    assert j["job_id"] is None
    assert j["selected_strength"] is None
    assert len(j["tiers"]) == 3
    # 逐档实测结果齐全（PSNR、恢复结果、失败原因）
    for t in j["tiers"]:
        assert {"strength", "psnr_db", "psnr_ok", "detected", "text",
                "text_match", "reason", "metrics", "width", "height"} <= set(t)
        assert t["psnr_db"] > 0
    # 数据目录没有新增任何任务文件
    assert sorted(p.name for p in data_dir.glob("*")) == existing

    # 同样：PSNR 门槛过高也不建任务
    r2 = _post_auto(client, jpeg_quality="90", crop_fraction="1",
                    min_psnr_db="60")
    assert r2.json()["passed"] is False
    assert r2.json()["job_id"] is None
    assert sorted(p.name for p in data_dir.glob("*")) == existing


def test_auto_endpoint_validation_and_bad_upload():
    client, _ = _fresh_client()
    for bad in (
        {"jpeg_quality": "19"},
        {"jpeg_quality": "101"},
        {"crop_fraction": "0.2"},
        {"crop_fraction": "1.2"},
        {"min_psnr_db": "15"},
        {"min_psnr_db": "61"},
    ):
        r = _post_auto(client, **bad)
        assert r.status_code == 400, (bad, r.text)

    # 空文本
    r = _post_auto(client, text="  ")
    assert r.status_code == 400

    # 缺字段（FastAPI 422）
    r = client.post(
        "/api/embed-auto",
        files={"image": ("img.png", _make_png(), "image/png")},
        data={"text": "x", "jpeg_quality": "50", "crop_fraction": "0.5"},
    )
    assert r.status_code == 422


def test_auto_endpoint_text_too_long():
    client, data_dir = _fresh_client()
    r = _post_auto(client, text="字" * 40)     # 120 字节 > 45 字节容量
    assert r.status_code == 400
    assert not list(data_dir.glob("*"))


def test_manual_embed_unchanged():
    """手动模式 /api/embed 行为与字段保持原样。"""
    client, data_dir = _fresh_client()
    r = client.post(
        "/api/embed",
        files={"image": ("img.png", _make_png(), "image/png")},
        data={"text": "手动档", "strength": "high"},
    )
    assert r.status_code == 200, r.text
    j = r.json()
    assert j["job_id"] == j["id"]
    assert j["strength"] == "high"
    assert "embed_mode" not in j
    assert "tiers" not in j
    meta = json.loads((data_dir / j["job_id"] / "meta.json").read_text("utf-8"))
    assert "auto_selection" not in meta and "embed_mode" not in meta


if __name__ == "__main__":
    import inspect
    sys.modules.pop("app.main", None)
    fns = [v for k, v in sorted(globals().items())
           if k.startswith("test_") and inspect.isfunction(v)]
    for fn in fns:
        fn()
        print(f"[PASS] {fn.__name__}")
    print(f"\n{len(fns)} 个自动选强度测试全部通过 ✔")
