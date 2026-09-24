"""检测报告可验证归档：端到端 API + 清单/哈希链校验。

运行（需项目依赖）：
    WM_DATA_DIR 由本测试自动指向临时目录；
    python -m pytest tests/test_reports.py -q
或直接：
    python tests/test_reports.py
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


def _make_png(size=384) -> bytes:
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


def _fresh_client():
    tmp = tempfile.mkdtemp(prefix="wm-reports-")
    os.environ["WM_DATA_DIR"] = tmp
    os.environ["WM_MAX_JOBS"] = "200"

    # main 在 import 时读取环境变量
    for mod in ("app.main", "app.archive"):
        sys.modules.pop(mod, None)
    from fastapi.testclient import TestClient
    from app.main import app, DATA_DIR  # noqa: E402

    return TestClient(app), Path(DATA_DIR)


def _embed_detect_attack(client, text="机密-报告-2026"):
    r = client.post(
        "/api/embed",
        files={"image": ("img.png", _make_png(), "image/png")},
        data={"text": text, "strength": "medium"},
    )
    assert r.status_code == 200, r.text
    job_id = r.json()["job_id"]

    r = client.post("/api/attack", json={
        "job_id": job_id, "jpeg_quality": 70, "crop_fraction": 0.75})
    assert r.status_code == 200, r.text

    r = client.post("/api/detect", data={"job_id": job_id})
    assert r.status_code == 200, r.text
    return job_id, r.json()


# ---------------------------------------------------------------- 用例

def test_report_lifecycle_verify_ok():
    client, data_dir = _fresh_client()
    job_id, det = _embed_detect_attack(client)
    assert det["detected"], det["reason"]
    assert det["detection_id"].startswith("det_")

    # 再做一次检测（同一受攻击图的第二次检测，独立快照）
    det2 = client.post("/api/detect", data={"job_id": job_id}).json()
    assert det2["job_id"] == job_id
    assert det2["detection_id"] != det["detection_id"]

    # 生成报告（多检测）
    r = client.post(f"/api/jobs/{job_id}/reports",
                    json={"detection_ids": [det["detection_id"],
                                            det2["detection_id"]]})
    assert r.status_code == 200, r.text
    summary = r.json()
    rid = summary["report_id"]
    assert rid.startswith("rep_")
    assert summary["detection_count"] == 2
    assert summary["archive"]["seq"] == 1
    assert summary["archive"]["previous_entry_sha256"] is None

    # 读取报告：自洽性
    r = client.get(f"/api/reports/{rid}")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("application/json")
    report = r.json()
    assert report["schema_version"] == "1.0"
    assert report["job_id"] == job_id
    assert report["embed"]["text"] == "机密-报告-2026"
    assert report["embed"]["algorithm"]["tile_px"] == 128
    assert report["detections"][0]["attack_history"]
    roles = {ref["role"] for d in report["detections"]
             for ref in d["references"]}
    assert {"original", "watermarked", "attacked",
            "detection_marked", "uploaded"} & roles
    assert "attacked" in roles

    # 清单项都存在，且报告自身条目大小等于实际文件
    p = data_dir / "reports" / rid / "report.json"
    self_entry = [e for e in report["manifest"]["entries"]
                  if e["path"].endswith("report.json")][0]
    assert self_entry["bytes"] == p.stat().st_size

    # 所有引用都可下载（检测资产新路由）
    for d in report["detections"]:
        for ref in d["references"]:
            assert ref["exists"]
            rr = client.get(ref["url"])
            assert rr.status_code == 200, (ref["url"], rr.status_code)
            assert rr.headers["content-type"] == "image/png"

    # 校验通过
    r = client.post(f"/api/reports/{rid}/verify")
    assert r.status_code == 200, r.text
    v = r.json()
    assert v["valid"] is True
    assert v["first_issue"] is None
    assert v["archive"]["found"] is True
    assert v["archive"]["chain_ok"] is True
    assert v["archive"]["seq"] == 1
    assert v["entries_total"] == len(report["manifest"]["entries"])


def test_verify_detects_tampered_file():
    client, data_dir = _fresh_client()
    job_id, det = _embed_detect_attack(client)
    rid = client.post(
        f"/api/jobs/{job_id}/reports",
        json={"detection_ids": [det["detection_id"]]}).json()["report_id"]

    # 篡改被引用的受攻击快照
    victim = next(
        data_dir.glob(f"{job_id}/detections/*.source.png"))
    victim.write_bytes(victim.read_bytes()[:-50] + b"\x00" * 50)

    r = client.post(f"/api/reports/{rid}/verify")
    assert r.status_code == 409
    v = r.json()
    assert v["valid"] is False
    issue = v["first_issue"]
    assert issue["kind"] == "digest_mismatch"
    assert issue["path"].endswith(".source.png")
    assert issue["expected"] != issue["actual"]


def test_verify_detects_missing_file():
    client, data_dir = _fresh_client()
    job_id, det = _embed_detect_attack(client)
    rid = client.post(
        f"/api/jobs/{job_id}/reports",
        json={"detection_ids": [det["detection_id"]]}).json()["report_id"]

    # 删除一个被引用文件
    victim = next(data_dir.glob(f"{job_id}/original.png"))
    victim.unlink()

    v = client.post(f"/api/reports/{rid}/verify").json()
    assert v["valid"] is False
    issue = v["first_issue"]
    assert issue["kind"] == "missing"
    assert issue["path"] == f"{job_id}/original.png"
    assert issue["actual"] is None


def test_verify_detects_tampered_report_body():
    client, data_dir = _fresh_client()
    job_id, det = _embed_detect_attack(client)
    rid = client.post(
        f"/api/jobs/{job_id}/reports",
        json={"detection_ids": [det["detection_id"]]}).json()["report_id"]

    p = data_dir / "reports" / rid / "report.json"
    report = json.loads(p.read_text("utf-8"))
    report["embed"]["text"] = "被篡改的文本"
    p.write_text(json.dumps(report, ensure_ascii=False, indent=2), "utf-8")

    v = client.post(f"/api/reports/{rid}/verify").json()
    assert v["valid"] is False
    assert v["first_issue"]["kind"] == "digest_mismatch"
    assert v["first_issue"]["path"].endswith("report.json")
    assert v["archive"]["chain_ok"] is False
    assert v["archive"]["issue"]["kind"] == "report_digest_mismatch"


def test_archive_hash_chain_second_report():
    client, data_dir = _fresh_client()
    job_id, det = _embed_detect_attack(client)
    r1 = client.post(
        f"/api/jobs/{job_id}/reports",
        json={"detection_ids": [det["detection_id"]]}).json()
    rid1, e1 = r1["report_id"], r1["archive"]["entry_sha256"] \
        if "entry_sha256" in r1["archive"] else None

    det_b = client.post("/api/detect", data={"job_id": job_id}).json()
    r2 = client.post(
        f"/api/jobs/{job_id}/reports",
        json={"detection_ids": [det_b["detection_id"]]}).json()
    assert r2["archive"]["seq"] == 2
    # 第二条链接到第一条（通过 verify 结果交叉确认）
    v1 = client.post(f"/api/reports/{rid1}/verify").json()
    assert v1["archive"]["chain_ok"] is True
    assert v1["archive"]["log_entries"] == 2

    # 日志为仅追加 JSONL，两条都在
    log_lines = (data_dir / "archive.log.jsonl").read_text("utf-8").strip().splitlines()
    assert len(log_lines) == 2
    first, second = json.loads(log_lines[0]), json.loads(log_lines[1])
    assert first["seq"] == 1 and second["seq"] == 2
    assert second["previous_entry_sha256"] == first["entry_sha256"]

    # 删除第一条 -> 哈希链断裂
    with (data_dir / "archive.log.jsonl").open("w", encoding="utf-8") as f:
        f.write(log_lines[1] + "\n")
    v = client.post(f"/api/reports/{r2['report_id']}/verify").json()
    assert v["archive"]["chain_ok"] is False
    assert v["archive"]["issue"]["kind"] == "chain_broken"


def test_upload_only_detection_is_archivable():
    client, data_dir = _fresh_client()
    # 直接上传含水印字节做盲检测
    job_id, _ = None, None
    emb = client.post(
        "/api/embed",
        files={"image": ("img.png", _make_png(), "image/png")},
        data={"text": "盲检测", "strength": "medium"}).json()
    wm_bytes = (data_dir / emb["job_id"] / "watermarked.png").read_bytes()
    r = client.post("/api/detect",
                    files={"image": ("up.png", wm_bytes, "image/png")})
    assert r.status_code == 200
    det = r.json()
    assert det["upload_only"] is True
    assert det["source"] == "uploaded"

    rid = client.post(
        f"/api/jobs/{det['job_id']}/reports",
        json={"detection_ids": [det["detection_id"]]}).json()["report_id"]
    report = client.get(f"/api/reports/{rid}").json()
    assert report["embed"] is None
    roles = {ref["role"] for ref in report["detections"][0]["references"]}
    assert roles == {"uploaded", "detection_marked"}
    assert client.post(f"/api/reports/{rid}/verify").json()["valid"] is True


def test_report_errors_and_download_header():
    client, data_dir = _fresh_client()
    job_id, det = _embed_detect_attack(client)

    # 空选择
    r = client.post(f"/api/jobs/{job_id}/reports", json={"detection_ids": []})
    assert r.status_code == 400

    # 不存在的检测
    r = client.post(
        f"/api/jobs/{job_id}/reports",
        json={"detection_ids": ["det_" + "0" * 12]})
    assert r.status_code == 404

    # 非法 ID
    r = client.post("/api/jobs/nope/reports", json={"detection_ids": ["x"]})
    assert r.status_code == 404

    # 重复选择
    r = client.post(
        f"/api/jobs/{job_id}/reports",
        json={"detection_ids": [det["detection_id"], det["detection_id"]]})
    assert r.status_code == 400

    # 不存在的报告
    assert client.get("/api/reports/rep_" + "0" * 12).status_code == 404

    rid = client.post(
        f"/api/jobs/{job_id}/reports",
        json={"detection_ids": [det["detection_id"]]}).json()["report_id"]
    r = client.get(f"/api/reports/{rid}?download=1")
    assert r.status_code == 200
    cd = r.headers["content-disposition"]
    assert "attachment" in cd and rid in cd


if __name__ == "__main__":
    import inspect
    sys.modules.pop("app.main", None)
    fns = [v for k, v in sorted(globals().items())
           if k.startswith("test_") and inspect.isfunction(v)]
    for fn in fns:
        fn()
        print(f"[PASS] {fn.__name__}")
    print(f"\n{len(fns)} 个归档接口测试全部通过 ✔")
