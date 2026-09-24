"""检测报告可验证归档。

报告为规范化 JSON，内含一次或多次检测的恢复文本、置信度、嵌入/攻击参数与
原图 / 含水印图 / 受攻击图（检测输入快照）/ 失真标注图的引用。

完整性设计
==========
* ``manifest.entries`` 列出报告引用的每个文件与报告自身的 SHA-256；
  报告自身摘要 = SHA-256(canonical(report 删除 ``manifest`` 键后的内容))，
  从而不存在“摘要依赖自身”的循环。
* 校验时按清单顺序重算摘要，返回**首个**缺失或摘要不一致项。
* 每次生成报告都会向 ``archive.log.jsonl`` 追加一条记录（仅追加），
  记录之间以 ``previous_entry_sha256`` 形成哈希链，单条记录自身也带摘要。

所有归档文件都位于任务数据目录（``WM_DATA_DIR``），随 compose 数据卷持久化。
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

from fastapi import HTTPException

REPORT_SCHEMA_VERSION = "1.0"
GENERATOR_NAME = "wm-bench"
GENERATOR_VERSION = "1.2.0"
MAX_REPORT_DETECTIONS = 20

REPORT_ID_RE = re.compile(r"^rep_[0-9a-f]{12}$")
DETECTION_ID_RE = re.compile(r"^det_[0-9a-f]{12}$")
JOB_ID_RE = re.compile(r"^[0-9a-f]{12}$")

ARCHIVE_LOG_REL = Path("archive.log.jsonl")
REPORT_REL_TMPL = "reports/{report_id}/report.json"

_archive_lock = threading.Lock()


# ---------------------------------------------------------------- 基础工具

def utc_now() -> tuple[float, str]:
    ts = time.time()
    iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    return ts, iso


def canonical_json(obj) -> bytes:
    """规范化序列化：UTF-8、键排序、无空白、不允许 NaN/Infinity。"""
    return json.dumps(
        obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _safe_path(root: Path, rel: str) -> Path:
    """把相对路径限制在数据目录内，杜绝 ``..`` 逃逸。"""
    root = root.resolve()
    p = (root / rel).resolve()
    if p != root and root not in p.parents:
        raise HTTPException(status_code=400, detail="非法的归档路径。")
    return p


def validate_ids(job_id: str, detection_ids: list[str]) -> None:
    if not JOB_ID_RE.match(job_id or ""):
        raise HTTPException(status_code=404, detail="任务不存在或已过期。")
    if not detection_ids:
        raise HTTPException(status_code=400, detail="请至少选择一次检测结果。")
    if len(detection_ids) > MAX_REPORT_DETECTIONS:
        raise HTTPException(
            status_code=400,
            detail=f"单份报告最多收录 {MAX_REPORT_DETECTIONS} 次检测。",
        )
    if any(not DETECTION_ID_RE.match(d or "") for d in detection_ids):
        raise HTTPException(status_code=400, detail="检测记录 ID 格式非法。")
    if len(set(detection_ids)) != len(detection_ids):
        raise HTTPException(status_code=400, detail="检测记录不能重复选择。")


# ---------------------------------------------------------------- 报告生成

def _file_entry(root: Path, rel: str) -> dict | None:
    p = _safe_path(root, rel)
    if not p.is_file():
        return None
    return {
        "path": rel,
        "sha256": sha256_file(p),
        "bytes": p.stat().st_size,
    }


def _build_references(root: Path, job_id: str, record: dict,
                      upload_only: bool) -> list[dict]:
    """一次检测的文件引用；不存在的文件以 exists=false 保留占位。"""
    did = record["detection_id"]
    source = record.get("source", "uploaded")
    refs: list[dict] = []

    def add(role: str, rel: str, url: str):
        p = _safe_path(root, rel)
        refs.append({
            "role": role,
            "path": rel,
            "url": url,
            "media_type": "image/png",
            "exists": p.is_file(),
            "bytes": p.stat().st_size if p.is_file() else None,
        })

    if upload_only:
        add("uploaded",
            f"{job_id}/detections/{did}.source.png",
            f"/api/jobs/{job_id}/detections/{did}/source")
    else:
        add("original",
            f"{job_id}/original.png",
            f"/api/jobs/{job_id}/original")
        add("watermarked",
            f"{job_id}/watermarked.png",
            f"/api/jobs/{job_id}/watermarked")
        # 检测输入快照：该次检测实际分析的图（受攻击图 / 含水印图），
        # 独立快照避免被后续攻击覆盖。
        snap_role = "attacked" if source == "target" else "detected_source"
        add(snap_role,
            f"{job_id}/detections/{did}.source.png",
            f"/api/jobs/{job_id}/detections/{did}/source")

    if record.get("has_overlay"):
        add("detection_marked",
            f"{job_id}/detections/{did}.png",
            f"/api/jobs/{job_id}/detections/{did}/marked")
    return refs


def _read_detection(root: Path, job_id: str, det_id: str) -> dict:
    p = _safe_path(root, f"{job_id}/detections/{det_id}.json")
    if not p.is_file():
        raise HTTPException(
            status_code=404, detail=f"检测记录 {det_id} 不存在或已随任务清理。")
    try:
        rec = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail=f"检测记录 {det_id} 已损坏。")
    if rec.get("job_id") != job_id or rec.get("detection_id") != det_id:
        raise HTTPException(status_code=400, detail="检测记录与任务不匹配。")
    return rec


def _read_archive_tail(root: Path) -> tuple[int, str | None]:
    """返回 (下一条序号, 上一条 entry_sha256)。日志损坏则拒绝继续追加。"""
    log = _safe_path(root, str(ARCHIVE_LOG_REL))
    if not log.is_file():
        return 1, None
    seq = 0
    prev = None
    for line_no, line in enumerate(
            log.read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
            seq = int(entry["seq"])
            prev = entry["entry_sha256"]
        except (json.JSONDecodeError, KeyError, ValueError, TypeError):
            raise HTTPException(
                status_code=409,
                detail=(f"归档日志第 {line_no} 行已损坏或被截断，"
                        "拒绝追加新记录（请先运行校验定位问题）。"))
    return seq + 1, prev


def _append_log(root: Path, entry: dict) -> str:
    """向仅追加归档日志写入一条哈希链记录，返回 entry_sha256。"""
    entry_sha = sha256_bytes(canonical_json(entry))
    row = dict(entry)
    row["entry_sha256"] = entry_sha
    log = _safe_path(root, str(ARCHIVE_LOG_REL))
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False, sort_keys=True,
                           separators=(",", ":")) + "\n")
        f.flush()
        os.fsync(f.fileno())
    return entry_sha


def create_report(data_dir: Path, job_id: str, detection_ids: list[str],
                  algo_params: dict) -> dict:
    """生成规范化报告、计算清单、写文件并追加归档记录。"""
    validate_ids(job_id, detection_ids)
    root = Path(data_dir)
    meta_p = _safe_path(root, f"{job_id}/meta.json")
    if not meta_p.is_file():
        raise HTTPException(status_code=404, detail="任务不存在或已过期。")
    meta = json.loads(meta_p.read_text(encoding="utf-8"))
    upload_only = bool(meta.get("upload_only"))

    records = [_read_detection(root, job_id, d) for d in detection_ids]

    report_id = "rep_" + os.urandom(6).hex()
    ts, iso = utc_now()
    report_rel = REPORT_REL_TMPL.format(report_id=report_id)

    # 嵌入参数（独立检测任务没有嵌入信息）
    embed = None
    if not upload_only:
        embed = {
            "text": meta.get("text"),
            "strength": meta.get("strength"),
            "width": meta.get("width"),
            "height": meta.get("height"),
            "psnr_db": meta.get("psnr_db"),
            "full_tiles": meta.get("full_tiles"),
            "payload_bytes": meta.get("payload_bytes"),
            "capacity_bytes": meta.get("capacity_bytes"),
            "votes_per_bit": meta.get("votes_per_bit"),
            "created_at": meta.get("created_at"),
            "algorithm": algo_params,
        }

    detections: list[dict] = []
    manifest_map: dict[str, dict] = {}
    for rec in records:
        refs = _build_references(root, job_id, rec, upload_only)
        for ref in refs:
            rel = ref["path"]
            if ref["exists"] and rel not in manifest_map:
                entry = _file_entry(root, rel)
                if entry is not None:
                    manifest_map[rel] = entry
            ref["sha256"] = manifest_map.get(rel, {}).get("sha256")
        detections.append({
            "detection_id": rec["detection_id"],
            "source": rec.get("source"),
            "detected_at": rec.get("detected_at"),
            "detected_at_iso": rec.get("detected_at_iso"),
            "detected": rec.get("detected", False),
            "text": rec.get("text", ""),
            "reason": rec.get("reason", ""),
            "confidence": rec.get("metrics", {}).get("confidence"),
            "metrics": rec.get("metrics", {}),
            "attack_history": rec.get("attack_history", []),
            "orientation_search": rec.get("orientation_search"),
            "references": refs,
        })

    with _archive_lock:
        seq, prev_entry_sha = _read_archive_tail(root)
        report = {
            "schema_version": REPORT_SCHEMA_VERSION,
            "report_id": report_id,
            "job_id": job_id,
            "generated_at": ts,
            "generated_at_iso": iso,
            "generator": {
                "name": GENERATOR_NAME,
                "version": GENERATOR_VERSION,
            },
            "embed": embed,
            "detections": detections,
            "archive": {
                "log_path": str(ARCHIVE_LOG_REL),
                "seq": seq,
                "previous_entry_sha256": prev_entry_sha,
                "append_only": True,
            },
        }

        # 报告自身摘要：删除 manifest 后规范化哈希（无自引用循环）
        self_digest = sha256_bytes(canonical_json(report))

        entries = list(manifest_map.values())
        entries.append({
            "path": report_rel,
            "role": "report",
            "sha256": self_digest,
            "bytes": -1,                       # 定点迭代后回填
        })
        report["manifest"] = {
            "hash_algorithm": "sha-256",
            "canonical_form": "json:utf8:sort_keys:no_whitespace; "
                              "report_digest 为删除 manifest 键后的规范化摘要",
            "entries": entries,
        }

        # 定点求自条目字节数（bytes 位数变化通常 1~2 轮即收敛）
        text = json.dumps(report, ensure_ascii=False, indent=2)
        size = len(text.encode("utf-8"))
        for _ in range(6):
            entries[-1]["bytes"] = size
            text = json.dumps(report, ensure_ascii=False, indent=2)
            new_size = len(text.encode("utf-8"))
            if new_size == size:
                break
            size = new_size

        out = _safe_path(root, report_rel)
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_suffix(".json.tmp")
        tmp.write_text(text, encoding="utf-8")
        os.replace(tmp, out)

        log_entry = {
            "seq": seq,
            "logged_at": ts,
            "logged_at_iso": iso,
            "report_id": report_id,
            "job_id": job_id,
            "report_path": report_rel,
            "report_sha256": self_digest,
            "bytes": size,
            "detection_count": len(detections),
            "previous_entry_sha256": prev_entry_sha,
        }
        entry_sha = _append_log(root, log_entry)

    report["archive"]["entry_sha256"] = entry_sha
    report["download_url"] = f"/api/reports/{report_id}"
    return report


# ---------------------------------------------------------------- 读取与校验

def report_path(data_dir: Path, report_id: str) -> Path:
    if not REPORT_ID_RE.match(report_id or ""):
        raise HTTPException(status_code=404, detail="报告不存在。")
    p = _safe_path(Path(data_dir),
                   REPORT_REL_TMPL.format(report_id=report_id))
    if not p.is_file():
        raise HTTPException(status_code=404, detail="报告不存在或已被删除。")
    return p


def load_report(data_dir: Path, report_id: str) -> tuple[dict, bytes]:
    p = report_path(data_dir, report_id)
    raw = p.read_bytes()
    try:
        return json.loads(raw.decode("utf-8")), raw
    except (json.JSONDecodeError, UnicodeDecodeError):
        raise HTTPException(status_code=422, detail="报告 JSON 已损坏，无法解析。")


def _issue(path: str, kind: str, expected: str | int | None,
           actual: str | int | None) -> dict:
    return {
        "path": path, "kind": kind,
        "expected": expected, "actual": actual,
    }


def verify_report(data_dir: Path, report_id: str) -> dict:
    """重算清单并报告首个缺失 / 摘要不一致项；同时核验归档日志哈希链。"""
    root = Path(data_dir)
    report, raw = load_report(root, report_id)
    ts, iso = utc_now()

    manifest = report.get("manifest") or {}
    entries = manifest.get("entries") or []
    if not isinstance(entries, list) or not entries:
        raise HTTPException(status_code=422, detail="报告清单缺失或格式非法。")

    first_issue: dict | None = None
    checked = 0
    report_rel = REPORT_REL_TMPL.format(report_id=report_id)
    expected_self = None

    for entry in entries:
        rel = entry.get("path", "")
        expected = entry.get("sha256")
        if rel == report_rel:
            expected_self = expected
            continue                       # 自身摘要最后校验
        p = _safe_path(root, rel)
        if not p.is_file():
            if first_issue is None:
                first_issue = _issue(rel, "missing", expected, None)
            continue
        actual = sha256_file(p)
        checked += 1
        if actual != expected and first_issue is None:
            first_issue = _issue(rel, "digest_mismatch", expected, actual)

    # 报告自身摘要
    without_manifest = {k: v for k, v in report.items() if k != "manifest"}
    actual_self = sha256_bytes(canonical_json(without_manifest))
    if actual_self != expected_self and first_issue is None:
        first_issue = _issue(report_rel, "digest_mismatch",
                             expected_self, actual_self)

    valid = first_issue is None
    return {
        "report_id": report_id,
        "valid": valid,
        "checked_at": ts,
        "checked_at_iso": iso,
        "entries_total": len(entries),
        "files_checked": checked,
        "first_issue": first_issue,
        "archive": verify_archive_chain(root, report_id, actual_self),
    }


def verify_archive_chain(root: Path, report_id: str,
                         current_report_digest: str) -> dict:
    """核验仅追加日志：逐条 entry 摘要、前后链接、报告收录与摘要一致。"""
    log = _safe_path(root, str(ARCHIVE_LOG_REL))
    if not log.is_file():
        return {"found": False, "chain_ok": False,
                "issue": {"kind": "log_missing", "path": str(ARCHIVE_LOG_REL)}}

    prev = None
    seq = 0
    found_entry = None
    issue = None
    for line in log.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            if issue is None:
                issue = {"kind": "log_corrupt", "seq": seq + 1}
            break
        seq = int(entry.get("seq", seq + 1))
        stored = entry.get("entry_sha256")
        recomputed = sha256_bytes(canonical_json(
            {k: v for k, v in entry.items() if k != "entry_sha256"}))
        if recomputed != stored and issue is None:
            issue = {"kind": "entry_digest_mismatch", "seq": seq,
                     "expected": stored, "actual": recomputed}
        if entry.get("previous_entry_sha256") != prev and issue is None:
            issue = {"kind": "chain_broken", "seq": seq,
                     "expected": prev,
                     "actual": entry.get("previous_entry_sha256")}
        prev = stored
        if entry.get("report_id") == report_id:
            found_entry = entry

    if issue is None and found_entry is None:
        issue = {"kind": "report_not_in_archive", "report_id": report_id}
    if issue is None and found_entry.get("report_sha256") != current_report_digest:
        issue = {
            "kind": "report_digest_mismatch",
            "expected": found_entry.get("report_sha256"),
            "actual": current_report_digest,
        }

    return {
        "found": found_entry is not None,
        "chain_ok": issue is None,
        "seq": found_entry.get("seq") if found_entry else None,
        "log_path": str(ARCHIVE_LOG_REL),
        "log_entries": seq,
        "entry_sha256": found_entry.get("entry_sha256") if found_entry else None,
        "issue": issue,
    }
