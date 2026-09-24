"""抗裁切水印实验台 —— FastAPI 后端。

接口：
  GET  /                 前端页面
  POST /api/embed        手动模式：上传图片 + 文本 + 指定强度 -> 含水印 PNG
  POST /api/embed-auto   自动选强度：按低/中/高逐档试嵌，每档先裁切后 JPEG
                          压缩并盲检，取首个「恢复文本逐字一致且 PSNR 达标」档
                          保存为任务；三档均不过则不建任务、返回逐档实测结果
  POST /api/attack       对任务图片施加 JPEG 重压缩 / 中心裁切
  POST /api/detect       上传图片或引用任务图片 -> 水印检测结果 + 失真叠加图；
                          独立上传可选 auto_orient=true 做 8 方向自动检索
  POST /api/jobs/{id}/robustness-matrix
                          只读评测：至多 3 个 JPEG 质量 × 至多 3 个裁切比例，
                          每格从同一份含水印原图独立「先裁切后压缩」再检测，
                          返回逐格恢复文本/一致性/指标与二维最严成功边界；
                          不改任务图片、攻击历史与检测归档
  POST /api/jobs/{id}/locate
                          传播副本来源定位（只读）：上传一张由 watermarked.png
                          裁切/缩放/JPEG 转存得到的图片，SIFT 特征匹配 +
                          RANSAC 单应估计副本四角在参照图中的位置；仅在内点
                          数量/比例、投影四边形面积与边界均可信时返回四角、
                          覆盖率与标注预览，否则返回「无法定位」及原因
  POST /api/privacy-check  上传 PNG/JPEG -> 只读解析 EXIF 拍摄时间/相机/GPS（不落盘、不建任务）
  POST /api/privacy-export 上传 PNG/JPEG -> EXIF Orientation 摆正像素 + 全新 PNG 重编码
                          （清除 EXIF/GPS/XMP/ICC/PNG 文本，保留透明通道），导出前
                          重新解析生成文件复查；复查不通过返回错误且不提供下载。
                          全程内存，不创建/改写任何水印任务
  GET  /api/jobs/{id}/{kind}  任务文件下载（original/watermarked/target/marked）
  GET  /api/jobs/{id}/detections/{det_id}/{kind}  某次检测的输入快照 / 标注图
  POST /api/jobs/{id}/reports   选择一次或多次检测 -> 可验证 JSON 报告并追加归档
  GET  /api/reports/{report_id}                  读取 / 下载归档报告
  POST /api/reports/{report_id}/verify           重算 SHA-256 清单并校验归档链
"""

from __future__ import annotations

import json
import os
import time
import uuid
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel

from . import archive, autostrength, dctwm as wm
from . import exifcheck, matrix, orientation, privacy_export, provenance

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
DATA_DIR = Path(__import__("os").environ.get("WM_DATA_DIR", "/data/wmbench"))
MAX_JOBS = int(__import__("os").environ.get("WM_MAX_JOBS", "200"))
MAX_UPLOAD_BYTES = 16 * 1024 * 1024
MAX_DIM = 4096
ALLOWED_KINDS = {"original", "watermarked", "target", "marked"}

app = FastAPI(title="抗裁切水印实验台", version="1.6.0")


# ---------------------------------------------------------------- 任务存储

def _ensure_data_dir() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def _prune_jobs() -> None:
    """超过上限时删除最旧的任务目录（归档报告目录不受任务清理影响）。"""
    jobs = sorted(
        (p for p in DATA_DIR.iterdir()
         if p.is_dir() and (p / "meta.json").is_file()),
        key=lambda p: p.stat().st_mtime,
    )
    for p in jobs[:-MAX_JOBS] if len(jobs) > MAX_JOBS else []:
        for f in p.rglob("*"):
            if f.is_file():
                f.unlink()
        for sub in sorted((q for q in p.rglob("*") if q.is_dir()),
                          reverse=True):
            sub.rmdir()
        p.rmdir()


def _job_dir(job_id: str) -> Path:
    d = DATA_DIR / job_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def _save_meta(job_id: str, meta: dict) -> None:
    (_job_dir(job_id) / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _load_meta(job_id: str) -> dict:
    p = DATA_DIR / job_id / "meta.json"
    if not p.exists():
        raise HTTPException(status_code=404, detail="任务不存在或已过期。")
    return json.loads(p.read_text(encoding="utf-8"))


def _resolve_image(file: UploadFile | None, job_id: str | None,
                   prefer: str = "watermarked") -> tuple[np.ndarray, str]:
    """从上传文件或已有任务中读取一张 RGB 图，返回 (RGB, 来源类型)。"""
    if file is not None and file.filename:
        data = file.file.read(MAX_UPLOAD_BYTES + 1)
        if len(data) > MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=413, detail="图片不能超过 16MB。")
        try:
            im = exifcheck.open_rgb_transposed(data)
        except exifcheck.PrivacyCheckError:
            raise HTTPException(status_code=400, detail="无法解析上传的图片。")
        return _check_dimensions(np.array(im)), "uploaded"

    if not job_id:
        raise HTTPException(status_code=400, detail="请提供图片文件或已有任务 ID。")
    meta = _load_meta(job_id)
    kind = prefer if (DATA_DIR / job_id / f"{prefer}.png").exists() else "original"
    if not (DATA_DIR / job_id / f"{kind}.png").exists():
        raise HTTPException(status_code=404, detail="任务图片不存在或已过期。")
    im = Image.open(DATA_DIR / job_id / f"{kind}.png").convert("RGB")
    return np.array(im), ("target" if kind == "target"
                          else "watermarked" if kind == "watermarked"
                          else "original")


def _check_dimensions(rgb: np.ndarray) -> np.ndarray:
    h, w = rgb.shape[:2]
    if max(h, w) > MAX_DIM:
        raise HTTPException(
            status_code=400,
            detail=f"图片尺寸 {w}x{h} 超过 {MAX_DIM}px 上限，请先缩小。",
        )
    if min(h, w) < wm.TILE * wm.BLK:
        raise HTTPException(
            status_code=400,
            detail=f"图片最短边至少需要 {wm.TILE * wm.BLK}px（当前 {min(h, w)}px）。",
        )
    return rgb


async def _read_embed_upload(image: UploadFile) -> np.ndarray:
    """读取并校验嵌入接口上传的原图，返回 RGB 数组。"""
    data = await image.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="图片不能超过 16MB。")
    if not data:
        raise HTTPException(status_code=400, detail="上传内容为空。")
    try:
        pil = exifcheck.open_rgb_transposed(data)
    except exifcheck.PrivacyCheckError:
        raise HTTPException(status_code=400, detail="无法解析上传的图片。")
    return _check_dimensions(np.array(pil))


def _embed_meta(
    job_id: str, rgb: np.ndarray, text: str, res: wm.EmbedResult,
    extra: dict | None = None,
) -> dict:
    meta = {
        "id": job_id,
        "created_at": time.time(),
        "text": text,
        "strength": res.strength,
        "width": int(rgb.shape[1]),
        "height": int(rgb.shape[0]),
        "psnr_db": res.psnr_db,
        "full_tiles": res.full_tiles,
        "payload_bytes": res.payload_bytes,
        "capacity_bytes": res.capacity_bytes,
        "votes_per_bit": res.votes_per_bit,
    }
    if extra:
        meta.update(extra)
    return meta


def _persist_embed_job(
    rgb: np.ndarray, text: str, res: wm.EmbedResult, extra_meta: dict | None = None,
) -> dict:
    """创建任务目录、保存原图/含水印 PNG 与 meta（手动/自动嵌入共用）。"""
    job_id = uuid.uuid4().hex[:12]
    d = _job_dir(job_id)
    Image.fromarray(rgb).save(d / "original.png", format="PNG", optimize=True)
    Image.fromarray(res.image).save(d / "watermarked.png", format="PNG", optimize=True)
    meta = _embed_meta(job_id, rgb, text, res, extra_meta)
    _save_meta(job_id, meta)
    _prune_jobs()
    return meta


# ---------------------------------------------------------------- 请求模型

class AttackRequest(BaseModel):
    job_id: str
    jpeg_quality: int | None = None        # 10~100；为空则不重压缩
    crop_fraction: float | None = None     # 0.2~1.0；为空则不裁切


class ReportRequest(BaseModel):
    detection_ids: list[str]               # 收录进报告的检测记录（至少一次）


class RobustnessMatrixRequest(BaseModel):
    jpeg_qualities: list[int]              # 1~3 个，20~100
    crop_fractions: list[float]            # 1~3 个，0.30~1.00


def _algo_params() -> dict:
    """嵌入/检测侧使用的算法参数快照（写入报告便于归档溯源）。"""
    return {
        "domain": "block-DCT Y-channel (DCT-II, 8x8)",
        "block_px": wm.BLK,
        "tile_blocks": wm.TILE,
        "tile_px": wm.TILE * wm.BLK,
        "sync_corner_blocks": wm.SYNC,
        "sync_pn_bits": len(wm.PN),
        "sync_score_threshold": wm.SYNC_SCORE_THRESHOLD,
        "weak_margin": wm.WEAK_MARGIN,
        "strength_delta": wm.STRENGTH_DELTA,
        "payload_capacity_bytes": wm.capacity_bytes(1),
        "modulated_coeff_pairs": [list(wm.PAIR_A), list(wm.PAIR_B)],
    }


# ---------------------------------------------------------------- 接口

@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.post("/api/embed")
async def embed(
    image: UploadFile = File(...),
    text: str = Form(...),
    strength: str = Form("medium"),
):
    """手动选强度嵌入（行为保持不变）。"""
    _ensure_data_dir()
    text = text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="水印文本不能为空。")
    rgb = await _read_embed_upload(image)

    try:
        res = wm.embed_watermark(rgb, text, strength=strength)
    except wm.WatermarkError as e:
        raise HTTPException(status_code=400, detail=str(e))

    meta = _persist_embed_job(rgb, text, res)
    return {"job_id": meta["id"], **meta}


@app.post("/api/embed-auto")
async def embed_auto(
    image: UploadFile = File(...),
    text: str = Form(...),
    jpeg_quality: int = Form(...),
    crop_fraction: float = Form(...),
    min_psnr_db: float = Form(...),
):
    """自动选强度嵌入。

    服务端按低、中、高顺序从**同一原图**分别嵌入；每档都执行
    「先中心裁切、后 JPEG 压缩」并调用现有盲检，仅当恢复文本与原文
    逐字一致、且嵌入图 PSNR 不低于用户下限时该档通过，取**首个**
    通过档。通过则沿用现有任务保存/下载流程；三档均未通过时
    **不创建任务**，仅返回逐档实测结果（HTTP 200 + passed=false）。
    """
    _ensure_data_dir()
    text = text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="水印文本不能为空。")
    rgb = await _read_embed_upload(image)

    try:
        chosen, tiers = autostrength.evaluate_tiers(
            rgb, text, jpeg_quality, crop_fraction, min_psnr_db)
    except autostrength.AutoStrengthError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except wm.WatermarkError as e:
        raise HTTPException(status_code=400, detail=str(e))

    tier_dicts = [tr.as_dict() for tr in tiers]
    request_params = {
        "jpeg_quality": int(jpeg_quality),
        "crop_fraction": round(float(crop_fraction), 3),
        "min_psnr_db": float(min_psnr_db),
    }

    if chosen is None:
        # 三档均未通过：不创建任务、不写盘，仅返回逐档实测结果。
        return JSONResponse({
            "passed": False,
            "mode": "auto",
            "text": text,
            "width": int(rgb.shape[1]),
            "height": int(rgb.shape[0]),
            "job_id": None,
            "selected_strength": None,
            "request": request_params,
            "tiers": tier_dicts,
        })

    chosen_tier = next(tr for tr in tiers if tr.strength == chosen)
    auto_meta = {
        "embed_mode": "auto",
        "auto_selection": {
            "selected_strength": chosen,
            "request": request_params,
            "tiers": tier_dicts,
        },
    }
    meta = _persist_embed_job(rgb, text, chosen_tier.embed, auto_meta)
    return JSONResponse({
        "passed": True,
        "mode": "auto",
        "request": request_params,
        "selected_strength": chosen,
        "selected_tier": next(d for d in tier_dicts if d["strength"] == chosen),
        "tiers": tier_dicts,
        "job_id": meta["id"],
        **meta,
    })


@app.post("/api/attack")
def attack(req: AttackRequest):
    _ensure_data_dir()
    meta = _load_meta(req.job_id)
    d = DATA_DIR / req.job_id
    source = d / "watermarked.png"
    if not source.exists():
        raise HTTPException(status_code=404, detail="该任务没有含水印图。")
    rgb = np.array(Image.open(source).convert("RGB"))

    history = meta.get("attack_history", [])
    if req.crop_fraction is not None:
        rgb = wm.center_crop(rgb, req.crop_fraction)
        history.append({"type": "center_crop", "fraction": round(float(req.crop_fraction), 3)})
    if req.jpeg_quality is not None:
        rgb = wm.jpeg_compress(rgb, int(req.jpeg_quality))
        history.append({"type": "jpeg", "quality": int(req.jpeg_quality)})

    Image.fromarray(rgb).save(d / "target.png", format="PNG", optimize=True)
    meta["attack_history"] = history
    meta["target_width"] = int(rgb.shape[1])
    meta["target_height"] = int(rgb.shape[0])
    _save_meta(req.job_id, meta)
    return {
        "job_id": req.job_id,
        "target_width": meta["target_width"],
        "target_height": meta["target_height"],
        "attack_history": history,
    }


def _truthy(v: str | None) -> bool:
    return (v or "").strip().lower() in ("1", "true", "yes", "on")


@app.post("/api/jobs/{job_id}/robustness-matrix")
def robustness_matrix(job_id: str, req: RobustnessMatrixRequest):
    """鲁棒性矩阵评测（只读、全内存）。

    从该任务的 watermarked.png 出发，对每个 (JPEG 质量, 裁切比例) 组合独立
    执行「先中心裁切、后 JPEG 压缩」并调用现有检测器；逐格返回恢复文本、
    与任务原文是否一致及同步/置信度指标，并给出只按实测成功格计算的二维
    最严成功边界。不写 target.png、不改 meta.json/攻击历史、不产生检测
    记录与归档文件。
    """
    _ensure_data_dir()
    meta = _load_meta(job_id)
    if meta.get("upload_only"):
        raise HTTPException(status_code=404, detail="该任务没有含水印图。")
    source = DATA_DIR / job_id / "watermarked.png"
    if not source.exists():
        raise HTTPException(status_code=404, detail="该任务没有含水印图。")

    watermarked = np.array(Image.open(source).convert("RGB"))
    try:
        result = matrix.run_robustness_matrix(
            watermarked, meta.get("text", ""),
            req.jpeg_qualities, req.crop_fractions)
    except matrix.MatrixError as e:
        raise HTTPException(status_code=400, detail=str(e))
    result["job_id"] = job_id
    result["source"] = "watermarked.png（评测前只读快照，不被改写）"
    return JSONResponse(result)


@app.post("/api/jobs/{job_id}/locate")
async def locate_propagation_copy(job_id: str, image: UploadFile = File(...)):
    """传播副本来源定位（只读、全内存）。

    以该任务的 watermarked.png 为参照，对上传的传播副本（裁切 / 缩放 /
    JPEG 转存等）做局部特征匹配 + RANSAC 单应估计，给出副本四角在参照图
    中的位置、覆盖率与多边形标注预览。仅当匹配数、内点数 / 比例、投影
    四边形面积与边界、重投影误差均通过门限时 located=True；否则
    located=False 且 reason 明确说明失败的判定项，不返回伪位置。

    不写任务文件、不改 meta.json 与攻击历史、不产生检测记录、不写归档。
    """
    _ensure_data_dir()
    meta = _load_meta(job_id)
    if meta.get("upload_only"):
        raise HTTPException(status_code=404, detail="该任务没有含水印图。")
    source = DATA_DIR / job_id / "watermarked.png"
    if not source.exists():
        raise HTTPException(status_code=404, detail="该任务没有含水印图。")

    data = await image.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="图片不能超过 16MB。")
    if not data:
        raise HTTPException(status_code=400, detail="上传内容为空。")
    try:
        query = exifcheck.open_rgb_transposed(data)
    except exifcheck.PrivacyCheckError:
        raise HTTPException(status_code=400, detail="无法解析上传的图片。")
    query_rgb = np.array(query)
    qh, qw = query_rgb.shape[:2]
    if max(qh, qw) > MAX_DIM:
        raise HTTPException(
            status_code=400,
            detail=f"图片尺寸 {qw}x{qh} 超过 {MAX_DIM}px 上限，请先缩小。")
    if min(qh, qw) < 64:
        raise HTTPException(
            status_code=400,
            detail=f"传播副本最短边至少需要 64px（当前 {min(qh, qw)}px）。")

    ref_rgb = np.array(Image.open(source).convert("RGB"))
    try:
        result = provenance.locate_propagation_copy(ref_rgb, query_rgb)
    except provenance.LocateError as e:
        raise HTTPException(status_code=400, detail=str(e))
    result["job_id"] = job_id
    result["filename"] = image.filename
    result["stored"] = False
    result["source"] = "watermarked.png（定位前只读快照，不被改写）"
    return JSONResponse(result)


@app.post("/api/detect")
async def detect(
    image: UploadFile | None = File(None),
    job_id: str | None = Form(None),
    auto_orient: str = Form("false"),
):
    _ensure_data_dir()
    rgb, source = _resolve_image(image, job_id, prefer="target")

    # 自动方向检索仅对“独立上传”的图片有意义；引用任务图片仍按原流程。
    use_orientation = source == "uploaded" and _truthy(auto_orient)
    orient_summary: dict | None = None
    chosen_spec: dict | None = None
    try:
        if use_orientation:
            search = orientation.search_orientations(rgb)
            orient_summary = search.summary()
            res = search.best_result
            if search.chosen is not None:
                chosen_spec = search.chosen.spec
        else:
            res = wm.detect_watermark(rgb)
    except wm.WatermarkError as e:
        raise HTTPException(status_code=400, detail=str(e))

    detection_id = "det_" + uuid.uuid4().hex[:12]
    detected_at, detected_at_iso = archive.utc_now()

    # 独立上传检测：建立轻量任务元数据，使检测记录同样可归档
    upload_only = False
    attack_history: list = []
    if job_id and (DATA_DIR / job_id / "meta.json").exists():
        meta = _load_meta(job_id)
        upload_only = bool(meta.get("upload_only"))
        attack_history = list(meta.get("attack_history", []))
    else:
        job_id = uuid.uuid4().hex[:12]
        upload_only = True
        _save_meta(job_id, {
            "id": job_id,
            "created_at": detected_at,
            "upload_only": True,
            "width": int(rgb.shape[1]),
            "height": int(rgb.shape[0]),
        })

    d = DATA_DIR / job_id
    marked_url = None
    has_overlay = res.overlay is not None
    overlay = res.overlay
    if has_overlay:
        # 自动方向检索：入选方向的标注图在该方向坐标系内生成，
        # 逆变换回上传图坐标后再保存（检测输入快照始终是原始上传图）。
        if use_orientation and chosen_spec is not None:
            overlay = orientation.invert_to_original(overlay, chosen_spec)
        if overlay.shape[:2] != rgb.shape[:2]:
            # 几何兜底：逆变换结果必须与上传图同尺寸，否则丢弃标注图
            has_overlay = False
        else:
            # 兼容旧前端的“当前标注图”，同时为每次检测保存独立标注图
            Image.fromarray(overlay).save(d / "marked.png", format="PNG", optimize=True)
            marked_url = f"/api/jobs/{job_id}/marked"

    det_dir = d / "detections"
    det_dir.mkdir(parents=True, exist_ok=True)
    # 该次检测实际分析的图片快照（避免后续攻击覆盖 target.png）；
    # 自动方向检索时保存的始终是未经几何变换的原始上传图。
    Image.fromarray(rgb).save(det_dir / f"{detection_id}.source.png",
                              format="PNG", optimize=True)
    if has_overlay:
        Image.fromarray(overlay).save(
            det_dir / f"{detection_id}.png", format="PNG", optimize=True)

    metrics = {
        "confidence": res.confidence,
        "sync_score": res.sync_score,
        "sync_agreement": res.sync_agreement,
        "payload_bytes": res.payload_bytes,
        "payload_blocks": res.payload_blocks,
        "sync_blocks": res.sync_blocks,
        "tiles_y": res.tiles_y,
        "tiles_x": res.tiles_x,
        "phase": list(res.phase),
        "votes_per_bit": res.votes_per_bit,
        "marked_blocks": res.marked_blocks,
        "total_blocks": res.total_blocks,
        "marked_fraction": res.marked_fraction,
        "bit_confidences": res.bit_confidences,
    }
    record = {
        "detection_id": detection_id,
        "job_id": job_id,
        "source": source,
        "detected_at": detected_at,
        "detected_at_iso": detected_at_iso,
        "detected": res.detected,
        "text": res.text,
        "reason": res.reason,
        "metrics": metrics,
        "attack_history": attack_history,
        "has_overlay": has_overlay,
    }
    if orient_summary is not None:
        record["orientation_search"] = orient_summary
    (det_dir / f"{detection_id}.json").write_text(
        json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")

    return JSONResponse({
        "job_id": job_id,
        "detection_id": detection_id,
        "source": source,
        "detected_at": detected_at,
        "detected_at_iso": detected_at_iso,
        "upload_only": upload_only,
        "detected": res.detected,
        "text": res.text,
        "reason": res.reason,
        "metrics": metrics,
        "orientation_search": orient_summary,
        "marked_url": marked_url,
        "marked_det_url": (
            f"/api/jobs/{job_id}/detections/{detection_id}/marked"
            if has_overlay else None
        ),
    })


# ---------------------------------------------------------------- 图片隐私检查

@app.post("/api/privacy-check")
async def privacy_check(image: UploadFile = File(...)):
    """只读解析 PNG/JPEG 的 EXIF 拍摄时间 / 相机品牌型号 / GPS。

    不落盘、不建任务、不触碰水印任务与报告归档；结果仅随本次响应返回。
    EXIF 损坏 / 无法解码不视为错误（200 + warnings），只有图片本身
    无法解码或格式非 PNG/JPEG 才返回 400。
    """
    data = await image.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="图片不能超过 16MB。")
    if not data:
        raise HTTPException(status_code=400, detail="上传内容为空。")
    try:
        result = exifcheck.analyze_privacy(data)
    except exifcheck.PrivacyCheckError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return JSONResponse({
        "filename": image.filename,
        "content_type": image.content_type,
        "bytes": len(data),
        **result,
        "stored": False,
    })


@app.post("/api/privacy-export")
async def export_privacy(image: UploadFile = File(...)):
    """导出隐私副本（全内存）：EXIF Orientation 摆正 + 全新 PNG 重编码。

    先按 EXIF Orientation 把像素真正旋转 / 镜像摆正，保留原有透明
    通道，再以全新 PNG 重编码，不带入 EXIF / GPS / XMP / ICC 及
    PNG 文本元数据；导出前重新解析生成文件复查，任一项不通过都
    返回错误且响应中不含可下载副本。不创建 / 改写任何水印任务、
    攻击记录与报告归档，上传字节与副本均不落盘。
    """
    data = await image.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="图片不能超过 16MB。")
    if not data:
        raise HTTPException(status_code=400, detail="上传内容为空。")
    try:
        result = privacy_export.export_privacy_copy(data)
    except privacy_export.PrivacyCheckError as e:
        # 图片本身无法解码或格式非 PNG/JPEG
        raise HTTPException(status_code=400, detail=str(e))
    except privacy_export.PrivacyExportError as e:
        # 摆正后尺寸超 4096px 等前置条件不满足
        raise HTTPException(status_code=400, detail=str(e))
    except privacy_export.PrivacyCopyRecheckError as e:
        # 复查仍发现被禁元数据 / 重编码结果异常：不提供下载
        return JSONResponse(
            status_code=500,
            content={
                "detail": "隐私副本导出复查未通过，已拒绝提供下载。",
                "offenders": e.offenders,
                **e.context,
                "recheck": e.recheck,
                "stored": False,
            },
        )
    return JSONResponse({"filename": image.filename, **result})


@app.get("/api/jobs/{job_id}/{kind}")
def get_job_file(job_id: str, kind: str):
    if kind not in ALLOWED_KINDS:
        raise HTTPException(status_code=404, detail="未知文件类型。")
    p = DATA_DIR / job_id / f"{kind}.png"
    if not p.exists():
        raise HTTPException(status_code=404, detail="文件不存在。")
    download = kind in ("watermarked", "marked", "original", "target")
    return FileResponse(
        p, media_type="image/png",
        filename=f"{kind}.png" if download else None,
    )


@app.get("/api/jobs/{job_id}/detections/{det_id}/{kind}")
def get_detection_file(job_id: str, det_id: str, kind: str):
    """某次检测的独立资产：marked=失真标注图，source=检测输入快照。"""
    if kind not in ("marked", "source"):
        raise HTTPException(status_code=404, detail="未知文件类型。")
    if not archive.DETECTION_ID_RE.match(det_id):
        raise HTTPException(status_code=404, detail="检测记录不存在。")
    suffix = ".png" if kind == "marked" else ".source.png"
    p = DATA_DIR / job_id / "detections" / f"{det_id}{suffix}"
    if not p.exists():
        raise HTTPException(status_code=404, detail="文件不存在。")
    return FileResponse(p, media_type="image/png",
                        filename=f"{det_id}-{kind}.png")


# ---------------------------------------------------------------- 报告归档

@app.post("/api/jobs/{job_id}/reports")
def create_job_report(job_id: str, req: ReportRequest):
    _ensure_data_dir()
    report = archive.create_report(
        DATA_DIR, job_id, req.detection_ids, _algo_params())
    return JSONResponse({
        "report_id": report["report_id"],
        "job_id": job_id,
        "generated_at": report["generated_at"],
        "generated_at_iso": report["generated_at_iso"],
        "detection_count": len(report["detections"]),
        "entries": len(report["manifest"]["entries"]),
        "archive": report["archive"],
        "download_url": report["download_url"],
        "verify_url": f"/api/reports/{report['report_id']}/verify",
    })


@app.get("/api/reports/{report_id}")
def get_report(report_id: str, download: bool = False):
    _ensure_data_dir()
    p = archive.report_path(DATA_DIR, report_id)
    return FileResponse(
        p, media_type="application/json; charset=utf-8",
        filename=f"{report_id}.json" if download else None,
    )


@app.post("/api/reports/{report_id}/verify")
def verify_report(report_id: str):
    _ensure_data_dir()
    result = archive.verify_report(DATA_DIR, report_id)
    return JSONResponse(result, status_code=200 if result["valid"] else 409)


# ---------------------------------------------------------------- 静态前端

@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
