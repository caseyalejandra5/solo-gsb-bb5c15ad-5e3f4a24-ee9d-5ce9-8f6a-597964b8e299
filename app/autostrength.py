"""自动选强度：低 → 中 → 高逐档试嵌入，首个「扛住攻击 + 达标 PSNR」档胜出。

流程（三档均从**同一张原图**独立出发）：

1. 以给定强度调用现有 :func:`dctwm.embed_watermark` 嵌入，记录嵌入图 PSNR；
2. 对嵌入图执行与攻击面板完全相同的「**先中心裁切、后 JPEG 压缩**」；
3. 对受攻击图调用现有盲检 :func:`dctwm.detect_watermark`；
4. 仅当**恢复文本与原文逐字一致**且**嵌入图 PSNR ≥ 用户下限**时，该档通过。

按 low / medium / high 顺序评测，取**首个通过档**。三档均不通过时不创建
任务，由调用方把逐档实测结果原样返回给页面。本模块全程在内存中计算，
不落盘、不触碰任务目录与归档。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import dctwm as wm

STRENGTH_ORDER = ("low", "medium", "high")

MIN_QUALITY = 20
MAX_QUALITY = 100
MIN_FRACTION = 0.30
MAX_FRACTION = 1.00
MIN_PSNR = 20.0
MAX_PSNR = 60.0

STRENGTH_LABELS = {"low": "低", "medium": "中", "high": "高"}


class AutoStrengthError(ValueError):
    """自动选强度的参数错误（映射为 HTTP 400）。"""


@dataclass
class TierResult:
    strength: str
    embed: wm.EmbedResult
    detect: wm.DetectResult
    attacked: np.ndarray          # 该档「先裁切后压缩」的受攻击图
    width: int
    height: int
    snapped_fraction_w: float
    snapped_fraction_h: float
    text_match: bool
    psnr_ok: bool

    def as_dict(self) -> dict:
        res, det = self.embed, self.detect
        return {
            "strength": self.strength,
            "strength_label": STRENGTH_LABELS[self.strength],
            "psnr_db": res.psnr_db,
            "psnr_ok": self.psnr_ok,
            "full_tiles": res.full_tiles,
            "votes_per_bit": res.votes_per_bit,
            "payload_bytes": res.payload_bytes,
            "capacity_bytes": res.capacity_bytes,
            "detected": bool(det.detected),
            "text": det.text or "",
            "text_match": bool(self.text_match),
            "reason": det.reason or "",
            "width": self.width,
            "height": self.height,
            "snapped_fraction_w": self.snapped_fraction_w,
            "snapped_fraction_h": self.snapped_fraction_h,
            "metrics": {
                "confidence": det.confidence,
                "sync_score": det.sync_score,
                "sync_agreement": det.sync_agreement,
                "sync_blocks": det.sync_blocks,
                "payload_blocks": det.payload_blocks,
                "tiles_y": det.tiles_y,
                "tiles_x": det.tiles_x,
                "votes_per_bit": det.votes_per_bit,
                "marked_blocks": det.marked_blocks,
                "total_blocks": det.total_blocks,
                "marked_fraction": det.marked_fraction,
            },
        }


def validate_params(
    jpeg_quality: int, crop_fraction: float, min_psnr_db: float,
) -> tuple[int, float, float]:
    """校验自动选强度参数，返回 (质量 int, 比例 float, PSNR float)。"""
    if isinstance(jpeg_quality, bool) or not isinstance(jpeg_quality, int):
        raise AutoStrengthError("JPEG 质量必须为整数。")
    if not (MIN_QUALITY <= jpeg_quality <= MAX_QUALITY):
        raise AutoStrengthError(
            f"JPEG 质量需在 {MIN_QUALITY}~{MAX_QUALITY} 之间（收到 {jpeg_quality}）。")

    if isinstance(crop_fraction, bool) or not isinstance(crop_fraction, (int, float)):
        raise AutoStrengthError("中心裁切保留比例必须为数字。")
    crop_fraction = float(crop_fraction)
    if not (MIN_FRACTION - 1e-9 <= crop_fraction <= MAX_FRACTION + 1e-9):
        raise AutoStrengthError(
            f"中心裁切保留比例需在 {int(MIN_FRACTION * 100)}%~"
            f"{int(MAX_FRACTION * 100)}% 之间（收到 {crop_fraction:g}）。")
    crop_fraction = min(max(crop_fraction, MIN_FRACTION), MAX_FRACTION)

    if isinstance(min_psnr_db, bool) or not isinstance(min_psnr_db, (int, float)):
        raise AutoStrengthError("最低 PSNR 必须为数字。")
    min_psnr_db = float(min_psnr_db)
    if not (MIN_PSNR - 1e-9 <= min_psnr_db <= MAX_PSNR + 1e-9):
        raise AutoStrengthError(
            f"最低 PSNR 需在 {int(MIN_PSNR)}~{int(MAX_PSNR)} dB 之间"
            f"（收到 {min_psnr_db:g}）。")
    return jpeg_quality, round(crop_fraction, 3), min_psnr_db


def _evaluate_one(
    rgb: np.ndarray, text: str, strength: str,
    jpeg_quality: int, crop_fraction: float, min_psnr_db: float,
) -> TierResult:
    """单个强度档：嵌入 → 先裁切后压缩 → 盲检 → 判定。"""
    embed_res = wm.embed_watermark(rgb, text, strength=strength)

    base_h, base_w = rgb.shape[:2]
    attacked = wm.center_crop(embed_res.image, crop_fraction)
    attacked = wm.jpeg_compress(attacked, jpeg_quality)
    h, w = attacked.shape[:2]

    det = wm.detect_watermark(attacked)
    text_match = bool(det.detected and det.text == text)
    # PSNR 口径与 /api/embed（亮度 Y 通道、EmbedResult.psnr_db）保持一致
    psnr_ok = bool(embed_res.psnr_db + 1e-9 >= min_psnr_db)

    return TierResult(
        strength=strength,
        embed=embed_res,
        detect=det,
        attacked=attacked,
        width=w,
        height=h,
        snapped_fraction_w=round(w / base_w, 4),
        snapped_fraction_h=round(h / base_h, 4),
        text_match=text_match,
        psnr_ok=psnr_ok,
    )


def evaluate_tiers(
    rgb: np.ndarray, text: str,
    jpeg_quality: int, crop_fraction: float, min_psnr_db: float,
) -> tuple[str | None, list[TierResult]]:
    """按低 → 中 → 高顺序逐档评测，返回 (首个通过档 or None, 三档结果)。

    三档始终全部评测，便于页面在失败时展示逐档实测数据。
    """
    jpeg_quality, crop_fraction, min_psnr_db = validate_params(
        jpeg_quality, crop_fraction, min_psnr_db)
    tiers: list[TierResult] = []
    chosen: str | None = None
    for strength in STRENGTH_ORDER:
        tr = _evaluate_one(
            rgb, text, strength, jpeg_quality, crop_fraction, min_psnr_db)
        tiers.append(tr)
        if chosen is None and tr.text_match and tr.psnr_ok:
            chosen = strength
    return chosen, tiers
