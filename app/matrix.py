"""鲁棒性矩阵评测（JPEG 质量 × 中心裁切保留比例）。

设计概述
========
* 用户为一个**已嵌入水印的任务**选择至多 3 个 JPEG 质量与至多 3 个中心裁切
  保留比例；两个轴做笛卡尔积，每个组合（格）都从**同一份含水印原图**独立地
  执行「先中心裁切、后 JPEG 压缩」，再调用现有检测器 ``dctwm.detect_watermark``。
* 每格返回：恢复文本、是否与任务原文一致、置信度与同步指标（相关峰 / 一致率等）。
* 「二维最严成功边界」**只按实际成功（文本与原文一致）的实测格**计算：
  若存在另一成功格，其 JPEG 质量不高于本格、保留比例不大于本格，且至少一项
  更严（数值更小），则本格被支配，不在边界上；否则本格属于最严成功边界。
  该定义只依赖实测格之间的偏序关系，**不以相邻 / 中间未测格推断结果**。
* 评测全程在内存中进行：不写 target.png、不改 meta.json 与攻击历史、不生成
  检测记录、不写归档——任务图片、攻击历史与检测归档均不被评测改写。
"""

from __future__ import annotations

import numpy as np

from . import dctwm as wm

MAX_PER_AXIS = 3
MIN_QUALITY = 20
MAX_QUALITY = 100
MIN_FRACTION = 0.30
MAX_FRACTION = 1.00


class MatrixError(ValueError):
    """矩阵评测参数错误（映射为 HTTP 400）。"""


def validate_axes(jpeg_qualities, crop_fractions) -> tuple[list[int], list[float]]:
    """校验并归一化两个轴：非空、各 1~3 个、去重、取值合法；返回升序列表。"""
    if not isinstance(jpeg_qualities, (list, tuple)) or not jpeg_qualities:
        raise MatrixError("请至少选择 1 个 JPEG 质量（至多 3 个）。")
    if not isinstance(crop_fractions, (list, tuple)) or not crop_fractions:
        raise MatrixError("请至少选择 1 个中心裁切保留比例（至多 3 个）。")
    if len(jpeg_qualities) > MAX_PER_AXIS:
        raise MatrixError(f"JPEG 质量至多选择 {MAX_PER_AXIS} 个。")
    if len(crop_fractions) > MAX_PER_AXIS:
        raise MatrixError(f"中心裁切保留比例至多选择 {MAX_PER_AXIS} 个。")

    qualities: list[int] = []
    for q in jpeg_qualities:
        if isinstance(q, bool) or not isinstance(q, int):
            raise MatrixError("JPEG 质量必须为整数。")
        if not (MIN_QUALITY <= q <= MAX_QUALITY):
            raise MatrixError(
                f"JPEG 质量需在 {MIN_QUALITY}~{MAX_QUALITY} 之间（收到 {q}）。")
        qualities.append(q)

    fractions: list[float] = []
    for f in crop_fractions:
        if isinstance(f, bool) or not isinstance(f, (int, float)):
            raise MatrixError("裁切保留比例必须为数字。")
        f = float(f)
        if not (MIN_FRACTION - 1e-9 <= f <= MAX_FRACTION + 1e-9):
            raise MatrixError(
                f"裁切保留比例需在 {int(MIN_FRACTION * 100)}%~"
                f"{int(MAX_FRACTION * 100)}% 之间（收到 {f:g}）。")
        fractions.append(round(min(max(f, MIN_FRACTION), MAX_FRACTION), 3))

    if len(set(qualities)) != len(qualities):
        raise MatrixError("JPEG 质量不能重复。")
    if len(set(fractions)) != len(fractions):
        raise MatrixError("裁切保留比例不能重复。")
    return sorted(qualities), sorted(fractions)


def success_boundary(cells: list[dict]) -> set[tuple[int, float]]:
    """计算二维最严成功边界。

    入参为格字典列表（含 ``text_match`` / ``jpeg_quality`` / ``crop_fraction``）。
    返回边界上成功格的 (质量, 保留比例) 集合。失败 / 文本不一致格不参与。

    判定严格按偏序：成功格 B 支配成功格 A ⇔
        B.q ≤ A.q 且 B.f ≤ A.f，且至少一项严格更小；
    与格是否相邻、中间是否还有未测组合完全无关。
    """
    succ = [c for c in cells if c.get("text_match")]
    boundary: set[tuple[int, float]] = set()
    for a in succ:
        qa, fa = a["jpeg_quality"], a["crop_fraction"]
        dominated = False
        for b in succ:
            if b is a:
                continue
            qb, fb = b["jpeg_quality"], b["crop_fraction"]
            if qb <= qa and fb <= fa and (qb < qa or fb < fa):
                dominated = True
                break
        if not dominated:
            boundary.add((qa, fa))
    return boundary


def _cell_metrics(res: wm.DetectResult) -> dict:
    return {
        "confidence": res.confidence,
        "sync_score": res.sync_score,
        "sync_agreement": res.sync_agreement,
        "sync_blocks": res.sync_blocks,
        "payload_blocks": res.payload_blocks,
        "tiles_y": res.tiles_y,
        "tiles_x": res.tiles_x,
        "votes_per_bit": res.votes_per_bit,
        "marked_blocks": res.marked_blocks,
        "total_blocks": res.total_blocks,
        "marked_fraction": res.marked_fraction,
    }


def run_robustness_matrix(
    watermarked_rgb: np.ndarray,
    truth_text: str,
    jpeg_qualities: list[int],
    crop_fractions: list[float],
) -> dict:
    """对每个 (质量, 裁切比例) 组合独立执行「先裁切后压缩 + 现有检测器」。

    全程不写盘；单格异常不影响其它格（该格 ok=false 并记录错误信息）。
    """
    qualities, fractions = validate_axes(jpeg_qualities, crop_fractions)
    base_h, base_w = watermarked_rgb.shape[:2]

    cells: list[dict] = []
    for fraction in fractions:               # 行：裁切保留比例（升序）
        for quality in qualities:            # 列：JPEG 质量（升序）
            cell: dict = {
                "jpeg_quality": quality,
                "crop_fraction": fraction,
                "ok": False,
                "detected": False,
                "text": "",
                "text_match": False,
                "reason": "",
                "width": None,
                "height": None,
                "snapped_fraction_w": None,
                "snapped_fraction_h": None,
                "metrics": None,
            }
            try:
                # 每格都从同一份含水印原图重新出发：先裁切，后压缩。
                attacked = wm.center_crop(watermarked_rgb, fraction)
                attacked = wm.jpeg_compress(attacked, quality)
                h, w = attacked.shape[:2]
                cell["width"] = w
                cell["height"] = h
                cell["snapped_fraction_w"] = round(w / base_w, 4)
                cell["snapped_fraction_h"] = round(h / base_h, 4)

                res = wm.detect_watermark(attacked)
                cell["ok"] = True
                cell["detected"] = bool(res.detected)
                cell["text"] = res.text or ""
                cell["text_match"] = bool(
                    res.detected and res.text == truth_text)
                cell["reason"] = res.reason or ""
                cell["metrics"] = _cell_metrics(res)
            except Exception as exc:         # 单格失败隔离，不拖垮整矩阵
                cell["reason"] = f"评测执行失败：{exc}"
            cells.append(cell)

    boundary = success_boundary(cells)
    for c in cells:
        c["on_boundary"] = (c["jpeg_quality"], c["crop_fraction"]) in boundary

    successful = sum(1 for c in cells if c["text_match"])
    return {
        "text": truth_text,
        "jpeg_qualities": qualities,
        "crop_fractions": fractions,
        "cells": cells,
        "boundary": [
            {"jpeg_quality": q, "crop_fraction": f}
            for (q, f) in sorted(
                boundary, key=lambda x: (x[1], x[0]))
        ],
        "counts": {
            "cells": len(cells),
            "successful": successful,
            "failed": len(cells) - successful,
            "boundary": len(boundary),
        },
        "inference": "boundary 仅由实测成功格按二维偏序计算，未测组合不推断",
    }
