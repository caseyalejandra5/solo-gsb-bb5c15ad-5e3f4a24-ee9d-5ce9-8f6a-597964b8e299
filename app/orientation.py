"""独立上传检测的可选自动方向检索。

对上传图按固定顺序枚举 8 个视图——原向、顺时针 90° / 180° / 270°，以及
水平镜像后的同样四个方向——逐一调用现有 :func:`dctwm.detect_watermark`。
本模块只做外层几何枚举，不修改检测算法本身：

* 仅把**通过载荷校验**（``DetectResult.detected``，即魔数 + 校验和同时通过）
  的方向记为有效候选；
* 有效候选按置信度取最佳，置信度相同则按固定枚举顺序取靠前者；
* 多个有效候选恢复出**不同文本**时标记冲突并列出候选文本，调用方不得把
  最佳候选当成唯一结论；
* 入选方向的失真标注图在该方向的坐标系内生成，经逆变换映射回上传图坐标；
  检测输入快照始终是未经几何变换的原始上传图。
"""

from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np

from .dctwm import DetectResult, detect_watermark

# 固定枚举顺序（同置信度时的取舍顺序即此列表的下标升序）。
# k 为 numpy.rot90 的旋转次数（1 = 逆时针 90°，3 = 顺时针 90°）。
ORIENTATIONS: list[dict] = [
    {"id": "original",         "label": "原向（0°）",                 "mirror": False, "k": 0, "rotation_cw": 0},
    {"id": "rot90cw",         "label": "顺时针旋转 90°",               "mirror": False, "k": 3, "rotation_cw": 90},
    {"id": "rot180",          "label": "旋转 180°",                   "mirror": False, "k": 2, "rotation_cw": 180},
    {"id": "rot270cw",        "label": "逆时针旋转 90°（顺时针 270°）", "mirror": False, "k": 1, "rotation_cw": 270},
    {"id": "mirror",          "label": "水平镜像",                    "mirror": True,  "k": 0, "rotation_cw": 0},
    {"id": "mirror_rot90cw",  "label": "水平镜像 + 顺时针 90°",        "mirror": True,  "k": 3, "rotation_cw": 90},
    {"id": "mirror_rot180",   "label": "水平镜像 + 旋转 180°",        "mirror": True,  "k": 2, "rotation_cw": 180},
    {"id": "mirror_rot270cw", "label": "水平镜像 + 逆时针 90°",       "mirror": True,  "k": 1, "rotation_cw": 270},
]


# ---------------------------------------------------------------- 几何变换

def apply_orientation(rgb: np.ndarray, spec: dict) -> np.ndarray:
    """把上传图变换到某个方向视图：先水平镜像（若需要），再旋转。"""
    view = np.flip(rgb, axis=1) if spec["mirror"] else rgb
    view = np.rot90(view, k=spec["k"])
    return np.ascontiguousarray(view)


def invert_to_original(arr: np.ndarray, spec: dict) -> np.ndarray:
    """方向视图坐标 -> 上传图坐标，:func:`apply_orientation` 的逆变换。"""
    out = np.rot90(arr, k=-spec["k"])
    if spec["mirror"]:
        out = np.flip(out, axis=1)
    return np.ascontiguousarray(out)


# ---------------------------------------------------------------- 候选聚合

@dataclass
class Candidate:
    index: int
    spec: dict
    result: DetectResult

    @property
    def valid(self) -> bool:
        """通过载荷校验（魔数 + 长度 + 校验和）才算有效。"""
        return bool(self.result.detected)

    def summary(self) -> dict:
        return {
            "index": self.index,
            "orientation": self.spec["id"],
            "label": self.spec["label"],
            "mirror": bool(self.spec["mirror"]),
            "rotation_cw": self.spec["rotation_cw"],
            "detected": self.valid,
            "text": self.result.text if self.valid else "",
            "confidence": self.result.confidence if self.valid else 0.0,
            "sync_score": self.result.sync_score,
            "reason": "" if self.valid else self.result.reason,
        }


def pick_best_index(candidate_summaries: list[dict]) -> int | None:
    """有效候选中按置信度降序、固定顺序下标升序选出最佳；无候选返回 None。"""
    valid = [c for c in candidate_summaries if c["detected"]]
    if not valid:
        return None
    return min(valid, key=lambda c: (-float(c["confidence"]), c["index"]))["index"]


def distinct_texts(candidate_summaries: list[dict]) -> list[str]:
    """有效候选恢复出的去重文本，按首次出现顺序排列。"""
    seen: list[str] = []
    for c in candidate_summaries:
        if c["detected"] and c["text"] not in seen:
            seen.append(c["text"])
    return seen


@dataclass
class OrientationSearch:
    candidates: list[Candidate]
    chosen_index: int | None
    conflict: bool
    conflict_texts: list[str]
    fallback_index: int

    @property
    def chosen(self) -> Candidate | None:
        return self.candidates[self.chosen_index] if self.chosen_index is not None else None

    @property
    def fallback(self) -> Candidate:
        return self.candidates[self.fallback_index]

    @property
    def best_result(self) -> DetectResult:
        """入选候选的检测结果；全部无效时退回同步峰最高的方向（仅供展示原因）。"""
        return (self.chosen.result if self.chosen is not None
                else self.fallback.result)

    def summary(self) -> dict:
        cands = [c.summary() for c in self.candidates]
        valid_indices = [c["index"] for c in cands if c["detected"]]
        chosen = self.chosen
        return {
            "enabled": True,
            "orientations_tried": len(self.candidates),
            "chosen_index": self.chosen_index,
            "chosen_orientation": chosen.spec["id"] if chosen else None,
            "chosen_label": chosen.spec["label"] if chosen else None,
            "valid_candidate_count": len(valid_indices),
            "valid_indices": valid_indices,
            "conflict": self.conflict,
            "conflict_texts": self.conflict_texts,
            "fallback_index": self.fallback_index,
            "fallback_label": self.fallback.spec["label"],
            "candidates": cands,
        }


def search_orientations(rgb: np.ndarray) -> OrientationSearch:
    """按固定顺序对 8 个方向各运行一次现有水印检测并聚合结果。"""
    candidates: list[Candidate] = []
    for i, spec in enumerate(ORIENTATIONS):
        view = apply_orientation(rgb, spec)
        candidates.append(Candidate(i, spec, detect_watermark(view)))

    summaries = [c.summary() for c in candidates]
    chosen_index = pick_best_index(summaries)
    texts = distinct_texts(summaries)

    # 无有效候选时，退回同步相关峰绝对值最高的方向（给出最接近的失败原因）
    fallback_index = max(
        range(len(candidates)),
        key=lambda i: (abs(float(candidates[i].result.sync_score)), -i),
    )

    return OrientationSearch(
        candidates=candidates,
        chosen_index=chosen_index,
        conflict=len(texts) > 1,
        conflict_texts=texts,
        fallback_index=fallback_index,
    )
