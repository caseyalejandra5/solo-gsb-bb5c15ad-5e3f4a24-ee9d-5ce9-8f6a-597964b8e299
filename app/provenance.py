"""传播副本来源定位（只读）。

给定一个已嵌入水印任务的参照图 ``watermarked.png`` 与一张**传播副本**
（可能由参照图裁切、缩放并经 JPEG 转存得到），在两侧提取局部特征
（SIFT），用 KNN 匹配 + Lowe 比率筛选后，以 **RANSAC 单应矩阵**估计
副本四角在参照图中的位置。

只有当以下条件**全部**成立时才返回四角坐标 / 覆盖率 / 标注预览：

1. 比率筛选后的良好匹配数量 ≥ ``MIN_GOOD_MATCHES``；
2. RANSAC 内点数量 ≥ ``MIN_INLIERS`` 且内点比例 ≥ ``MIN_INLIER_RATIO``；
3. 单应可解且四角投影均为有限值；
4. 投影四边形**凸且不退化**（最小内角、最短边、轴间尺度比可信）；
5. 投影四边形与参照图边界可信：四角基本落在参照图范围内、
   面积占比落在合理区间（``MIN_COVERAGE``~``MAX_COVERAGE``）；
6. 内点重投影 RMS 误差 ≤ ``MAX_RMS_PX``（参照图像素口径）。

任何一条不满足都返回 ``located=False`` 与具体原因，**绝不给出伪位置**。
全部计算在内存中完成：不写任务目录、不写检测归档、不追加归档日志。
"""

from __future__ import annotations

import base64
import math

import cv2
import numpy as np

# ---------------------------------------------------------------- 门限常量

WORK_MAX_DIM = 1600          # 匹配工作图最长边（超出则等比缩小）
PREVIEW_MAX_DIM = 1280       # 标注预览最长边
SIFT_NFEATURES = 8000        # 单侧最多保留的特征点数
MATCH_RATIO = 0.75           # Lowe 比率阈值
MIN_GOOD_MATCHES = 12        # 比率筛选后最少良好匹配数
MIN_INLIERS = 10             # RANSAC 最少内点数
MIN_INLIER_RATIO = 0.35      # 内点占良好匹配的最低比例
RANSAC_REPROJ_THRESH = 3.0   # RANSAC 重投影门限（工作图像素；1600px 口径）
RANSAC_CONFIDENCE = 0.995
MAX_RMS_PX = 6.0             # 内点重投影 RMS 上限（参照图像素）
MIN_QUAD_ANGLE_DEG = 15.0    # 投影四边形最小内角
MAX_AXIS_SCALE_RATIO = 6.0   # 两轴尺度比上限（再大视为几何退化）
MIN_SIDE_FRACTION = 0.01     # 投影四边形最短边 / 参照图最短边的下限
MIN_COVERAGE = 0.005         # 投影四边形面积 / 参照图面积的下限
MAX_COVERAGE = 1.02          # 允许略微越界（插值 / 亚像素），超过即不可信
BOUND_MARGIN_FRAC = 0.02     # 允许四角越界的相对余量（参照图短边）

# 判定项顺序（同时用于失败原因与 checks 回显）
CHECK_GOOD_MATCHES = "good_matches"
CHECK_HOMOGRAPHY = "homography"
CHECK_INLIER_COUNT = "inlier_count"
CHECK_INLIER_RATIO = "inlier_ratio"
CHECK_QUAD_FINITE = "quad_finite"
CHECK_QUAD_SHAPE = "quad_shape"
CHECK_QUAD_BOUNDS = "quad_bounds"
CHECK_COVERAGE = "coverage"
CHECK_RMS = "reprojection_rms"


class LocateError(ValueError):
    """入参本身非法（映射为 HTTP 400），与“无法定位”语义不同。"""


def thresholds() -> dict:
    """返回门限快照（写入响应，便于结果可复核）。"""
    return {
        "feature": "SIFT",
        "matcher": "BFMatcher(L2) + Lowe ratio",
        "match_ratio": MATCH_RATIO,
        "ransac_reproj_thresh": RANSAC_REPROJ_THRESH,
        "ransac_confidence": RANSAC_CONFIDENCE,
        "min_good_matches": MIN_GOOD_MATCHES,
        "min_inliers": MIN_INLIERS,
        "min_inlier_ratio": MIN_INLIER_RATIO,
        "max_rms_px": MAX_RMS_PX,
        "min_quad_angle_deg": MIN_QUAD_ANGLE_DEG,
        "max_axis_scale_ratio": MAX_AXIS_SCALE_RATIO,
        "min_coverage": MIN_COVERAGE,
        "max_coverage": MAX_COVERAGE,
    }


# ---------------------------------------------------------------- 几何工具

def _scale_for(width: int, height: int, max_dim: int) -> float:
    longest = max(int(width), int(height))
    if longest <= max_dim:
        return 1.0
    return max_dim / longest


def _to_gray(rgb: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)


def _detect_and_compute(gray: np.ndarray):
    # 适度降低对比度门限、放宽边缘响应门限，使小尺寸 / 经 JPEG 重压的
    # 传播副本仍能提取足够特征；误配由 Lowe 比率与 RANSAC 双重筛除。
    sift = cv2.SIFT_create(
        nfeatures=SIFT_NFEATURES,
        contrastThreshold=0.02,
        edgeThreshold=20,
        sigma=1.6,
    )
    return sift.detectAndCompute(gray, None)


def _good_matches(desc_ref, desc_query):
    """Lowe 比率筛选；描述子不足或类型不符时返回空列表。"""
    if desc_ref is None or desc_query is None:
        return []
    if len(desc_ref) < 2 or len(desc_query) < 2:
        return []
    # KNN 要求训练侧至少 2 个描述子
    bf = cv2.BFMatcher(cv2.NORM_L2)
    knn = bf.knnMatch(desc_query, desc_ref, k=2)
    good = [m for pair in knn
            if len(pair) == 2 and pair[0].distance < MATCH_RATIO * pair[1].distance
            for m in (pair[0],)]
    return good


def _query_corners(h: int, w: int) -> np.ndarray:
    # 左上、右上、右下、左下
    return np.array(
        [[0.0, 0.0], [w, 0.0], [w, h], [0.0, h]], dtype=np.float64)


def _project_points(points: np.ndarray, H: np.ndarray) -> np.ndarray:
    n = points.shape[0]
    homog = np.concatenate(
        [points, np.ones((n, 1), dtype=np.float64)], axis=1)
    warped = homog @ H.T
    w = warped[:, 2]
    return warped[:, :2] / w[:, None]


def _polygon_area(poly: np.ndarray) -> float:
    x = poly[:, 0]
    y = poly[:, 1]
    return 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def _quad_angles_deg(poly: np.ndarray) -> np.ndarray:
    """按顶点顺序返回四个内角（度），凸四边形应均在 (0,180)。"""
    angles = []
    n = len(poly)
    for i in range(n):
        v1 = poly[(i - 1) % n] - poly[i]
        v2 = poly[(i + 1) % n] - poly[i]
        cosang = float(np.dot(v1, v2) / (
            np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-12))
        angles.append(math.degrees(math.acos(np.clip(cosang, -1.0, 1.0))))
    return np.array(angles)


def _is_convex(poly: np.ndarray) -> bool:
    """相邻边叉积同号（允许极小容差）才算凸。"""
    n = len(poly)
    signs = []
    for i in range(n):
        a = poly[(i + 1) % n] - poly[i]
        b = poly[(i + 2) % n] - poly[(i + 1) % n]
        cross = a[0] * b[1] - a[1] * b[0]
        if abs(cross) > 1e-9:
            signs.append(cross > 0)
    return bool(signs) and all(s == signs[0] for s in signs)


def _axis_lengths(poly: np.ndarray) -> tuple[float, float]:
    """顶/底与左/右两组对边的平均长度（允许透视带来的差异）。"""
    top = np.linalg.norm(poly[1] - poly[0])
    bottom = np.linalg.norm(poly[2] - poly[3])
    left = np.linalg.norm(poly[3] - poly[0])
    right = np.linalg.norm(poly[2] - poly[1])
    horiz = 0.5 * (top + bottom)
    vert = 0.5 * (left + right)
    return horiz, vert


def _rms_reprojection(
    src_pts: np.ndarray, dst_pts: np.ndarray, H: np.ndarray,
) -> float:
    """src -> dst 方向的内点重投影 RMS（像素）。"""
    proj = _project_points(src_pts, H)
    err = np.linalg.norm(proj - dst_pts, axis=1)
    return float(np.sqrt(np.mean(err ** 2))) if len(err) else float("inf")


# ---------------------------------------------------------------- 标注预览

def render_preview(
    ref_rgb: np.ndarray,
    quad_ref: np.ndarray,
    *,
    inliers: int,
    good_matches: int,
    inlier_ratio: float,
    coverage: float,
) -> str:
    """在参照图缩略图上绘制投影四边形并返回 data:image/jpeg;base64。"""
    h0, w0 = ref_rgb.shape[:2]
    s = _scale_for(w0, h0, PREVIEW_MAX_DIM)
    if s != 1.0:
        canvas = cv2.resize(
            ref_rgb, (max(1, int(round(w0 * s))), max(1, int(round(h0 * s)))),
            interpolation=cv2.INTER_AREA)
    else:
        canvas = ref_rgb.copy()
    ch, cw = canvas.shape[:2]
    q = quad_ref * s

    mean = ref_rgb.reshape(-1, 3).mean(axis=0)
    stroke = (255, 255, 255) if mean.mean() < 128 else (0, 0, 0)
    red = (54, 54, 235)      # RGB：醒目红
    yellow = (60, 220, 240)

    thickness = max(2, int(round(min(cw, ch) / 300)))
    # 外衬描边 + 红色主描边，保证在亮/暗背景上都清晰
    cv2.polylines(canvas, [q.astype(np.int32)], True, stroke,
                  thickness * 3, cv2.LINE_AA)
    cv2.polylines(canvas, [q.astype(np.int32)], True, red,
                  thickness, cv2.LINE_AA)

    # cv2 自带字体不含中文字形（会渲染成 ??），预览图内使用拉丁角标；
    # 中文角名仍随响应 JSON 的 quad[].label 返回，由前端展示。
    labels = ["TL", "TR", "BR", "BL"]
    radius = max(4, int(round(min(cw, ch) / 200)))
    # 角名相对角点的偏移：左角向右、右角向左，避免越出图像边缘
    offsets = [(8, -radius - 5), (-radius - 22, -radius - 5),
               (-radius - 22, radius + 16), (8, radius + 16)]
    for i, (x, y) in enumerate(q):
        cv2.circle(canvas, (int(round(x)), int(round(y))),
                   radius, yellow, -1, cv2.LINE_AA)
        cv2.circle(canvas, (int(round(x)), int(round(y))),
                   radius, stroke, 1, cv2.LINE_AA)
        tx = int(round(x)) + offsets[i][0]
        ty = int(round(y)) + offsets[i][1]
        # 四边形贴边时把文字收进画布，避免被裁掉
        tx = max(4, min(tx, cw - 34))
        ty = max(14, min(ty, ch - 6))
        cv2.putText(canvas, labels[i], (tx, ty),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, stroke, 3, cv2.LINE_AA)
        cv2.putText(canvas, labels[i], (tx, ty),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, yellow, 1, cv2.LINE_AA)

    lines = [
        f"coverage={coverage * 100:.2f}% of reference",
        f"RANSAC inliers={inliers}/{good_matches} ({inlier_ratio * 100:.1f}%)",
    ]
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = max(0.5, min(cw, ch) / 900)
    for i, text in enumerate(lines):
        y = ch - 10 - (len(lines) - 1 - i) * int(26 * scale)
        cv2.putText(canvas, text, (10, y), font, scale,
                    stroke, 4, cv2.LINE_AA)
        cv2.putText(canvas, text, (10, y), font, scale,
                    red, 1, cv2.LINE_AA)

    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR),
                           [int(cv2.IMWRITE_JPEG_QUALITY), 88])
    if not ok:
        raise LocateError("标注预览编码失败。")
    return "data:image/jpeg;base64," + base64.b64encode(
        buf.tobytes()).decode("ascii")


# ---------------------------------------------------------------- 主流程

def _fail(reason: str, stage: str, diagnostics: dict, checks: dict) -> dict:
    checks = {k: bool(v) for k, v in checks.items()}
    return {
        "located": False,
        "reason": reason,
        "failed_check": stage,
        "checks": checks,
        **diagnostics,
        "quad": None,
        "coverage": None,
        "preview_data_url": None,
        "thresholds": thresholds(),
    }


def locate_propagation_copy(
    ref_rgb: np.ndarray, query_rgb: np.ndarray,
) -> dict:
    """以参照图为基准定位传播副本；成功返回四角与覆盖率，失败返回原因。

    返回字典固定包含 ``located``；``located=False`` 时除诊断量外
    （``good_matches`` / ``inlier_count`` 等，能算多少算多少），
    ``quad`` / ``coverage`` / ``preview_data_url`` 均为 ``None``。
    """
    if not isinstance(ref_rgb, np.ndarray) or ref_rgb.ndim != 3:
        raise LocateError("参照图无效。")
    if not isinstance(query_rgb, np.ndarray) or query_rgb.ndim != 3:
        raise LocateError("上传图无效。")

    ref_h0, ref_w0 = ref_rgb.shape[:2]
    q_h0, q_w0 = query_rgb.shape[:2]
    ref_area0 = float(ref_w0 * ref_h0)

    s_ref = _scale_for(ref_w0, ref_h0, WORK_MAX_DIM)
    s_q = _scale_for(q_w0, q_h0, WORK_MAX_DIM)

    def prep(rgb, s):
        if s != 1.0:
            h, w = rgb.shape[:2]
            rgb = cv2.resize(rgb, (max(1, int(round(w * s))),
                                   max(1, int(round(h * s)))),
                             interpolation=cv2.INTER_AREA)
        return rgb

    ref_work = prep(ref_rgb, s_ref)
    query_work = prep(query_rgb, s_q)
    ref_gray = _to_gray(ref_work)
    query_gray = _to_gray(query_work)

    kp_ref, desc_ref = _detect_and_compute(ref_gray)
    kp_query, desc_query = _detect_and_compute(query_gray)

    good = _good_matches(desc_ref, desc_query)
    n_good = len(good)
    diagnostics = {
        "reference_size": {"width": int(ref_w0), "height": int(ref_h0)},
        "query_size": {"width": int(q_w0), "height": int(q_h0)},
        "keypoints_ref": int(len(kp_ref)),
        "keypoints_query": int(len(kp_query)),
        "good_matches": n_good,
        "inlier_count": None,
        "inlier_ratio": None,
        "rms_px": None,
    }
    checks = {CHECK_GOOD_MATCHES: n_good >= MIN_GOOD_MATCHES}
    if n_good < MIN_GOOD_MATCHES:
        return _fail(
            f"局部特征匹配不足：仅 {n_good} 对通过比率筛选"
            f"（至少需要 {MIN_GOOD_MATCHES} 对）。该副本可能裁切区域过小、"
            "画面过于平坦、经过重度缩放/压缩或并非来自本任务的含水印图。",
            CHECK_GOOD_MATCHES, diagnostics, checks)

    src = np.float32([kp_query[m.queryIdx].pt for m in good])  # 副本坐标
    dst = np.float32([kp_ref[m.trainIdx].pt for m in good])    # 参照图坐标

    H, mask = cv2.findHomography(
        src, dst, cv2.RANSAC,
        ransacReprojThreshold=RANSAC_REPROJ_THRESH,
        maxIters=5000, confidence=RANSAC_CONFIDENCE)
    checks[CHECK_HOMOGRAPHY] = H is not None
    if H is None or mask is None:
        return _fail(
            "RANSAC 无法估计出一致的单应矩阵（匹配对在几何上不一致），"
            "无法定位。",
            CHECK_HOMOGRAPHY, diagnostics, checks)

    inlier_mask = mask.ravel().astype(bool)
    n_in = int(inlier_mask.sum())
    inlier_ratio = n_in / n_good
    diagnostics["inlier_count"] = n_in
    diagnostics["inlier_ratio"] = round(inlier_ratio, 4)
    checks[CHECK_INLIER_COUNT] = bool(n_in >= MIN_INLIERS)
    checks[CHECK_INLIER_RATIO] = bool(inlier_ratio >= MIN_INLIER_RATIO)
    if n_in < MIN_INLIERS:
        return _fail(
            f"RANSAC 内点不足：仅 {n_in} 个内点"
            f"（至少需要 {MIN_INLIERS} 个），几何一致性不可信，无法定位。",
            CHECK_INLIER_COUNT, diagnostics, checks)
    if inlier_ratio < MIN_INLIER_RATIO:
        return _fail(
            f"RANSAC 内点比例过低：{n_in}/{n_good} = "
            f"{inlier_ratio * 100:.1f}%（至少需要 "
            f"{MIN_INLIER_RATIO * 100:.0f}%），匹配多为误配，无法定位。",
            CHECK_INLIER_RATIO, diagnostics, checks)

    inlier_src = src[inlier_mask]
    inlier_dst = dst[inlier_mask]

    # 用全部 RANSAC 内点重估单应（再做一次带权最小二乘细化），
    # 降低角点（尤其特征稀疏边缘）的外推误差；细化后须仍为良好单应。
    H_refined, _ = cv2.findHomography(inlier_src, inlier_dst, 0)
    if H_refined is not None and np.isfinite(H_refined).all():
        H = H_refined

    # 重投影 RMS（工作图口径 -> 换算回参照图原尺寸口径）
    rms_work = _rms_reprojection(inlier_src, inlier_dst, H)
    rms_ref = rms_work / s_ref if rms_work < float("inf") else float("inf")
    diagnostics["rms_px"] = round(rms_ref, 3) if math.isfinite(rms_ref) else None

    # 副本四角（工作图坐标）-> 参照图工作坐标 -> 参照图原尺寸坐标
    q_corners_work = _query_corners(query_gray.shape[0], query_gray.shape[1])
    quad_work = _project_points(q_corners_work, H)
    checks[CHECK_QUAD_FINITE] = bool(np.isfinite(quad_work).all())
    if not checks[CHECK_QUAD_FINITE]:
        return _fail(
            "单应矩阵将副本角点投影到无穷远（几何退化），无法定位。",
            CHECK_QUAD_FINITE, diagnostics, checks)

    quad_ref = quad_work / s_ref

    area = _polygon_area(quad_ref)
    coverage = area / ref_area0
    angles = _quad_angles_deg(quad_ref)
    horiz, vert = _axis_lengths(quad_ref)
    short_side = min(horiz, vert)
    ref_short = min(ref_w0, ref_h0)
    axis_ratio = max(horiz, vert) / max(min(horiz, vert), 1e-9)

    checks[CHECK_QUAD_SHAPE] = (
        _is_convex(quad_ref)
        and float(angles.min()) >= MIN_QUAD_ANGLE_DEG
        and short_side >= MIN_SIDE_FRACTION * ref_short
        and axis_ratio <= MAX_AXIS_SCALE_RATIO
    )
    if not checks[CHECK_QUAD_SHAPE]:
        return _fail(
            "投影四边形几何退化（非凸、出现极小夹角、过短边或两轴尺度比 "
            f"{axis_ratio:.1f}:1 异常），结果不可信，无法定位。",
            CHECK_QUAD_SHAPE, diagnostics, checks)

    margin = BOUND_MARGIN_FRAC * ref_short
    within_bounds = bool(
        np.all(quad_ref[:, 0] >= -margin)
        and np.all(quad_ref[:, 0] <= ref_w0 + margin)
        and np.all(quad_ref[:, 1] >= -margin)
        and np.all(quad_ref[:, 1] <= ref_h0 + margin))
    checks[CHECK_QUAD_BOUNDS] = within_bounds
    if not within_bounds:
        return _fail(
            "投影四边形明显越出参照图边界（副本与该任务的含水印图在范围上"
            "对不上），无法定位。",
            CHECK_QUAD_BOUNDS, diagnostics, checks)

    checks[CHECK_COVERAGE] = MIN_COVERAGE <= coverage <= MAX_COVERAGE
    if not checks[CHECK_COVERAGE]:
        return _fail(
            f"投影四边形面积覆盖率 {coverage * 100:.2f}% 超出可信区间"
            f"（{MIN_COVERAGE * 100:.1f}%~{MAX_COVERAGE * 100:.0f}%），"
            "几何结果不可信，无法定位。",
            CHECK_COVERAGE, diagnostics, checks)

    checks[CHECK_RMS] = math.isfinite(rms_ref) and rms_ref <= MAX_RMS_PX
    if not checks[CHECK_RMS]:
        rms_text = f"{rms_ref:.2f}" if math.isfinite(rms_ref) else "∞"
        return _fail(
            f"内点重投影误差过大（RMS {rms_text}px，上限 {MAX_RMS_PX:g}px），"
            "单应拟合不可信，无法定位。",
            CHECK_RMS, diagnostics, checks)

    # 全部门限通过：给出四角与预览
    preview = render_preview(
        ref_rgb, quad_ref,
        inliers=n_in, good_matches=n_good,
        inlier_ratio=inlier_ratio, coverage=coverage)

    corner_names = ["top_left", "top_right", "bottom_right", "bottom_left"]
    corner_labels = ["左上", "右上", "右下", "左下"]
    quad = []
    for i, (x, y) in enumerate(quad_ref):
        quad.append({
            "name": corner_names[i],
            "label": corner_labels[i],
            "x": round(float(x), 2),
            "y": round(float(y), 2),
        })

    return {
        "located": True,
        "reason": None,
        "failed_check": None,
        "checks": {k: bool(v) for k, v in checks.items()},
        **diagnostics,
        "inlier_ratio": round(inlier_ratio, 4),
        "rms_px": round(rms_ref, 3),
        "quad": quad,
        "coverage": round(float(coverage), 6),
        "coverage_percent": round(float(coverage) * 100, 2),
        "quad_area_px2": round(float(area), 1),
        "min_angle_deg": round(float(angles.min()), 2),
        "axis_scale_ratio": round(float(axis_ratio), 3),
        "preview_data_url": preview,
        "thresholds": thresholds(),
    }
