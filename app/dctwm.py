"""分块 DCT 频域水印核心算法。

设计概述
========
* 全部处理在 YCrCb 的亮度(Y)通道完成，8x8 块 DCT（与 JPEG 一致的正交 DCT-II）。
* 每块嵌入 2 bit：利用两对中频系数的"差值调制"
      bit = 1 -> c(u1,v1) - c(u2,v2) > +delta
      bit = 0 -> 差值 < -delta
  中频系数对 JPEG 重压缩有天然鲁棒性，差值调制对图像符号变化不敏感。
* 块按 16x16 块（128x128 像素）为一个"瓦片"平铺：
      - 瓦片四角的 4x4 块区域为同步块（96 个），携带 31 位 m 序列伪随机标记；
      - 其余 160 个块为载荷块，携带经重复编码(R=2)的文本码流。
* 检测时对 64 种块相移做全相关搜索定位同步网格（抗任意 8 像素对齐的裁切），
  再把所有瓦片相同位置的载荷块投票合并（跨瓦片多数表决）。
* 码流 = 魔数(0xAB) + 载荷长度 + UTF-8 文本 + 校验和；检测端按长度假设
  从大到小扫描，魔数与校验同时通过才采信。
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field
import numpy as np
import cv2
from PIL import Image

# ---------------------------------------------------------------- 常量参数

BLK = 8                 # DCT 块边长（像素）
TILE = 16               # 一个瓦片的块边长
SYNC = 4                # 瓦片四角同步区的块边长
SYNC_MIN_SAMPLES = 12
SYNC_SCORE_THRESHOLD = 0.42     # 归一化相关系数阈值
WEAK_MARGIN = 14.0              # |软值| 小于此值视为可疑（可能失真）

# 强度档位 -> 基础嵌入差值 delta（约按 JPEG 量化步长标定）
STRENGTH_DELTA = {"low": 26.0, "medium": 36.0, "high": 48.0}

PTS_PER_TILE = TILE * TILE
SYNC_POINTS = 4 * SYNC * SYNC                    # 64 个同步块（四角）
SLOTS = PTS_PER_TILE - SYNC_POINTS              # 192 个载荷槽/瓦片
STREAM_BITS_MAX = 2 * SLOTS                     # 每个载荷块 2 bit → 384 bit

# 两个中频系数对（行, 列），JPEG Q50 亮度量化步长分别约 59/48 与 64/40
PAIR_A = ((3, 2), (2, 3))
PAIR_B = ((4, 1), (1, 4))

# 码流结构
MAGIC = 0xAB
STREAM_FIXED_BYTES = 3          # magic + length + checksum


# ---------------------------------------------------------------- 基础工具

def _build_dct_matrix() -> np.ndarray:
    """与 cv2.dct 一致的正交 DCT-II 矩阵 C；正变换 C@x@C^T，逆变换 C^T@X@C。"""
    n = np.arange(BLK)
    k = n.reshape(-1, 1)
    c = np.cos(np.pi * (2 * n + 1) * k / (2 * BLK))
    c *= np.sqrt(2.0 / BLK)
    c[0] = 1.0 / np.sqrt(BLK)
    return c.astype(np.float64)


D = _build_dct_matrix()
DT = D.T


def _build_pn(length: int = 31) -> np.ndarray:
    """5 级 LFSR m 序列（本原多项式 x^5+x^2+1），长度 31，自相关旁瓣极低。"""
    bits = []
    state = 1
    for _ in range(length):
        bits.append(state & 1)
        fb = ((state & 1) ^ ((state >> 2) & 1)) & 1
        state = (state >> 1) | (fb << 4)
    pn = np.array(bits, dtype=np.int8) * 2 - 1
    return pn


PN = _build_pn()


def _corner_points() -> list[tuple[int, int]]:
    """瓦片内同步块的 (残行, 残列) 坐标：四个 4x4 角。"""
    pts: list[tuple[int, int]] = []
    for r in range(TILE):
        for c in range(TILE):
            in_corner = (
                (r < SYNC or r >= TILE - SYNC)
                and (c < SYNC or c >= TILE - SYNC)
            )
            if in_corner:
                pts.append((r, c))
    return pts


CORNER_POINTS = _corner_points()

# 瓦片(16x16)模板：+1/-1 同步 PN 权重，0 表示载荷块
_TILE_WEIGHT = np.zeros((TILE, TILE), dtype=np.float64)
# 同步块在角点列表中的序号
_TILE_PT_IDX = np.full((TILE, TILE), -1, dtype=np.int32)
# 载荷块在槽位序列中的序号（按行优先）
_TILE_SLOT = np.full((TILE, TILE), -1, dtype=np.int32)
for idx, (r, c) in enumerate(CORNER_POINTS):
    _TILE_WEIGHT[r, c] = PN[idx % len(PN)]
    _TILE_PT_IDX[r, c] = idx
_slot = 0
for r in range(TILE):
    for c in range(TILE):
        if _TILE_WEIGHT[r, c] == 0.0:
            _TILE_SLOT[r, c] = _slot
            _slot += 1
assert _slot == SLOTS

# 64 种瓦片相移（块单位）
ALL_PHASES = [(dr, dc) for dr in range(TILE) for dc in range(TILE)]


class WatermarkError(ValueError):
    """水印参数 / 图片尺寸等可读错误。"""


# ---------------------------------------------------------------- 图像转换

def load_luma(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """RGB uint8 -> (Y float64, YCrCb uint8)。"""
    ycc = cv2.cvtColor(rgb, cv2.COLOR_RGB2YCrCb)
    return ycc[:, :, 0].astype(np.float64), ycc


def merge_luma(ycc: np.ndarray, y_new: np.ndarray) -> np.ndarray:
    """把新的 Y 通道合回并转回 RGB uint8（已做 round/clip）。"""
    out = ycc.copy()
    out[:, :, 0] = np.clip(np.rint(y_new), 0, 255).astype(np.uint8)
    return cv2.cvtColor(out, cv2.COLOR_YCrCb2RGB)


def _blocks_of(y: np.ndarray, gh: int | None = None, gw: int | None = None) -> np.ndarray:
    """Y 平面 -> (gh, gw, 8, 8) 块视图（拷贝）。"""
    H, W = y.shape
    gh = gh if gh is not None else H // BLK
    gw = gw if gw is not None else W // BLK
    return y[:gh * BLK, :gw * BLK].reshape(
        gh, BLK, gw, BLK
    ).transpose(0, 2, 1, 3).copy()


def _dct_all(blocks: np.ndarray) -> np.ndarray:
    tmp = np.matmul(D, blocks)
    return np.matmul(tmp, DT)


def _idct_all(coef: np.ndarray) -> np.ndarray:
    tmp = np.matmul(DT, coef)
    return np.matmul(tmp, D)


# ---------------------------------------------------------------- 码流编解码

def encode_stream(text: str) -> tuple[np.ndarray, int]:
    """文本 -> ±1 码流（未做瓦片内重复，码流在各瓦片按槽位周期平铺）。"""
    payload = text.encode("utf-8")
    if len(payload) > capacity_bytes(1):
        raise WatermarkError(
            f"文本过长：最多 {capacity_bytes(1)} 字节"
            "（ASCII 约 45 字符，UTF-8 中文约 15 个汉字）。"
        )
    body = bytes([MAGIC, len(payload)]) + payload
    chk = sum(payload) & 0xFF
    body += bytes([chk])
    bits = np.unpackbits(np.frombuffer(body, dtype=np.uint8)).astype(np.int8)
    return bits.astype(np.float64) * 2 - 1, len(payload)


def capacity_bytes(full_tiles: int = 1) -> int:
    """单个完整瓦片可容纳的最大文本字节数（跨瓦片复制，不影响容量）。"""
    return STREAM_BITS_MAX // 8 - STREAM_FIXED_BYTES


# ---------------------------------------------------------------- 嵌入

@dataclass
class EmbedResult:
    image: np.ndarray                 # RGB uint8 含水印图
    psnr_db: float
    grid_blocks: int
    full_tiles: int
    payload_bytes: int
    capacity_bytes: int
    votes_per_bit: int                # 单个码流 bit 的跨瓦片复制票数
    strength: str


def embed_watermark(rgb: np.ndarray, text: str, strength: str = "medium") -> EmbedResult:
    if strength not in STRENGTH_DELTA:
        raise WatermarkError(f"未知强度档位：{strength}")
    if not text:
        raise WatermarkError("水印文本不能为空。")

    coded, plen = encode_stream(text)
    clen = len(coded)

    h, w = rgb.shape[:2]
    gh, gw = h // BLK, w // BLK
    gb = min(gh, gw)
    if gb < TILE:
        raise WatermarkError(
            f"图片过小：最短边至少需要 {TILE * BLK}px（当前 {min(h, w)}px）。"
        )
    ntr, ntc = gh // TILE, gw // TILE
    full_tiles = ntr * ntc
    cap = capacity_bytes(1)
    if plen > cap:
        raise WatermarkError(
            f"文本 {plen} 字节超出单瓦片容量 {cap} 字节，请缩短文本。"
        )

    y, ycc = load_luma(rgb)
    blocks = _blocks_of(y, gh, gw)
    y_orig = y[: gh * BLK, : gw * BLK].copy()

    # 块网格坐标（每个瓦片都嵌入同一份码流，故只依赖残坐标槽位）
    rr_h, rr_w = np.arange(gh) % TILE, np.arange(gw) % TILE
    rr_g, rc_g = np.meshgrid(rr_h, rr_w, indexing="ij")

    slot = _TILE_SLOT[rr_g, rc_g]                  # -1 表示同步块
    is_payload = slot >= 0
    is_sync = ~is_payload

    # 槽位 -> 码流 bit 索引（码流在瓦片内周期平铺）
    pos = 2 * np.maximum(slot, 0)
    w_a = np.where(
        is_payload,
        coded[pos % clen],
        _TILE_WEIGHT[rr_g, rc_g],                  # 同步块：PN 符号
    )
    w_b = np.where(is_payload, coded[(pos + 1) % clen], 0.0)

    # 纹理自适应：纹理越弱 delta 越小（更不可见），但不低于 1.0 倍（保 JPEG 存活）
    texture = blocks.std(axis=(2, 3))
    factor = np.clip(0.9 + texture / 40.0, 1.0, 1.3)
    delta = STRENGTH_DELTA[strength] * factor
    t_a = delta
    t_b = np.where(is_payload, delta, 0.0)

    (a0r, a0c), (a1r, a1c) = PAIR_A
    (b0r, b0c), (b1r, b1c) = PAIR_B

    # 多轮迭代补偿 IDCT 舍入 / 裁剪带来的差值回落
    for _ in range(3):
        coef = _dct_all(blocks)
        ca1, ca2 = coef[..., a0r, a0c], coef[..., a1r, a1c]
        diff_a = ca1 - ca2
        need_a = np.maximum(t_a - diff_a * w_a, 0.0)
        coef[..., a0r, a0c] = ca1 + need_a * w_a / 2.0
        coef[..., a1r, a1c] = ca2 - need_a * w_a / 2.0

        cb1, cb2 = coef[..., b0r, b0c], coef[..., b1r, b1c]
        diff_b = cb1 - cb2
        need_b = np.maximum(t_b - diff_b * w_b, 0.0)
        coef[..., b0r, b0c] = cb1 + need_b * w_b / 2.0
        coef[..., b1r, b1c] = cb2 - need_b * w_b / 2.0

        blocks = np.clip(np.rint(_idct_all(coef)), 0, 255)

    y_new = y.copy()
    y_new[: gh * BLK, : gw * BLK] = blocks.transpose(0, 2, 1, 3).reshape(
        gh * BLK, gw * BLK
    )
    out_rgb = merge_luma(ycc, y_new)

    mse = float(np.mean((y_orig - y_new[: gh * BLK, : gw * BLK]) ** 2))
    psnr = 99.0 if mse < 1e-9 else float(20 * np.log10(255.0 / np.sqrt(mse)))

    return EmbedResult(
        image=out_rgb,
        psnr_db=round(psnr, 2),
        grid_blocks=int(min(gh, gw)),
        full_tiles=full_tiles,
        payload_bytes=plen,
        capacity_bytes=cap,
        votes_per_bit=full_tiles,
        strength=strength,
    )


# ---------------------------------------------------------------- 攻击模拟

def jpeg_compress(rgb: np.ndarray, quality: int) -> np.ndarray:
    quality = int(np.clip(quality, 10, 100))
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="JPEG", quality=quality, subsampling=2)
    buf.seek(0)
    return np.array(Image.open(buf).convert("RGB"))


def center_crop(rgb: np.ndarray, fraction: float) -> np.ndarray:
    """按比例中心裁切，边界吸附到 8 像素以与 DCT 网格对齐（实验台约定）。"""
    fraction = float(np.clip(fraction, 0.2, 1.0))
    h, w = rgb.shape[:2]
    side_h = int(round(h * fraction / BLK)) * BLK
    side_w = int(round(w * fraction / BLK)) * BLK
    side_h = max(side_h, TILE * BLK)
    side_w = max(side_w, TILE * BLK)
    y0 = ((h - side_h) // 2 // BLK) * BLK
    x0 = ((w - side_w) // 2 // BLK) * BLK
    return rgb[y0:y0 + side_h, x0:x0 + side_w].copy()


# ---------------------------------------------------------------- 检测

@dataclass
class DetectResult:
    detected: bool
    text: str = ""
    confidence: float = 0.0          # 0~100
    sync_score: float = 0.0          # 归一化相关 -1~1
    sync_agreement: float = 0.0      # 同步块符号一致比例
    bit_confidences: list[float] = field(default_factory=list)
    payload_bytes: int = 0
    payload_blocks: int = 0
    sync_blocks: int = 0
    tiles_x: int = 0
    tiles_y: int = 0
    phase: tuple[int, int] = (0, 0)
    votes_per_bit: int = 0
    marked_blocks: int = 0
    total_blocks: int = 0
    marked_fraction: float = 0.0
    reason: str = ""
    overlay: np.ndarray | None = None


def _tile_weight_obs(or_: int, oc_: int) -> np.ndarray:
    """相移 (or_,oc_) 下，观测残坐标块对应的同步权重（16x16）。"""
    return np.roll(np.roll(_TILE_WEIGHT, or_, axis=0), oc_, axis=1)


def detect_watermark(rgb: np.ndarray) -> DetectResult:
    h, w = rgb.shape[:2]
    gh, gw = h // BLK, w // BLK
    gb = min(gh, gw)
    total_blocks = gh * gw
    if gb < TILE:
        return DetectResult(False, total_blocks=total_blocks,
                            reason=f"图片过小（最短边 {min(h, w)}px），不足以容纳同步网格。")

    y, _ = load_luma(rgb)
    blocks = _blocks_of(y, gh, gw)
    coef = _dct_all(blocks)

    (a0r, a0c), (a1r, a1c) = PAIR_A
    (b0r, b0c), (b1r, b1c) = PAIR_B
    soft_a = coef[..., a0r, a0c] - coef[..., a1r, a1c]
    soft_b = coef[..., b0r, b0c] - coef[..., b1r, b1c]

    ridx_h, ridx_w = np.arange(gh), np.arange(gw)
    tr_g, tc_g = np.meshgrid(ridx_h // TILE, ridx_w // TILE, indexing="ij")
    rr_g, rc_g = np.meshgrid(ridx_h % TILE, ridx_w % TILE, indexing="ij")
    ntr = (gh + TILE - 1) // TILE
    ntc = (gw + TILE - 1) // TILE

    # ---- 1) 64 相移全相关搜索同步网格
    best = (-1.0, 0, 0, 1.0)        # |score|, or, oc, 符号
    abs_a = np.abs(soft_a)
    for or_, oc_ in ALL_PHASES:
        wobs = _tile_weight_obs(or_, oc_)
        wgrid = wobs[rr_g, rc_g]
        mask = wgrid != 0.0
        denom = float(abs_a[mask].sum())
        if denom < 1e-9:
            continue
        score = float((wgrid[mask] * soft_a[mask]).sum() / denom)
        if abs(score) > best[0]:
            best = (abs(score), or_, oc_, 1.0 if score >= 0 else -1.0)

    score_abs, or_, oc_, sign_flip = best
    if score_abs < SYNC_SCORE_THRESHOLD:
        return DetectResult(
            False, sync_score=round(score_abs, 3), total_blocks=total_blocks,
            phase=(or_, oc_),
            reason="未找到同步网格（相关峰值过低），图片可能未嵌入水印或已被严重破坏。",
        )

    # 代码残坐标（每个瓦片嵌入同一份码流，只需对齐残坐标，不依赖绝对瓦片号）
    crr = (rr_g - or_) % TILE
    crc = (rc_g - oc_) % TILE

    sync_mask = _TILE_WEIGHT[crr, crc] != 0.0
    n_sync = int(sync_mask.sum())
    if n_sync < SYNC_MIN_SAMPLES:
        return DetectResult(
            False, sync_score=round(score_abs, 3), total_blocks=total_blocks,
            sync_blocks=n_sync, phase=(or_, oc_),
            reason="同步块样本过少，裁切可能过于严重。",
        )

    # ---- 2) 同步块一致性
    pn_sign = _TILE_WEIGHT[crr, crc] * sign_flip
    sync_soft = soft_a * sign_flip
    sync_ok = sync_mask & (np.sign(sync_soft) == np.sign(pn_sign))
    sync_agreement = float(sync_ok.sum()) / n_sync

    # ---- 3) 载荷块投票（相同槽位跨瓦片多数表决）
    slot = _TILE_SLOT[crr, crc]
    pay_mask = slot >= 0
    n_pay = int(pay_mask.sum())
    if n_pay < SLOTS:
        return DetectResult(
            False, sync_score=round(score_abs, 3), sync_blocks=n_sync,
            sync_agreement=round(sync_agreement, 3), total_blocks=total_blocks,
            phase=(or_, oc_),
            reason="有效载荷块不足一个完整瓦片，无法恢复文本。",
        )

    pos = 2 * np.maximum(slot, 0)
    pos_p = pos[pay_mask]
    sa = soft_a[pay_mask] * sign_flip
    sb = soft_b[pay_mask] * sign_flip
    residues = np.concatenate([sa, sb])
    base_idx = np.concatenate([pos_p, pos_p + 1]).astype(np.int64)

    max_l = STREAM_BITS_MAX // 8 - STREAM_FIXED_BYTES     # 37 字节

    def try_decode(length: int):
        stream_bytes_len = STREAM_FIXED_BYTES + length
        clen = 8 * stream_bytes_len
        idx = base_idx % clen
        cnt = np.bincount(idx, minlength=clen).astype(np.float64)
        res = np.bincount(idx, weights=residues, minlength=clen)
        votes = np.bincount(idx, weights=np.sign(residues), minlength=clen)
        if not (cnt > 0).all():
            return None
        bit_val = (res > 0).astype(np.uint8)
        data = np.packbits(bit_val)
        magic, L = int(data[0]), int(data[1])
        if magic != MAGIC or L != length or L + STREAM_FIXED_BYTES > len(data):
            return None
        payload = bytes(data[2:2 + L])
        chk = int(data[2 + L])
        if (sum(payload) & 0xFF) != chk:
            return None
        mean_margin = float(np.mean(np.abs(res / cnt)))
        return {
            "payload": payload, "res": res, "votes": votes, "cnt": cnt,
            "clen": clen, "margin": mean_margin,
        }

    chosen = None
    for L in range(max_l, -1, -1):
        chosen = try_decode(L)
        if chosen is not None:
            break

    if chosen is None:
        return DetectResult(
            False, sync_score=round(score_abs, 3),
            sync_agreement=round(sync_agreement, 3),
            sync_blocks=n_sync, payload_blocks=n_pay,
            tiles_x=ntc, tiles_y=ntr, total_blocks=total_blocks, phase=(or_, oc_),
            reason="同步成功但文本校验失败（码流可能被 JPEG/裁切破坏）。",
        )

    # ---- 4) 置信度与逐块失真标记
    res_s, vot_s, cnt_s = chosen["res"], chosen["votes"], chosen["cnt"]
    mean_res = res_s / np.maximum(cnt_s, 1)
    vote_ratio = np.abs(vot_s) / np.maximum(cnt_s, 1)
    conf_res = np.clip(np.abs(mean_res) / 24.0, 0, 1)
    conf_vote = np.clip((vote_ratio - 0.5) / 0.5, 0, 1)
    bit_conf = np.clip(0.5 * conf_res + 0.5 * conf_vote, 0, 1)
    confidence = float(np.mean(bit_conf)) * 100.0

    clen = chosen["clen"]
    header = np.concatenate([
        np.array([MAGIC, len(chosen["payload"])], dtype=np.uint8),
        np.frombuffer(chosen["payload"], dtype=np.uint8),
        np.array([sum(chosen["payload"]) & 0xFF], dtype=np.uint8),
    ])
    expected_coded = np.unpackbits(header).astype(np.float64) * 2 - 1

    idx_a = pos % clen
    idx_b = (pos + 1) % clen
    exp_a = expected_coded[idx_a]
    exp_b = expected_coded[idx_b]

    block_marked = np.zeros((gh, gw), dtype=bool)
    block_ok = np.zeros((gh, gw), dtype=bool)
    block_weak = np.zeros((gh, gw), dtype=bool)

    block_ok[sync_mask] = sync_ok[sync_mask]
    block_marked[sync_mask & ~sync_ok] = True

    pa_ok = np.sign(soft_a * sign_flip) == exp_a
    pb_ok = np.sign(soft_b * sign_flip) == exp_b
    pay_right = pay_mask & pa_ok & pb_ok
    pay_wrong = pay_mask & (~pa_ok | ~pb_ok)
    pay_weak = pay_right & (
        (np.abs(soft_a * sign_flip) < WEAK_MARGIN)
        | (np.abs(soft_b * sign_flip) < WEAK_MARGIN)
    )
    block_ok[pay_right] = True
    block_weak[pay_weak] = True
    block_marked[pay_wrong | pay_weak] = True

    marked = int(block_marked.sum())

    # 瓦片同步健康度（按观测瓦片）
    tile_health = np.zeros((ntr, ntc), dtype=np.float64)
    for tr_i in range(ntr):
        for tc_i in range(ntc):
            sel = sync_mask & (tr_g == tr_i) & (tc_g == tc_i)
            if sel.any():
                tile_health[tr_i, tc_i] = float(sync_ok[sel].sum()) / sel.sum()

    overlay = _render_overlay(
        rgb, gh, gw, sync_mask, pay_mask, block_ok, block_weak, block_marked,
        tile_health, tr_g, tc_g,
    )

    try:
        text = chosen["payload"].decode("utf-8")
    except UnicodeDecodeError:
        text = chosen["payload"].decode("utf-8", errors="replace")

    return DetectResult(
        detected=True,
        text=text,
        confidence=round(confidence, 1),
        sync_score=round(float(score_abs * sign_flip), 3),
        sync_agreement=round(sync_agreement, 3),
        bit_confidences=[round(float(x), 3) for x in bit_conf],
        payload_bytes=len(chosen["payload"]),
        payload_blocks=n_pay,
        sync_blocks=n_sync,
        tiles_x=ntc,
        tiles_y=ntr,
        phase=(int(or_), int(oc_)),
        votes_per_bit=int(cnt_s.min() // 1),
        marked_blocks=marked,
        total_blocks=total_blocks,
        marked_fraction=round(marked / max(total_blocks, 1), 4),
        overlay=overlay,
    )


def _render_overlay(
    rgb: np.ndarray, gh: int, gw: int,
    sync_mask: np.ndarray, pay_mask: np.ndarray,
    block_ok: np.ndarray, block_weak: np.ndarray, block_marked: np.ndarray,
    tile_health: np.ndarray, tr_g: np.ndarray, tc_g: np.ndarray,
) -> np.ndarray:
    """在待检测图上叠加瓦片健康色与疑似失真块（全向量化、单次合成）。"""
    h, w = rgb.shape[:2]

    # 按优先级（数字越大越优先）与不透明度构造块网格着色表
    health = tile_health[tr_g, tc_g]
    prio = np.zeros((gh, gw), dtype=np.int32)
    col_r = np.zeros((gh, gw), dtype=np.uint8)
    col_g = np.zeros((gh, gw), dtype=np.uint8)
    col_b = np.zeros((gh, gw), dtype=np.uint8)
    alpha = np.zeros((gh, gw), dtype=np.uint8)

    def paint(mask, color, a, p):
        m = mask & (prio <= p)
        prio[m] = p
        col_r[m], col_g[m], col_b[m], alpha[m] = color[0], color[1], color[2], a

    # 瓦片健康底色（优先级最低）
    paint(health > 0.8, (60, 200, 120), 22, 1)
    paint((health > 0.0) & (health > 0.4) & (health <= 0.8), (240, 170, 40), 32, 1)
    paint((health > 0.0) & (health <= 0.4), (235, 70, 70), 45, 1)
    # 同步块
    paint(sync_mask & block_ok, (60, 220, 120), 40, 2)
    paint(sync_mask & ~block_ok, (235, 60, 60), 100, 3)
    # 载荷块
    paint(pay_mask & block_marked & ~block_weak, (235, 60, 60), 110, 3)
    paint(block_weak, (245, 210, 60), 80, 3)

    # 展开到像素网格
    ar = np.repeat(np.repeat(alpha, BLK, axis=0), BLK, axis=1)[:h, :w]
    cr_ = np.repeat(np.repeat(col_r, BLK, axis=0), BLK, axis=1)[:h, :w]
    cg_ = np.repeat(np.repeat(col_g, BLK, axis=0), BLK, axis=1)[:h, :w]
    cb_ = np.repeat(np.repeat(col_b, BLK, axis=0), BLK, axis=1)[:h, :w]

    # 定点 alpha 混合：out = src*(255-a)/255 + color*a/255
    a255 = (255 - ar).astype(np.int32)
    base = rgb[:h, :w].astype(np.int16)
    merged_r = (base[..., 0].astype(np.int32) * a255 + cr_.astype(np.int32) * ar) // 255
    merged_g = (base[..., 1].astype(np.int32) * a255 + cg_.astype(np.int32) * ar) // 255
    merged_b = (base[..., 2].astype(np.int32) * a255 + cb_.astype(np.int32) * ar) // 255

    out = rgb.copy()
    region = np.stack(
        [merged_r, merged_g, merged_b], axis=-1
    ).clip(0, 255).astype(np.uint8)
    active = ar > 0
    out[:h, :w][active] = region[active]

    # 瓦片边框（蓝线）
    span = TILE * BLK
    for y in range(0, gh * BLK + 1, span):
        if y < h:
            out[max(y - 1, 0):y + 1, :gw * BLK] = (70, 150, 255)
    for x in range(0, gw * BLK + 1, span):
        if x < w:
            out[:gh * BLK, max(x - 1, 0):x + 1] = (70, 150, 255)
    return out


# ---------------------------------------------------------------- PNG 编解码

def to_png_bytes(rgb: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="PNG", optimize=True)
    return buf.getvalue()
