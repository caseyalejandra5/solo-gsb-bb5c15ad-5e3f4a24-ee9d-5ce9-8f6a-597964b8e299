"""核心算法数值验证：DCT 正交性、PN 自相关、嵌入/检测/攻击全链路。"""
import sys
import numpy as np
from PIL import Image

sys.path.insert(0, "/workspace/app")
import dctwm as wm


def make_test_image(size=512, seed=0):
    rng = np.random.RandomState(seed)
    # 类自然图像：低频渐变 + 中频纹理 + 噪声
    y, x = np.mgrid[0:size, 0:size]
    base = (
        128
        + 40 * np.sin(2 * np.pi * x / 130)
        + 30 * np.cos(2 * np.pi * y / 95)
        + 20 * np.sin(2 * np.pi * (x + y) / 60)
    )
    noise = rng.normal(0, 12, (size, size))
    ych = np.clip(base + noise, 0, 255)
    cr = np.clip(128 + 30 * np.sin(2 * np.pi * x / 200) + rng.normal(0, 8, (size, size)), 16, 240)
    cb = np.clip(128 + 30 * np.cos(2 * np.pi * y / 220) + rng.normal(0, 8, (size, size)), 16, 240)
    ycc = np.stack([ych, cr, cb], axis=-1).astype(np.uint8)
    import cv2
    return cv2.cvtColor(ycc, cv2.COLOR_YCrCb2RGB)


def test_dct_matches_cv2():
    rng = np.random.RandomState(1)
    blocks = rng.rand(1000, 8, 8) * 255
    mine = wm._dct_all(blocks)
    ref = np.stack([__import__("cv2").dct(b.astype(np.float64)) for b in blocks])
    err = np.abs(mine - ref).max()
    rt = np.abs(wm._idct_all(mine) - blocks).max()
    print(f"[DCT] 与 cv2.dct 最大误差={err:.2e}，IDCT 回环误差={rt:.2e}")
    assert err < 1e-8 and rt < 1e-8


def test_pn_autocorr():
    pn = wm.PN
    ac = [int((pn * np.roll(pn, k)).sum()) for k in range(len(pn))]
    print(f"[PN ] 峰值={ac[0]}，最大旁瓣={max(abs(v) for v in ac[1:])}")
    assert ac[0] == 31 and max(abs(v) for v in ac[1:]) <= 1


def run_case(name, img, text="水印ABC-2026", strength="medium", attack=None):
    emb = wm.embed_watermark(img, text, strength=strength)
    target = emb.image
    if attack:
        target = attack(emb.image)
    det = wm.detect_watermark(target)
    ok = det.detected and det.text == text
    print(
        f"[{'PASS' if ok else 'FAIL'}] {name}: PSNR={emb.psnr_db}dB "
        f"检测={det.detected} 文本={det.text!r} 置信={det.confidence} "
        f"同步={det.sync_score} 一致={det.sync_agreement} "
        f"瓦片={det.tiles_y}x{det.tiles_x} 相位={det.phase} "
        f"票/bit={det.votes_per_bit} 失真块={det.marked_blocks}/{det.total_blocks}"
        + (" " + det.reason if not ok else "")
    )
    assert ok, f"{name} failed"
    return emb, det


def main():
    test_dct_matches_cv2()
    test_pn_autocorr()

    img = make_test_image(512)
    for strength in ("low", "medium", "high"):
        run_case(f"无攻击/{strength}", img, "抗裁切水印测试", strength)

    run_case("JPEG Q90", img, "JPEG鲁棒", attack=lambda im: wm.jpeg_compress(im, 90))
    run_case("JPEG Q70", img, "JPEG鲁棒", attack=lambda im: wm.jpeg_compress(im, 70))
    run_case("JPEG Q50", img, "JPEG鲁棒", attack=lambda im: wm.jpeg_compress(im, 50))
    run_case("裁切90%", img, "中心裁切鲁棒", attack=lambda im: wm.center_crop(im, 0.9))
    run_case("裁切75%", img, "中心裁切鲁棒", attack=lambda im: wm.center_crop(im, 0.75))
    run_case("裁切50%", img, "中心裁切鲁棒", attack=lambda im: wm.center_crop(im, 0.5))
    run_case(
        "裁切75%+JPEG60",
        img, "组合攻击",
        attack=lambda im: wm.jpeg_compress(wm.center_crop(im, 0.75), 60),
    )
    run_case(
        "裁切50%+JPEG50",
        img, "组合攻击",
        attack=lambda im: wm.jpeg_compress(wm.center_crop(im, 0.5), 50),
    )

    # 无水印图应检测不到
    det = wm.detect_watermark(img)
    print(f"[{'PASS' if not det.detected else 'FAIL'}] 无水印图: {det.reason}")
    assert not det.detected

    # 小图（2x2 瓦片 = 256px）
    small = make_test_image(256, seed=3)
    run_case("小图256", small, "hi")

    # 叠加图可生成 PNG
    emb = wm.embed_watermark(img, "overlay测试")
    att = wm.jpeg_compress(wm.center_crop(emb.image, 0.75), 60)
    det = wm.detect_watermark(att)
    png = wm.to_png_bytes(det.overlay)
    Image.open(__import__("io").BytesIO(png)).verify()
    print(f"[PASS] 叠加图 PNG {len(png)//1024}KB")
    print("\n全部测试通过 ✔")


if __name__ == "__main__":
    main()
