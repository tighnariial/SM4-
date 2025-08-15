import numpy as np
import cv2
from numpy.random import default_rng

//dct变换
def dct2(block):
    return cv2.dct(block.astype(np.float32))
//dct逆变换
def idct2(block):
    return cv2.idct(block.astype(np.float32))
//RGB图像变为YCbCr图像，便于在Y通道中隐藏细节
def rgb_to_ycbcr(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
//YCbCr图像变为RGB图像
def ycbcr_to_rgb(img_ycrcb):
    return cv2.cvtColor(img_ycrcb, cv2.COLOR_YCrCb2BGR)
//图像分块为8*8
def split_blocks(ch, bs=8):
    h, w = ch.shape
    blocks = (ch.reshape(h//bs, bs, -1, bs)
                .swapaxes(1,2)
                .reshape(-1, bs, bs))
    return blocks, (h, w)
//融合分块图片
def merge_blocks(blocks, shape, bs=8):
    h, w = shape
    n_h, n_w = h//bs, w//bs
    out = (blocks.reshape(n_h, n_w, bs, bs)
              .swapaxes(1,2)
              .reshape(h, w))
    return out
    
//在中频通道中嵌入水印，达到鲁棒性与不可见性的平衡；对应中频通道掩码
def midband_mask():
    mask = np.zeros((8,8), dtype=bool)
    coords = [(3,2),(2,3),(4,1),(1,4)]
    for r, c in coords:
        mask[r,c] = True
    return mask

//使用spread_specrtum法嵌入水印
def embed_spread_spectrum(img_bgr, wm_bits, key=1234, alpha=2.0):
    ycrcb = rgb_to_ycbcr(img_bgr)
    Y = ycrcb[:,:,0].astype(np.float32)//转换ycbcr表示
    blocks, shape = split_blocks(Y)
    mask = midband_mask()
    idx = np.where(mask.flatten())[0]
    m = len(idx)//掩码为1的通道数

    rng = default_rng(key)//根据输入密钥生成随机pn序列
    pn = rng.choice([-1.0, 1.0], size=m).astype(np.float32)

    wm = np.array(wm_bits, dtype=np.uint8)//调整水印bit数，使其对应图片的每一个block
    if len(wm) < len(blocks):
        reps = (len(blocks) + len(wm)-1)//len(wm)
        wm = np.tile(wm, reps)[:len(blocks)]
    else:
        wm = wm[:len(blocks)]

    out_blocks = []
    for b, bit in zip(blocks, wm)://对原图的每一block做DCT
        B = dct2(b).flatten()
        B[idx] += alpha * pn * (1 if bit else -1)//对于中频序列的每一bit，根据pn以及水印对应位的bit，调整其取值
        out_blocks.append(idct2(B.reshape(8,8)))

    out_blocks = np.stack(out_blocks)
    Yw = np.clip(merge_blocks(out_blocks, shape), 0, 255).astype(np.uint8)
    ycrcb[:,:,0] = Yw//得到嵌入水印后的RGB格式图像
    return ycbcr_to_rgb(ycrcb)
    
//提取水印
def extract_spread_spectrum(img_bgr_w, n_bits, key=1234, alpha=2.0):
    ycrcb = rgb_to_ycbcr(img_bgr_w)
    Y = ycrcb[:,:,0].astype(np.float32)
    blocks, _ = split_blocks(Y)
    mask = midband_mask()
    idx = np.where(mask.flatten())[0]
    m = len(idx)

    rng = default_rng(key)
    pn = rng.choice([-1.0, 1.0], size=m).astype(np.float32)

    bits = []
    for b in blocks[:n_bits]:
        B = dct2(b).flatten()
        sub = B[idx]
        corr = float(np.dot(sub, pn))
        bits.append(1 if corr > 0 else 0)

    return np.array(bits, dtype=np.uint8)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    img = cv2.imread("input.png")
    watermark = [1,0,1,1,0,0,1,0]
    watermarked = embed_spread_spectrum(img, watermark, key=2025, alpha=3.0)
    cv2.imwrite("watermarked.png", watermarked)

    extracted = extract_spread_spectrum(watermarked, len(watermark), key=2025)
    print("Extracted bits:", extracted)
