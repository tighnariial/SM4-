import os
import cv2
import numpy as np
from watermark_embedding&extraction import embed_watermark, extract_watermark

def attack_jpeg(img, quality=50):
    cv2.imwrite('temp.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    return cv2.imread('temp.jpg')

def attack_noise(img, sigma=5):
    noisy = img + np.random.normal(0, sigma, img.shape).astype(np.uint8)
    return np.clip(noisy,0,255).astype(np.uint8)

def attack_crop(img, crop_ratio=0.2):
    h, w, _ = img.shape
    h_crop = int(h * crop_ratio)
    w_crop = int(w * crop_ratio)
    cropped = img[h_crop:h, w_crop:w].copy()
    return cv2.resize(cropped, (w, h))

def attack_rotate(img, angle=5):
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w//2,h//2), angle, 1)
    return cv2.warpAffine(img, M, (w,h), borderMode=cv2.BORDER_REFLECT)

def attack_blur(img, ksize=3):
    return cv2.GaussianBlur(img, (ksize, ksize), 0)

def batch_test_dir(img_dir, wm_bits, alpha=2.0, key=1234):
    img_files = [os.path.join(img_dir, f) for f in os.listdir(img_dir)
                 if f.lower().endswith(('.png', '.jpg', '.bmp'))]
    n_bits = len(wm_bits)
    results = []

    for img_path in img_files:
        img = cv2.imread(img_path)
        watermarked = embed_watermark(img, wm_bits, key=key, alpha=alpha)

        attacks = {
            'original': watermarked,
            'jpeg': attack_jpeg(watermarked, quality=50),
            'noise': attack_noise(watermarked, sigma=5),
            'crop': attack_crop(watermarked, crop_ratio=0.2),
            'rotate': attack_rotate(watermarked, angle=5),
            'blur': attack_blur(watermarked, ksize=3)
        }

        accs = {}
        for name, att_img in attacks.items():
            extracted = extract_watermark(att_img, n_bits, key=key, alpha=alpha)
            accs[name] = np.mean(extracted == wm_bits)
        results.append({'image': img_path, **accs})

    avg_acc = {k: np.mean([r[k] for r in results]) for k in attacks.keys()}
    print("每种攻击平均提取正确率:")
    for k,v in avg_acc.items():
        print(f"{k}: {v:.4f}")

    return results

if __name__ == "__main__":
    img_dir = './testsubject' 
    wm_bits = np.random.randint(0,2,64)
    results = batch_test_dir(img_dir, wm_bits)
    for r in results:
        print(r)
