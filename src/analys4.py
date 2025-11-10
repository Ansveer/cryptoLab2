import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim, peak_signal_noise_ratio as psnr
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from steganalysis.attack import rs_analysis
from sewar.full_ref import msssim, uqi
import os

def professional_analysis(cover_path, stego_path, output_dir):
    # Загрузка изображений
    cover = cv2.imread(cover_path)
    stego = cv2.imread(stego_path)
    
    # Конвертация в RGB для корректной работы метрик
    cover_rgb = cv2.cvtColor(cover, cv2.COLOR_BGR2RGB)
    stego_rgb = cv2.cvtColor(stego, cv2.COLOR_BGR2RGB)
    
    # 1. Расчет всех метрик качества
    metrics = {}
    
    # PSNR
    metrics['psnr'] = psnr(cover_rgb, stego_rgb, data_range=255)
    
    # SSIM
    metrics['ssim'] = ssim(cover_rgb, stego_rgb, channel_axis=2, data_range=255)
    
    # MSSSIM (более продвинутая версия SSIM)
    metrics['msssim'] = msssim(cover_rgb, stego_rgb)
    
    # Universal Quality Index
    metrics['uqi'] = uqi(cover_rgb, stego_rgb)
    
    # 2. RS-анализ
    try:
        rs_result = rs_analysis(stego_rgb)
        metrics['rs_difference'] = rs_result['R_m'] - rs_result['S_m']
        metrics['rs_detection'] = abs(rs_result['R_m'] - rs_result['S_m']) > 0.02
    except:
        metrics['rs_detection'] = "Analysis failed"
    
    # 3. Разностное изображение
    diff = cv2.absdiff(cover_rgb, stego_rgb)
    diff_normalized = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX)
    
    # 4. Гистограммы
    plt.figure(figsize=(15, 10))
    
    for i, color in enumerate(['Red', 'Green', 'Blue']):
        plt.subplot(2, 3, i+1)
        plt.hist(cover_rgb[:,:,i].ravel(), bins=256, alpha=0.7, color=color.lower(), label='Cover')
        plt.hist(stego_rgb[:,:,i].ravel(), bins=256, alpha=0.7, color='black', label='Stego')
        plt.title(f'{color} Channel Histogram')
        plt.legend()
    
    # 5. Визуализация результатов
    plt.subplot(2, 3, 4)
    plt.imshow(cover_rgb)
    plt.title('Cover Image')
    plt.axis('off')
    
    plt.subplot(2, 3, 5)
    plt.imshow(stego_rgb)
    plt.title('Stego Image')
    plt.axis('off')
    
    plt.subplot(2, 3, 6)
    plt.imshow(diff_normalized)
    plt.title('Difference Map')
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/professional_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Вывод результатов
    print("=== PROFESSIONAL STEGANALYSIS RESULTS ===")
    for metric, value in metrics.items():
        print(f"{metric.upper()}: {value}")
    
    return metrics

def bulk_analysis(cover_dir, stego_dir, output_dir):
    """Анализ набора изображений"""
    cover_images = [f for f in os.listdir(cover_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
    
    all_metrics = []
    
    for img_name in cover_images:
        cover_path = os.path.join(cover_dir, img_name)
        stego_path = os.path.join(stego_dir, img_name)
        
        if os.path.exists(stego_path):
            metrics = professional_analysis(cover_path, stego_path, output_dir)
            metrics['image'] = img_name
            all_metrics.append(metrics)
    
    return all_metrics