import os
import argparse
import math
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

def change_bit(pixel_value, bit):
    return pixel_value & 0xFE | bit

def bytes_to_bits(bytes_data):
    bits = []
    for byte in bytes_data:
        for i in range(7, -1, -1):
            bits.append((byte >> i) & 1)
    return bits

def bits_to_bytes(bits):
    bytes_list = []
    for i in range(0, len(bits), 8):
        byte = 0
        for j in range(8):
            if i + j < len(bits):
                byte = (byte << 1) | bits[i + j]
        bytes_list.append(byte)
    return bytes(bytes_list)

def calculate_psnr(original, stego):
    mse = np.mean((original - stego) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 255.0
    psnr = 20 * math.log10(max_pixel / math.sqrt(mse))
    return psnr

def calculate_ssim(original, stego):
    from scipy.signal import convolve2d
    
    def _ssim_channel(orig_channel, stego_channel):
        C1 = (0.01 * 255) ** 2
        C2 = (0.03 * 255) ** 2
        
        kernel = np.ones((8, 8)) / 64
        mu1 = convolve2d(orig_channel, kernel, mode='valid')
        mu2 = convolve2d(stego_channel, kernel, mode='valid')
        
        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2
        
        sigma1_sq = convolve2d(orig_channel ** 2, kernel, mode='valid') - mu1_sq
        sigma2_sq = convolve2d(stego_channel ** 2, kernel, mode='valid') - mu2_sq
        sigma12 = convolve2d(orig_channel * stego_channel, kernel, mode='valid') - mu1_mu2
        
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        return np.mean(ssim_map)
    
    if len(original.shape) == 3:
        ssim_values = []
        for channel in range(3):
            ssim_values.append(_ssim_channel(original[:, :, channel], stego[:, :, channel]))
        return np.mean(ssim_values)
    else:
        return _ssim_channel(original, stego)

def create_difference_map(original, stego):
    diff = np.abs(original.astype(np.float32) - stego.astype(np.float32))
    if len(diff.shape) == 3:
        diff = np.mean(diff, axis=2)
    return diff.astype(np.uint8)

def plot_histograms(original, stego, channel_names=['Red', 'Green', 'Blue']):
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    
    for i in range(3):
        axes[0, i].hist(original[:, :, i].flatten(), bins=256, alpha=0.7, color='red', label='Original')
        axes[0, i].set_title(f'{channel_names[i]} Channel - Original')
        axes[0, i].set_xlim(0, 255)
        
        axes[1, i].hist(stego[:, :, i].flatten(), bins=256, alpha=0.7, color='blue', label='Stego')
        axes[1, i].set_title(f'{channel_names[i]} Channel - Stego')
        axes[1, i].set_xlim(0, 255)
    
    plt.tight_layout()
    return fig

def rs_analysis(image, group_size=4):
    pixels = image.flatten()
    height, width = image.shape[:2]
    total_pixels = height * width
    
    if len(image.shape) == 3:
        pixels = image.reshape(-1, 3)
        total_pixels = height * width
    
    groups = []
    for i in range(0, total_pixels - group_size + 1, group_size):
        if len(image.shape) == 3:
            groups.append(pixels[i:i+group_size].flatten())
        else:
            groups.append(pixels[i:i+group_size])
    
    def is_regular(group):
        if len(group) < 2:
            return True
        for i in range(1, len(group)):
            if group[i] < group[i-1]:
                return False
        return True
    
    def is_singular(group):
        if len(group) < 2:
            return False
        has_increase = False
        has_decrease = False
        for i in range(1, len(group)):
            if group[i] > group[i-1]:
                has_increase = True
            elif group[i] < group[i-1]:
                has_decrease = True
        return has_increase and has_decrease
    
    regular_count = 0
    singular_count = 0
    
    for group in groups:
        if is_regular(group):
            regular_count += 1
        elif is_singular(group):
            singular_count += 1
    
    total_groups = len(groups)
    r_ratio = regular_count / total_groups
    s_ratio = singular_count / total_groups
    
    return r_ratio, s_ratio, r_ratio - s_ratio

def generate_stego_image(cover_image, payload_percent, message_text=None):
    height, width = cover_image.shape[:2]
    capacity_bits = height * width * 3
    
    service_bits = 8 * 8  # маркер 'klqweofd'
    message_bits = int((payload_percent / 100) * capacity_bits) - service_bits
    message_bits = (message_bits // 8) * 8  # округление до кратного 8
    
    if message_text is None:
        message_bytes = os.urandom(message_bits // 8)
    else:
        text_bytes = message_text.encode('utf-8')
        needed_bytes = message_bits // 8
        if len(text_bytes) >= needed_bytes:
            message_bytes = text_bytes[:needed_bytes]
        else:
            message_bytes = text_bytes + b'\x00' * (needed_bytes - len(text_bytes))
    
    marker = b'klqweofd'
    marker_bits = bytes_to_bits(marker)
    message_bits_list = bytes_to_bits(message_bytes)
    
    encode_message = message_bits_list + marker_bits
    
    stego_pixels = cover_image.copy()
    
    i = 0
    end_embed = False
    for y in range(height):
        for x in range(width):
            for channel in range(3):
                if i >= len(encode_message):
                    end_embed = True
                    break
                old_pixel = stego_pixels[y, x, channel]
                new_pixel = change_bit(old_pixel, encode_message[i])
                stego_pixels[y, x, channel] = new_pixel
                i += 1
            if end_embed:
                break
        if end_embed:
            break
    
    return stego_pixels

def analyze_distortion(cover_path, stego_path, output_dir):
    cover_img = Image.open(cover_path)
    stego_img = Image.open(stego_path)
    
    cover_array = np.array(cover_img)
    stego_array = np.array(stego_img)
    
    # 1) PSNR и SSIM
    psnr_value = calculate_psnr(cover_array, stego_array)
    ssim_value = calculate_ssim(cover_array, stego_array)
    
    print(f"PSNR: {psnr_value:.2f} dB")
    print(f"SSIM: {ssim_value:.4f}")
    
    # 2) Карта разности
    diff_map = create_difference_map(cover_array, stego_array)
    
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.imshow(cover_array)
    plt.title('Original Image')
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.imshow(stego_array)
    plt.title('Stego Image')
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.imshow(diff_map, cmap='hot')
    plt.title('Difference Map')
    plt.colorbar()
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/difference_map.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    # 3) Гистограммы
    hist_fig = plot_histograms(cover_array, stego_array)
    hist_fig.savefig(f"{output_dir}/histograms.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    return psnr_value, ssim_value

def detectability_analysis(cover_path, stego_path):
    cover_img = Image.open(cover_path)
    stego_img = Image.open(stego_path)
    
    cover_array = np.array(cover_img)
    stego_array = np.array(stego_img)
    
    cover_r, cover_s, cover_diff = rs_analysis(cover_array)
    stego_r, stego_s, stego_diff = rs_analysis(stego_array)
    
    print("RS Analysis Results:")
    print(f"Cover Image: R={cover_r:.4f}, S={cover_s:.4f}, R-S={cover_diff:.4f}")
    print(f"Stego Image: R={stego_r:.4f}, S={stego_s:.4f}, R-S={stego_diff:.4f}")
    
    if abs(stego_diff) < abs(cover_diff):
        print("Detection: LIKELY contains hidden message")
    else:
        print("Detection: UNLIKELY to contain hidden message")
    
    return cover_diff, stego_diff

def payload_analysis(cover_path, payload_levels, message_text, output_dir):
    cover_img = Image.open(cover_path)
    cover_array = np.array(cover_img)
    
    results = []
    
    for payload in payload_levels:
        print(f"\nAnalyzing payload: {payload}%")
        
        # Генерация stego-изображения
        stego_array = generate_stego_image(cover_array, payload, message_text)
        
        # Расчет метрик качества
        psnr_value = calculate_psnr(cover_array, stego_array)
        ssim_value = calculate_ssim(cover_array, stego_array)
        
        # RS-анализ
        cover_r, cover_s, cover_diff = rs_analysis(cover_array)
        stego_r, stego_s, stego_diff = rs_analysis(stego_array)
        
        results.append({
            'payload': payload,
            'psnr': psnr_value,
            'ssim': ssim_value,
            'cover_rs_diff': cover_diff,
            'stego_rs_diff': stego_diff,
            'detection_score': abs(stego_diff) - abs(cover_diff)
        })
        
        print(f"  PSNR: {psnr_value:.2f} dB, SSIM: {ssim_value:.4f}")
        print(f"  RS Detection Score: {results[-1]['detection_score']:.4f}")
    
    # Построение графиков
    payloads = [r['payload'] for r in results]
    psnrs = [r['psnr'] for r in results]
    ssims = [r['ssim'] for r in results]
    detection_scores = [r['detection_score'] for r in results]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    ax1.plot(payloads, psnrs, 'bo-', label='PSNR')
    ax1.set_xlabel('Payload (%)')
    ax1.set_ylabel('PSNR (dB)')
    ax1.set_title('Quality vs Payload')
    ax1.grid(True)
    
    ax2_twin = ax2.twinx()
    ax2.plot(payloads, ssims, 'ro-', label='SSIM')
    ax2_twin.plot(payloads, detection_scores, 'go-', label='Detection Score')
    ax2.set_xlabel('Payload (%)')
    ax2.set_ylabel('SSIM')
    ax2_twin.set_ylabel('Detection Score')
    ax2.set_title('SSIM & Detectability vs Payload')
    ax2.legend(loc='upper left')
    ax2_twin.legend(loc='upper right')
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/payload_analysis.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    return results

def roc_analysis(cover_path, stego_images, output_dir):
    # Создание набора тестовых изображений
    cover_img = Image.open(cover_path)
    cover_array = np.array(cover_img)
    
    # Генерация отрицательных примеров (чистые изображения)
    negative_samples = [cover_array]
    
    # Положительные примеры (stego изображения)
    positive_samples = []
    for stego_path in stego_images:
        stego_img = Image.open(stego_path)
        positive_samples.append(np.array(stego_img))
    
    # Расчет RS-метрик для всех изображений
    y_true = []
    y_scores = []
    
    # Отрицательные классы (cover)
    for sample in negative_samples:
        r, s, diff = rs_analysis(sample)
        y_true.append(0)
        y_scores.append(abs(diff))
    
    # Положительные классы (stego)
    for sample in positive_samples:
        r, s, diff = rs_analysis(sample)
        y_true.append(1)
        y_scores.append(abs(diff))
    
    # Расчет ROC-кривой
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    
    # Построение ROC-кривой
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.savefig(f"{output_dir}/roc_curve.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"AUC: {roc_auc:.4f}")
    
    return fpr, tpr, roc_auc


in_img = "../imgs/noise_texture.png"
out_img = "../results/test.png"
output = "../results/"
message = "helloworld"

print("=== DISTORTION ANALYSIS ===")
analyze_distortion(in_img, out_img, output)

print("\n=== DETECTABILITY ANALYSIS ===")
detectability_analysis(in_img, out_img)

print("\n=== PAYLOAD ANALYSIS ===")
payload_levels = [0.1, 0.5, 1.0, 5.0]
payload_results = payload_analysis(in_img, payload_levels, message, output)

# Сохранение результатов
with open(f"{output}/payload_results.txt", 'w') as f:
    f.write("Payload Analysis Results\n")
    f.write("Payload%\tPSNR(dB)\tSSIM\t\tCover RS\tStego RS\tDetection Score\n")
    for result in payload_results:
        f.write(f"{result['payload']:.1f}\t\t{result['psnr']:.2f}\t\t{result['ssim']:.4f}\t"
               f"{result['cover_rs_diff']:.4f}\t\t{result['stego_rs_diff']:.4f}\t\t{result['detection_score']:.4f}\n")