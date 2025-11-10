import sys
import argparse
import math
import os
import random
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
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

def generate_random_message(num_bits):
    """Генерация случайного сообщения заданной длины в битах"""
    num_bytes = num_bits // 8
    message_bytes = bytes([random.randint(0, 255) for _ in range(num_bytes)])
    return message_bytes

def calculate_psnr(original, stego):
    mse = np.mean((original - stego) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 255.0
    psnr = 20 * math.log10(max_pixel / math.sqrt(mse))
    return psnr

def calculate_ssim(original, stego):
    # Упрощенная версия SSIM для RGB изображений
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2
    
    original = original.astype(np.float64)
    stego = stego.astype(np.float64)
    
    mu_x = np.mean(original)
    mu_y = np.mean(stego)
    
    sigma_x = np.std(original)
    sigma_y = np.std(stego)
    sigma_xy = np.cov(original.flatten(), stego.flatten())[0, 1]
    
    ssim = ((2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)) / \
           ((mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x ** 2 + sigma_y ** 2 + C2))
    
    return ssim


def rs_analysis(image_array, group_size=4):
    """
    RS-анализ для обнаружения LSB-стеганографии
    Возвращает p-value: 0 = нет стего, 1 = полное заполнение
    """
    height, width = image_array.shape[:2]
    
    # Используем только зеленый канал для анализа (обычно самый чувствительный)
    channel_data = image_array[:, :, 1].flatten()
    
    # Функция для инвертирования LSB
    def flip_lsb(pixel):
        return pixel ^ 1
    
    # Функция дискретизации (разности между соседними пикселями в группе)
    def f(group):
        return sum(abs(group[i] - group[i+1]) for i in range(len(group)-1))
    
    # Классификация групп
    def classify_group(group, mask=None):
        if mask:
            masked_group = [flip_lsb(group[i]) if mask[i] else group[i] for i in range(len(group))]
        else:
            masked_group = group.copy()
        
        g0 = f(masked_group)
        flipped_group = [flip_lsb(p) for p in masked_group]
        g1 = f(flipped_group)
        
        if g1 > g0:
            return 'S'  # Singular
        elif g1 < g0:
            return 'R'  # Regular
        else:
            return 'U'  # Unchanged
    
    # Разбиваем на группы
    groups = []
    for i in range(0, len(channel_data) - group_size + 1, group_size):
        group = [(channel_data[i+j] & 1) for j in range(group_size)]  # Берем только LSB
        groups.append(group)
    
    # Подсчет R, S, Rm, Sm
    R = S = Rm = Sm = 0
    
    # Маски для инвертирования
    mask1 = [0, 1, 1, 0]  # Отрицательная маска
    mask2 = [1, 0, 0, 1]  # Положительная маска
    
    for group in groups:
        # Без маски
        class_normal = classify_group(group)
        if class_normal == 'R': R += 1
        elif class_normal == 'S': S += 1
        
        # С отрицательной маской
        class_masked = classify_group(group, mask1)
        if class_masked == 'R': Rm += 1
        elif class_masked == 'S': Sm += 1
    
    total_groups = len(groups)
    
    if total_groups == 0:
        return 0.0
    
    # Нормализуем значения
    r = R / total_groups
    s = S / total_groups
    rm = Rm / total_groups
    sm = Sm / total_groups
    
    # Вычисляем показатель стеганографии
    denominator = (rm - r)
    if abs(denominator) > 0.001:  # Избегаем деления на 0
        p = (s - sm) / denominator
        # Нормализуем в диапазон [0, 1]
        p_normalized = min(max(abs(p), 0), 1)
    else:
        p_normalized = 0.0
    
    return p_normalized

def generate_stego_image(cover_path, message_bits, output_path):
    """Генерация стего-изображения с заданным количеством бит"""
    marker = 'klqweofd'.encode("UTF-8")
    marker_bits = bytes_to_bits(marker)
    
    img = Image.open(cover_path)
    width, height = img.size
    
    # Генерируем случайное сообщение нужной длины
    num_message_bits = message_bits - len(marker_bits)
    num_message_bytes = num_message_bits // 8
    message_bytes = generate_random_message(num_message_bits)
    
    # Преобразуем сообщение в биты
    actual_message_bits = bytes_to_bits(message_bytes)
    
    encode_message = actual_message_bits + marker_bits
    
    pixels = np.array(img)
    
    # Встраивание сообщения
    i = 0
    for y in range(height):
        for x in range(width):
            for channel in range(3):
                if i >= len(encode_message):
                    break
                old_pixel = pixels[y, x, channel]
                new_pixel = change_bit(old_pixel, encode_message[i])
                pixels[y, x, channel] = new_pixel
                i += 1
            if i >= len(encode_message):
                break
        if i >= len(encode_message):
            break
    
    result_img = Image.fromarray(pixels)
    result_img.save(output_path)
    return pixels

def analyze_steganography(cover_path, stego_path, output_dir):
    """Основной анализ стеганографических изображений"""
    
    # Загрузка изображений
    cover_img = Image.open(cover_path)
    stego_img = Image.open(stego_path)
    
    cover_array = np.array(cover_img)
    stego_array = np.array(stego_img)
    
    # 1. Анализ искажений (незаметность)
    print("=== АНАЛИЗ ИСКАЖЕНИЙ ===")
    
    # 1.1 PSNR и SSIM
    psnr_r = calculate_psnr(cover_array[:,:,0], stego_array[:,:,0])
    psnr_g = calculate_psnr(cover_array[:,:,1], stego_array[:,:,1])
    psnr_b = calculate_psnr(cover_array[:,:,2], stego_array[:,:,2])
    psnr_avg = (psnr_r + psnr_g + psnr_b) / 3
    
    ssim_val = calculate_ssim(cover_array, stego_array)
    
    print(f"PSNR (R/G/B/avg): {psnr_r:.2f}/{psnr_g:.2f}/{psnr_b:.2f}/{psnr_avg:.2f} dB")
    print(f"SSIM: {ssim_val:.6f}")
    
    # 1.2 Карта разности
    difference = np.abs(cover_array.astype(float) - stego_array.astype(float))
    difference_gray = np.mean(difference, axis=2)
    
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 3, 1)
    plt.imshow(cover_array)
    plt.title('Cover Image')
    plt.axis('off')
    
    plt.subplot(2, 3, 2)
    plt.imshow(stego_array)
    plt.title('Stego Image')
    plt.axis('off')
    
    plt.subplot(2, 3, 3)
    plt.imshow(difference_gray, cmap='hot')
    plt.title('Difference Map (Heat)')
    plt.colorbar()
    plt.axis('off')
    
    plt.subplot(2, 3, 4)
    plt.imshow(difference_gray, cmap='gray')
    plt.title('Difference Map (Gray)')
    plt.colorbar()
    plt.axis('off')
    
    # 1.3 Гистограммы
    colors = ['red', 'green', 'blue']
    channel_names = ['Red', 'Green', 'Blue']
    
    plt.subplot(2, 3, 5)
    for i, color in enumerate(colors):
        plt.hist(cover_array[:,:,i].flatten(), bins=50, alpha=0.7, 
                color=color, label=f'{channel_names[i]} (cover)')
    plt.title('Cover Image Histograms')
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')
    plt.legend()
    
    plt.subplot(2, 3, 6)
    for i, color in enumerate(colors):
        plt.hist(stego_array[:,:,i].flatten(), bins=50, alpha=0.7, 
                color=color, label=f'{channel_names[i]} (stego)')
    plt.title('Stego Image Histograms')
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/quality_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Проверка обнаружимости (RS-анализ)
    print("\n=== ПРОВЕРКА ОБНАРУЖИМОСТИ (RS-анализ) ===")
    
    # Анализ для cover и stego изображений
    rs_cover = rs_analysis(cover_array)
    rs_stego = rs_analysis(stego_array)
    
    print(f"Cover image RS-value: {rs_cover:.6f}")
    print(f"Stego image RS-value: {rs_stego:.6f}")
    
    # Порог обнаружения (эмпирический)
    threshold = 0.1
    cover_detected = rs_cover > threshold
    stego_detected = rs_stego > threshold
    
    print(f"Cover detected as stego: {'ДА' if cover_detected else 'НЕТ'}")
    print(f"Stego detected as stego: {'ДА' if stego_detected else 'НЕТ'}")
    
    # Визуализация RS-анализа
    plt.figure(figsize=(10, 6))
    images = ['Cover', 'Stego']
    rs_values = [rs_cover, rs_stego]
    colors = ['green' if val < threshold else 'red' for val in rs_values]
    
    bars = plt.bar(images, rs_values, color=colors, alpha=0.7)
    plt.axhline(y=threshold, color='red', linestyle='--', label=f'Detection threshold ({threshold})')
    plt.ylabel('RS-value')
    plt.title('RS-analysis: Steganography Detection')
    plt.legend()
    
    # Добавляем значения на столбцы
    for bar, value in zip(bars, rs_values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{value:.4f}', ha='center', va='bottom')
    
    plt.savefig(f'{output_dir}/rs_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return psnr_avg, ssim_val, (rs_cover, rs_stego)

def payload_analysis(cover_path, output_dir, payloads=[0.1, 0.5, 1.0, 5.0]):
    """Анализ влияния payload на качество и обнаружимость"""
    
    print("\n=== АНАЛИЗ ВЛИЯНИЯ PAYLOAD ===")
    
    cover_img = Image.open(cover_path)
    width, height = cover_img.size
    capacity_bits = width * height * 3
    service_bits = 64  # 8 байт маркера
    
    results = []
    
    for payload in payloads:
        print(f"\nPayload: {payload}%")
        
        # Вычисляем количество бит сообщения
        message_bits = int((payload / 100) * capacity_bits) - service_bits
        message_bits = (message_bits // 8) * 8  # Округление до кратного 8
        
        if message_bits <= 0:
            print(f"Payload {payload}% слишком мал, пропускаем")
            continue
        
        # Генерируем стего-изображение
        stego_path = f"{output_dir}/stego_{payload}percent.png"
        stego_array = generate_stego_image(cover_path, message_bits, stego_path)
        
        # Анализируем
        cover_array = np.array(cover_img)
        psnr_val = calculate_psnr(cover_array, stego_array)
        
        # RS-анализ
        rs_value = rs_analysis(stego_array)
        
        results.append({
            'payload': payload,
            'message_bits': message_bits,
            'psnr': psnr_val,
            'rs_value': rs_value,
            'detected': rs_value > 0.1
        })
        
        print(f"Message bits: {message_bits}")
        print(f"PSNR: {psnr_val:.2f} dB")
        print(f"RS-value: {rs_value:.6f}")
        print(f"Detected: {'ДА' if rs_value > 0.1 else 'НЕТ'}")
    
    # Визуализация результатов
    if results:
        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 2, 1)
        payloads_list = [r['payload'] for r in results]
        psnrs = [r['psnr'] for r in results]
        plt.plot(payloads_list, psnrs, 'bo-', linewidth=2)
        plt.xlabel('Payload (%)')
        plt.ylabel('PSNR (dB)')
        plt.title('Качество vs Payload')
        plt.grid(True)
        
        plt.subplot(2, 2, 2)
        rs_values = [r['rs_value'] for r in results]
        plt.plot(payloads_list, rs_values, 'ro-', linewidth=2)
        plt.axhline(y=0.1, color='red', linestyle='--', alpha=0.7, label='Detection threshold')
        plt.xlabel('Payload (%)')
        plt.ylabel('RS-value')
        plt.title('Обнаружимость vs Payload')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(2, 2, 3)
        detection_status = [1 if r['detected'] else 0 for r in results]
        plt.plot(payloads_list, detection_status, 'go-', linewidth=2, markersize=8)
        plt.xlabel('Payload (%)')
        plt.ylabel('Detection (1=ДА, 0=НЕТ)')
        plt.title('Факт обнаружения vs Payload')
        plt.grid(True)
        
        plt.subplot(2, 2, 4)
        message_bits_list = [r['message_bits'] for r in results]
        plt.plot(message_bits_list, rs_values, 'mo-', linewidth=2)
        plt.axhline(y=0.1, color='red', linestyle='--', alpha=0.7, label='Detection threshold')
        plt.xlabel('Message Bits')
        plt.ylabel('RS-value')
        plt.title('RS-value vs Message Size')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/payload_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    return results

def roc_analysis(cover_path, output_dir, num_samples=50):
    """ROC-анализ для детектора стеганографии"""
    print("\n=== ROC-АНАЛИЗ ===")
    
    cover_img = Image.open(cover_path)
    width, height = cover_img.size
    capacity_bits = width * height * 3
    payload = 1.0  # 1% payload для ROC анализа
    
    # Генерируем набор изображений
    labels = []  # 0 = cover, 1 = stego
    rs_scores = []  # RS-value от анализа
    
    # Cover images (без стеганографии)
    print("Генерация cover изображений...")
    cover_array = np.array(cover_img)
    for i in range(num_samples // 2):
        labels.append(0)
        rs_value = rs_analysis(cover_array)
        rs_scores.append(rs_value)
    
    # Stego images (со стеганографией)
    print("Генерация stego изображений...")
    message_bits = int((payload / 100) * capacity_bits) - 64
    message_bits = (message_bits // 8) * 8
    
    for i in range(num_samples // 2):
        labels.append(1)
        temp_stego_path = f"{output_dir}/temp_stego_{i}.png"
        stego_array = generate_stego_image(cover_path, message_bits, temp_stego_path)
        rs_value = rs_analysis(stego_array)
        rs_scores.append(rs_value)
        
        # Удаляем временный файл
        if os.path.exists(temp_stego_path):
            os.remove(temp_stego_path)
    
    # Вычисляем ROC кривую
    fpr, tpr, thresholds = roc_curve(labels, rs_scores)
    roc_auc = auc(fpr, tpr)
    
    # Визуализация ROC кривой
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve for RS-analysis Steganography Detector')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.savefig(f'{output_dir}/roc_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"AUC: {roc_auc:.4f}")
    
    return roc_auc, fpr, tpr

if __name__ == "__main__":
    in_img = "../imgs/noise_texture.png"
    out_img = "../results/test.png"
    output = "../results/"

    psnr, ssim, rs_results = analyze_steganography(in_img, out_img, output)
    
    # Анализ влияния payload
    print("Анализ влияния payload...")
    payload_results = payload_analysis(in_img, output)
    
    # ROC анализ
    print("ROC анализ...")
    roc_auc, fpr, tpr = roc_analysis(in_img, output)
    
    print(f"\n=== ИТОГИ ===")
    print(f"Все результаты сохранены в директории: {output}")
    print(f"ROC AUC: {roc_auc:.4f}")