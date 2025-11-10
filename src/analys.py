import sys
import argparse
import math
import os
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
        return sum(abs(group[i] - group[i + 1]) for i in range(len(group) - 1))

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
        group = [(channel_data[i + j] & 1) for j in range(group_size)]  # Берем только LSB
        groups.append(group)

    # Подсчет R, S, Rm, Sm
    R = S = Rm = Sm = 0

    # Маски для инвертирования
    mask1 = [0, 1, 1, 0]  # Отрицательная маска
    mask2 = [1, 0, 0, 1]  # Положительная маска

    for group in groups:
        # Без маски
        class_normal = classify_group(group)
        if class_normal == 'R':
            R += 1
        elif class_normal == 'S':
            S += 1

        # С отрицательной маской
        class_masked = classify_group(group, mask1)
        if class_masked == 'R':
            Rm += 1
        elif class_masked == 'S':
            Sm += 1

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


# def chi_square_test(image_array, channel=0):
#     """χ²-тест для обнаружения стеганографии в LSB"""
#     height, width = image_array.shape[:2]
#
#     # Собираем гистограмму для выбранного канала
#     hist = np.zeros(256, dtype=int)
#     for y in range(height):
#         for x in range(width):
#             pixel_value = image_array[y, x, channel]
#             hist[pixel_value] += 1
#
#     # Группируем в пары (2k, 2k+1)
#     chi_square = 0
#     degrees_of_freedom = 128
#
#     for k in range(128):
#         observed_2k = hist[2*k]
#         observed_2k1 = hist[2*k+1]
#
#         if observed_2k + observed_2k1 > 0:
#             expected = (observed_2k + observed_2k1) / 2
#             if expected > 0:
#                 chi_square += ((observed_2k - expected) ** 2) / expected
#                 chi_square += ((observed_2k1 - expected) ** 2) / expected
#
#     # Вычисляем p-value
#     p_value = 1 - stats.chi2.cdf(chi_square, degrees_of_freedom)
#
#     return chi_square, p_value, hist


def generate_stego_image(cover_path, message_bits, output_path):
    """Генерация стего-изображения с заданным количеством бит"""
    marker = 'klqweofd'.encode("UTF-8")
    marker_bits = bytes_to_bits(marker)
    
    img = Image.open(cover_path)
    width, height = img.size
    
    # Преобразуем сообщение в биты
    message_bytes = b'A' * (len(message_bits) // 8)  # Произвольные данные
    actual_message_bits = bytes_to_bits(message_bytes)[:len(message_bits)]
    
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
    
    # 2. Проверка обнаружимости (χ²-тест)
    print("\n=== ПРОВЕРКА ОБНАРУЖИМОСТИ (χ²-тест) ===")
    
    chi_results = []
    for channel in range(3):
        chi_square, p_value, hist = rs_analysis(stego_array, channel)
        chi_results.append((chi_square, p_value))
        print(f"Channel {channel}: χ² = {chi_square:.2f}, p-value = {p_value:.6f}")
        
        # Визуализация пар 2k/2k+1
        plt.figure(figsize=(10, 6))
        x_values = list(range(0, 256, 2))
        pairs_2k = [hist[i] for i in range(0, 256, 2)]
        pairs_2k1 = [hist[i+1] for i in range(0, 256, 2)]
        
        plt.bar([x-0.4 for x in x_values], pairs_2k, width=0.4, 
               label='Even values (2k)', alpha=0.7)
        plt.bar([x for x in x_values], pairs_2k1, width=0.4, 
               label='Odd values (2k+1)', alpha=0.7)
        plt.xlabel('Pixel Value')
        plt.ylabel('Frequency')
        plt.title(f'χ²-test for Channel {channel} (p-value: {p_value:.6f})')
        plt.legend()
        plt.savefig(f'{output_dir}/chi2_channel_{channel}.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    return psnr_avg, ssim_val, chi_results

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
        
        # Генерируем стего-изображение
        stego_path = f"{output_dir}/stego_{payload}percent.png"
        stego_array = generate_stego_image(cover_path, str(message_bits), stego_path)
        
        # Анализируем
        cover_array = np.array(cover_img)
        psnr_val = calculate_psnr(cover_array, stego_array)
        
        # χ²-тест для красного канала
        chi_square, p_value, _ = rs_analysis(stego_array, 0)
        
        results.append({
            'payload': payload,
            'message_bits': message_bits,
            'psnr': psnr_val,
            'chi_square': chi_square,
            'p_value': p_value
        })
        
        print(f"Message bits: {message_bits}")
        print(f"PSNR: {psnr_val:.2f} dB")
        print(f"χ²: {chi_square:.2f}, p-value: {p_value:.6f}")
    
    # Визуализация результатов
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
    chi_squares = [r['chi_square'] for r in results]
    plt.plot(payloads_list, chi_squares, 'ro-', linewidth=2)
    plt.xlabel('Payload (%)')
    plt.ylabel('χ² statistic')
    plt.title('Обнаружимость vs Payload')
    plt.grid(True)
    
    plt.subplot(2, 2, 3)
    p_values = [r['p_value'] for r in results]
    plt.semilogy(payloads_list, p_values, 'go-', linewidth=2)
    plt.xlabel('Payload (%)')
    plt.ylabel('p-value (log scale)')
    plt.title('p-value vs Payload')
    plt.grid(True)
    
    plt.subplot(2, 2, 4)
    message_bits_list = [r['message_bits'] for r in results]
    plt.plot(message_bits_list, p_values, 'mo-', linewidth=2)
    plt.xlabel('Message Bits')
    plt.ylabel('p-value')
    plt.title('p-value vs Message Size')
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
    p_values = []  # p-value от χ²-теста
    
    # Cover images (без стеганографии)
    for i in range(num_samples // 2):
        labels.append(0)
        cover_array = np.array(cover_img)
        _, p_value, _ = rs_analysis(cover_array, 0)
        p_values.append(p_value)
    
    # Stego images (со стеганографией)
    message_bits = int((payload / 100) * capacity_bits) - 64
    message_bits = (message_bits // 8) * 8
    
    for i in range(num_samples // 2):
        labels.append(1)
        temp_stego_path = f"{output_dir}/temp_stego_{i}.png"
        stego_array = generate_stego_image(cover_path, str(message_bits), temp_stego_path)
        _, p_value, _ = rs_analysis(stego_array, 0)
        p_values.append(p_value)
        
        # Удаляем временный файл
        if os.path.exists(temp_stego_path):
            os.remove(temp_stego_path)
    
    # Вычисляем ROC кривую
    # Используем 1-p_value как score для детектора (чем выше, тем более вероятно stego)
    scores = [1 - p for p in p_values]
    
    fpr, tpr, thresholds = roc_curve(labels, scores)
    roc_auc = auc(fpr, tpr)
    
    # Визуализация ROC кривой
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve for Steganography Detector')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.savefig(f'{output_dir}/roc_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"AUC: {roc_auc:.4f}")
    
    return roc_auc, fpr, tpr


if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description='Steganography Analysis Tool')
    # parser.add_argument('--cover', required=True, help='Path to cover image')
    # parser.add_argument('--stego', help='Path to stego image (optional)')
    # parser.add_argument('--output', required=True, help='Output directory for results')
    # in_img = "../imgs/checkerboard.png"
    # in_img = "../imgs/gradient.png"
    in_img = "../imgs/noise_texture.png"
    out_img = "../results/test.png"
    output = "../results/"
    
    # Анализ конкретного стего-изображения если предоставлен
    psnr, ssim, chi_results = analyze_steganography(in_img, out_img, output)
    
    # Анализ влияния payload
    payload_results = payload_analysis(in_img, output)
    
    # ROC анализ
    roc_auc, fpr, tpr = roc_analysis(in_img, output)
    
    print(f"\n=== ИТОГИ ===")
    print(f"Все результаты сохранены в директории: {output}")
    print(f"ROC AUC: {roc_auc:.4f}")