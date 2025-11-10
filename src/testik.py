import argparse
import sys
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import cv2
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

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

def change_bit(pixel_value, bit):
    return pixel_value & 0xFE | bit

def embed_message(cover_path, message_bits, output_path):
    """Встраивание сообщения в изображение"""
    img = Image.open(cover_path)
    width, height = img.size
    pixels = np.array(img)
    
    marker = 'klqweofd'.encode("UTF-8")
    marker_bits = bytes_to_bits(marker)
    encode_message = message_bits + marker_bits
    
    i = 0
    endEmbed = False
    for y in range(height):
        for x in range(width):
            for channel in range(3):
                if i >= len(encode_message):
                    endEmbed = True
                    break
                old_pixel = pixels[y][x][channel]  # исправлено: [y][x] вместо [x][y]
                new_pixel = change_bit(old_pixel, encode_message[i])
                pixels[y][x][channel] = new_pixel
                i += 1
            if endEmbed:
                break
        if endEmbed:
            break
    
    result_img = Image.fromarray(pixels)
    result_img.save(output_path)
    return output_path

def calculate_psnr_ssim(cover_path, stego_path):
    """Расчет PSNR и SSIM между cover и stego изображениями"""
    cover = cv2.imread(cover_path)
    stego = cv2.imread(stego_path)
    
    if cover is None or stego is None:
        raise ValueError("Не удалось загрузить изображения")
    
    # PSNR
    psnr_value = psnr(cover, stego)
    
    # SSIM
    ssim_value = ssim(cover, stego, multichannel=True, channel_axis=2)
    
    return psnr_value, ssim_value

def create_difference_map(cover_path, stego_path, output_dir):
    """Создание карты разности между изображениями"""
    cover = cv2.imread(cover_path)
    stego = cv2.imread(stego_path)
    
    # Разность в абсолютных значениях
    diff = np.abs(cover.astype(float) - stego.astype(float))
    diff_normalized = (diff / diff.max() * 255).astype(np.uint8)
    
    # Серое изображение разности
    diff_gray = cv2.cvtColor(diff_normalized, cv2.COLOR_BGR2GRAY)
    
    # Тепловая карта
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.imshow(cv2.cvtColor(cover, cv2.COLOR_BGR2RGB))
    plt.title('Cover Image')
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.imshow(cv2.cvtColor(stego, cv2.COLOR_BGR2RGB))
    plt.title('Stego Image')
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.imshow(diff_gray, cmap='hot')
    plt.title('Difference Map (Heat)')
    plt.colorbar()
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/difference_map.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    return diff_gray

def plot_histograms(cover_path, stego_path, output_dir):
    """Построение гистограмм каналов до и после"""
    cover = cv2.imread(cover_path)
    stego = cv2.imread(stego_path)
    
    colors = ('b', 'g', 'r')
    plt.figure(figsize=(12, 8))
    
    for i, color in enumerate(colors):
        plt.subplot(2, 3, i + 1)
        plt.hist(cover[:,:,i].ravel(), bins=256, color=color, alpha=0.7)
        plt.title(f'Cover {color.upper()} channel')
        plt.xlim([0, 256])
        
        plt.subplot(2, 3, i + 4)
        plt.hist(stego[:,:,i].ravel(), bins=256, color=color, alpha=0.7)
        plt.title(f'Stego {color.upper()} channel')
        plt.xlim([0, 256])
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/histograms.png", dpi=150, bbox_inches='tight')
    plt.close()

def rs_analysis(image_path, group_size=4):
    """RS-анализ для обнаружения стеганографии"""
    img = Image.open(image_path)
    pixels = np.array(img)
    height, width, channels = pixels.shape
    
    # Функции для классификации групп
    def f(u):
        return (u[0] + u[2]) - (u[1] + u[3])
    
    def g(u):
        return (u[0] + u[2]) - (u[1] + u[3])
    
    # Инвертирование LSB
    def invert_lsb(pixel):
        return pixel ^ 1
    
    regular_groups = 0
    singular_groups = 0
    regular_groups_inv = 0
    singular_groups_inv = 0
    total_groups = 0
    
    # Анализ для каждого канала
    for channel in range(3):
        channel_pixels = pixels[:, :, channel].flatten()
        
        for i in range(0, len(channel_pixels) - group_size + 1, group_size):
            group = channel_pixels[i:i+group_size]
            group_lsb = [p & 1 for p in group]
            
            # Оригинальная группа
            f_val = f(group_lsb)
            if abs(f_val) > 0:
                if f_val > 0:
                    regular_groups += 1
                else:
                    singular_groups += 1
            
            # Инвертированная группа
            group_inv = [invert_lsb(p) for p in group_lsb]
            f_val_inv = f(group_inv)
            if abs(f_val_inv) > 0:
                if f_val_inv > 0:
                    regular_groups_inv += 1
                else:
                    singular_groups_inv += 1
            
            total_groups += 1
    
    # Расчет метрик RS
    if total_groups > 0:
        Rm = regular_groups / total_groups
        Sm = singular_groups / total_groups
        Rm_inv = regular_groups_inv / total_groups
        Sm_inv = singular_groups_inv / total_groups
        
        # Детектор стеганографии
        z = (Rm - Rm_inv) + (Sm - Sm_inv)
        detected = abs(z) > 0.02  # Пороговое значение
        
        return {
            'detected': detected,
            'z_score': z,
            'Rm': Rm,
            'Sm': Sm,
            'Rm_inv': Rm_inv,
            'Sm_inv': Sm_inv,
            'total_groups': total_groups
        }
    
    return None

def generate_payload_images(cover_path, original_message, payload_levels, output_dir):
    """Генерация stego-изображений с разным уровнем payload"""
    # Чтение cover изображения для определения capacity
    img = Image.open(cover_path)
    width, height = img.size
    capacity_bits = width * height * 3
    
    # Преобразование оригинального сообщения в биты
    message_bytes = original_message.encode('utf-8')
    message_bits_full = bytes_to_bits(message_bytes)
    
    results = []
    
    for payload_percent in payload_levels:
        # Расчет размера сообщения
        service_bits = 64  # 8 байт маркера
        message_bits_target = int((payload_percent / 100) * capacity_bits) - service_bits
        message_bits_target = (message_bits_target // 8) * 8  # Округление до кратного 8
        
        # Выбор части сообщения
        if message_bits_target <= len(message_bits_full):
            message_bits = message_bits_full[:message_bits_target]
        else:
            # Если сообщение слишком короткое, дополняем нулями
            message_bits = message_bits_full + [0] * (message_bits_target - len(message_bits_full))
        
        # Создание stego-изображения
        stego_path = f"{output_dir}/stego_{payload_percent}percent.png"
        embed_message(cover_path, message_bits, stego_path)
        
        # Расчет метрик
        psnr_val, ssim_val = calculate_psnr_ssim(cover_path, stego_path)
        rs_result = rs_analysis(stego_path)
        
        results.append({
            'payload_percent': payload_percent,
            'stego_path': stego_path,
            'psnr': psnr_val,
            'ssim': ssim_val,
            'rs_detected': rs_result['detected'] if rs_result else False,
            'rs_z_score': rs_result['z_score'] if rs_result else 0,
            'message_bits': len(message_bits)
        })
    
    return results

def analyze_detector_performance(cover_dir, stego_dir, payload_levels):
    """Анализ производительности детектора и построение ROC-кривой"""
    # Сбор данных для ROC-анализа
    y_true = []  # 0 для cover, 1 для stego
    y_scores = []  # RS z-score
    
    # Анализ cover изображений
    for cover_file in os.listdir(cover_dir):
        if cover_file.endswith(('.png', '.jpg', '.jpeg')):
            cover_path = os.path.join(cover_dir, cover_file)
            rs_result = rs_analysis(cover_path)
            if rs_result:
                y_true.append(0)  # cover
                y_scores.append(abs(rs_result['z_score']))
    
    # Анализ stego изображений для разных уровней payload
    for payload in payload_levels:
        stego_pattern = f"*{payload}percent*"
        for stego_file in os.listdir(stego_dir):
            if stego_file.endswith('.png') and f"{payload}percent" in stego_file:
                stego_path = os.path.join(stego_dir, stego_file)
                rs_result = rs_analysis(stego_path)
                if rs_result:
                    y_true.append(1)  # stego
                    y_scores.append(abs(rs_result['z_score']))
    
    # Построение ROC-кривой
    if len(y_true) > 0 and len(y_scores) > 0:
        fpr, tpr, thresholds = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve for Steganography Detector')
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.savefig("roc_curve.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        return {
            'fpr': fpr,
            'tpr': tpr,
            'auc': roc_auc,
            'thresholds': thresholds
        }
    
    return None

def main():
    parser = argparse.ArgumentParser(description='Steganography Analysis Tool')
    parser.add_argument('--cover', required=True, help='Path to cover image')
    parser.add_argument('--stego', help='Path to stego image (for single analysis)')
    parser.add_argument('--message', required=True, help='Message for payload analysis')
    output = "../results/"
    
    args = parser.parse_args()
    
    print("=== Steganography Analysis ===")
    
    # 1. Анализ искажений (если указано stego изображение)
    if args.stego:
        print("\n1. Analyzing distortions...")
        
        # PSNR и SSIM
        psnr_val, ssim_val = calculate_psnr_ssim(args.cover, args.stego)
        print(f"PSNR: {psnr_val:.2f} dB")
        print(f"SSIM: {ssim_val:.4f}")
        
        # Карта разности
        diff_map = create_difference_map(args.cover, args.stego, output)
        print("Difference map saved")
        
        # Гистограммы
        plot_histograms(args.cover, args.stego, output)
        print("Histograms saved")
        
        # RS-анализ
        rs_result = rs_analysis(args.stego)
        if rs_result:
            print(f"RS Analysis - Detected: {rs_result['detected']}, Z-score: {rs_result['z_score']:.4f}")
    
    # 2. Анализ влияния payload
    print("\n2. Analyzing payload impact...")
    payload_levels = [0.1, 0.5, 1, 5]
    
    payload_results = generate_payload_images(args.cover, args.message, payload_levels, output)
    
    print("\nPayload Analysis Results:")
    print("Payload% | PSNR (dB) | SSIM     | RS Detected | Z-score")
    print("-" * 60)
    for result in payload_results:
        print(f"{result['payload_percent']:8.1f} | {result['psnr']:9.2f} | {result['ssim']:8.4f} | "
              f"{result['rs_detected']:11} | {result['rs_z_score']:8.4f}")
    
    # 3. ROC-анализ
    print("\n3. Performing ROC analysis...")
    roc_results = analyze_detector_performance(
        os.path.dirname(args.cover), 
        output,
        payload_levels
    )
    
    if roc_results:
        print(f"ROC Analysis - AUC: {roc_results['auc']:.4f}")
        print("ROC curve saved as 'roc_curve.png'")
    
    # Сохранение сводного отчета
    with open(f"{args.output_dir}/analysis_report.txt", 'w') as f:
        f.write("Steganography Analysis Report\n")
        f.write("=" * 40 + "\n\n")
        
        if args.stego:
            f.write(f"Single Image Analysis:\n")
            f.write(f"PSNR: {psnr_val:.2f} dB\n")
            f.write(f"SSIM: {ssim_val:.4f}\n")
            if rs_result:
                f.write(f"RS Detection: {rs_result['detected']}\n")
                f.write(f"RS Z-score: {rs_result['z_score']:.4f}\n\n")
        
        f.write("Payload Analysis:\n")
        for result in payload_results:
            f.write(f"Payload {result['payload_percent']}%: "
                   f"PSNR={result['psnr']:.2f}dB, "
                   f"SSIM={result['ssim']:.4f}, "
                   f"Detected={result['rs_detected']}, "
                   f"Z-score={result['rs_z_score']:.4f}\n")
        
        if roc_results:
            f.write(f"\nROC Analysis: AUC = {roc_results['auc']:.4f}\n")
    
    print(f"\nAnalysis complete! Results saved in '{output}'")

if __name__ == "__main__":
    main()