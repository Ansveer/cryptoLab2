import argparse
import math
import json
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from sklearn.metrics import roc_curve, auc
from collections import Counter
from scipy.integrate import quad
from main import *


def integrand(x, k):
    return math.exp(-(x/2)) * ( x ** ( (k / 2) - 1 ) )


def myChi2(x_2, k):
    tmp = (1 / ( (2 ** (k / 2)) * math.gamma(k/2) )) * quad(integrand, 0, x_2, args=(k))[0]
    return tmp


def myChi2Function(arr, channels, meta=None, img=None):
    p_values = []
    for c in range(channels):
        degrees_of_freedom = 0
        x_2 = 0
        tmp = Counter(arr[:, :, c].flatten())
        channelHist = [tmp.get(j, 0) for j in range(256)]

        for k in range(0, 128, 1):
            n_2k = channelHist[2 * k]
            n_2k1 = channelHist[2 * k + 1]

            e = (n_2k + n_2k1) / 2

            if e > 0:
                x_2 += (((n_2k - e) ** 2) / e) + (((n_2k1 - e) ** 2) / e)
                degrees_of_freedom += 1

        p_value = 1 - myChi2(x_2, degrees_of_freedom - 1)
        p_values.append(p_value)

        if meta:
            meta2 = {
                "img": f"{img}",
                "channel": f"{channel_names[c]}",
                "x_2": f"{x_2}",
                "p_value": f"{p_value}",
            }
            meta["chi2_test"].append(meta2)

    if meta:
        with open(f"../results/Chi2_{img}", "w", encoding="utf-8") as file:
            json.dump(meta, file, indent=3, ensure_ascii=False)

    if not meta:
        return p_values


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--cover', required=True)
    parser.add_argument('--stego', required=True)
    args = parser.parse_args()

    cover = Image.open(f"../imgs/{args.cover}")
    coverArr = np.array(cover)
    stego = Image.open(f"../results/{args.stego}")
    stegoArr = np.array(stego)

    # ========== Оценка незаметности ==========

    # # 1. PSNR и SSIM
    # imgsPSNR = psnr(coverArr, stegoArr)
    # imgsSSIM = ssim(coverArr, stegoArr, channel_axis=2)
    #
    # print(f"PSNR - {imgsPSNR}")
    # print(f"SSIM - {imgsSSIM}")
    #
    # # 2. | cover - stego | и карта разности
    # diffMap = abs(coverArr.astype(np.int16) - stegoArr.astype(np.int16))
    #
    # fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    #
    # axes[0].imshow(cover)
    # axes[0].set_title('Cover изображение')
    # axes[0].axis('off')
    #
    # axes[1].imshow(stego)
    # axes[1].set_title('Stego изображение')
    # axes[1].axis('off')
    #
    # # Тепловая карта разности
    # im1 = axes[2].imshow(diffMap, cmap='hot')
    # axes[2].set_title('Тепловая карта разности')
    # axes[2].axis('off')
    # plt.colorbar(im1, ax=axes[2], fraction=0.05, pad=0.05)
    #
    # fig.savefig(f'../results/DiffMap.png', dpi=300, bbox_inches='tight')
    #
    # # 3. Гистограммы каналов до и после
    channel_names = ['Red', 'Green', 'Blue']
    #
    # fig2, axes2 = plt.subplots(2, 3, figsize=(15, 8))
    #
    # for i in range(3):
    #     axes2[0, i].hist(coverArr[:, :, i].flatten(), bins=256, color=channel_names[i], label='Cover')
    #     axes2[0, i].set_title(f'{channel_names[i]} канал Cover')
    #     axes2[0, i].set_xlim(0, 255)
    #
    #     axes2[1, i].hist(stegoArr[:, :, i].flatten(), bins=256, color=channel_names[i], label='Stego')
    #     axes2[1, i].set_title(f'{channel_names[i]} канал Stego')
    #     axes2[1, i].set_xlim(0, 255)
    #
    # fig2.savefig(f'../results/HistChannels.png', dpi=300, bbox_inches='tight')

    # ========== Проверка обнаружимости ==========

    arr = coverArr
    width, height, channels = arr.shape
    meta = {
        "chi2_test": []
    }

    myChi2Function(arr, channels, meta, "cover")

    arr = stegoArr
    width, height, channels = arr.shape
    meta = {
        "chi2_test": []
    }

    myChi2Function(arr, channels, meta, "stego")

    # ========== Payload ==========

    payloads = [0.1, 0.5, 1, 5, 50, 60, 70, 80]
    capacity_bits = width * height * 3
    marker = 'klqweofd'.encode("UTF-8")
    marker_bits = bytes_to_bits(marker)

    all_scores = []
    all_labels = []

    i = 0
    for p in payloads:
        pixels = np.array(cover)
        width, height, channels = pixels.shape

        message_bits = (( (p / 100) * capacity_bits ) // 8) * 8 - len(marker_bits)
        # print("p - ", p)
        # print("message_bits - ", message_bits)
        message = generate_random_string(int(message_bits) // 8)
        message_bytes = message.encode('utf-8')

        encode_message = bytes_to_bits(message_bytes) + marker_bits
        # print(len(encode_message))

        i = 0
        endEmbed = False
        for y in range(height):
            for x in range(width):
                for channel in range(channels):
                    if i >= len(encode_message):
                        endEmbed = True
                        break
                    old_pixel = pixels[x][y][channel]
                    new_pixel = change_bit(old_pixel, encode_message[i])
                    pixels[x][y][channel] = new_pixel
                    i += 1
                if endEmbed:
                    break
            if endEmbed:
                break

        result_img = Image.fromarray(pixels)

        arr = coverArr
        width, height, channels = arr.shape

        p_values_cover = myChi2Function(arr, channels)
        all_scores.append(np.mean(p_values_cover))
        all_labels.append(1)

        arr = np.array(result_img)
        width, height, channels = arr.shape

        p_values_stego = myChi2Function(arr, channels)
        # tmp = []
        # for i in p_values_stego:
        #     if i > 0.5:
        #         tmp.append(1)
        #     else:
        #         tmp.append(0)
        all_scores.append(int(np.mean(p_values_stego)))
        # all_scores.append(p_values_stego)
        all_labels.append(0)

    fig3 = plt.figure(figsize=(10, 8))
    for i, payload in enumerate(payloads):
        scores = all_scores[i]
        labels = all_labels[i]

        # print(scores)
        # print(labels)

        fpr, tpr, thresholds = roc_curve(labels, scores)
        # print(fpr, tpr, thresholds)
        roc_auc = auc(fpr, tpr)

        plt.plot(fpr, tpr, lw=2, label=f'Payload {payload}% (AUC = {roc_auc:.4f})')

    # print(all_scores)
    # print(all_labels)
    # fpr, tpr, thresholds = roc_curve(all_labels, all_scores)
    # roc_auc = auc(fpr, tpr)
    # plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Случайный детектор (AUC = 0.5)')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.title('ROC-кривые для детектора Хи-квадрат (LSB)')
    plt.legend(loc="lower right")
    plt.grid(True)
    fig3.savefig(f'../results/Payloads.png', dpi=300, bbox_inches='tight')

    plt.show()
