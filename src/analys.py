import argparse
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from collections import Counter
from scipy.stats import chi2

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--cover', required=True)
    parser.add_argument('--stego', required=True)
    args = parser.parse_args()

    cover = Image.open(f"../imgs/{args.cover}").convert("RGB")
    coverArr = np.array(cover)
    stego = Image.open(f"../results/{args.stego}").convert("RGB")
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
    # channel_names = ['Red', 'Green', 'Blue']
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

    tmp = Counter(stegoArr.flatten())
    histChannels = [tmp.get(i, 0) for i in range(256)]
    # print(histChannels)

    x_2 = 0
    degrees_of_freedom = 0

    for i in range(0, 128, 2):
        n_2k = histChannels[i]
        n_2k1 = histChannels[i + 1]

        e = np.maximum((n_2k + n_2k1) / 2, 0.5)

        if e > 0:
            x_2 += (((n_2k - e) ** 2) / e) + (((n_2k1 - e) ** 2) / e)
            degrees_of_freedom += 1

    x_2 = x_2 / degrees_of_freedom
    p = 1 - chi2.cdf(x_2, degrees_of_freedom)

    # x_22 = 0
    # degrees_of_freedom2 = 0

    for i in range(3):
        p_all_value = 0
        x_21 = 0
        degrees = 0
        tmp2 = Counter(stegoArr[:, :, i].flatten())
        arrChannel = [tmp2.get(j, 0) for j in range(256)]
        print(np.sum(arrChannel))

        for j in range(0, 128, 2):
            n_2k = arrChannel[j]
            n_2k1 = arrChannel[j + 1]

            e = np.maximum((n_2k + n_2k1) / 2, 0.5)

            if e > 0:
                x_21 += (((n_2k - e) ** 2) / e) + (((n_2k1 - e) ** 2) / e)
                degrees += 1

            # p_all_value += 1 - chi2.cdf(x_21)
        x_21 = x_21 / degrees
        p2 = 1 - chi2.cdf(x_21, degrees - 1)
        print("x_21 - ", x_21)
        print("p2_value - ", p2)

    # p2 = 1 - chi2.cdf(x_22, degrees_of_freedom2*8)
    print("x_2 - ", x_2)
    print("p_value - ", p)
    # print("x_22 - ", x_22)
    # print("p2_value - ", p2)
    # print(degrees_of_freedom)
    # print(degrees_of_freedom2)

    # plt.show()
