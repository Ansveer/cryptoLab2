import sys
import argparse
from PIL import Image
import numpy as np


def change_bit(pixel_value, bit):
    return pixel_value & 0xFE | bit


def bytes_to_bits(bytes):
    bits = []
    for byte in bytes:
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--text', required=True)
    parser.add_argument('--in', dest="_in", required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()

    marker = 'klqweofd'.encode("UTF-8")
    marker_bits = bytes_to_bits(marker)
    message = args.text
    message_bytes = message.encode('utf-8')

    img = Image.open(f"../imgs/{args._in}")
    width, height = img.size

    need_bits = 8 * len(marker) + 8 * len(message_bytes)
    if need_bits > width * height * 3:
        print("Сообщение слишком большое")
        sys.exit(0)

    pixels = np.array(img)
    encode_message = bytes_to_bits(message_bytes) + marker_bits

    # Встраивание сообщения в изображение
    i = 0
    endEmbed = False
    for y in range(height):
        for x in range(width):
            for channel in range(3):
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
    result_img.save(f"../results/{args.out}")

    pixels2 = np.array(result_img)

    # Достаём сообщение
    markerFound = False
    extractedBits = []
    for y in range(height):
        for x in range(width):
            for channel in range(3):
                bit = pixels2[x][y][channel] & 1
                extractedBits.append(bit)

                if len(extractedBits) >= len(marker_bits):
                    recentBits = extractedBits[-len(marker_bits):]
                    if recentBits == marker_bits:
                        markerFound = True
                        break
            if markerFound:
                break
        if markerFound:
            break

    if not markerFound:
        print("Маркер не найден")
        sys.exit(0)

    extractedBits = extractedBits[:-len(marker_bits)]

    extractedBytes = bits_to_bytes(extractedBits)

    messageExtracted = extractedBytes.decode('utf-8')

    print(f"Восстановленный текст - {messageExtracted}")
