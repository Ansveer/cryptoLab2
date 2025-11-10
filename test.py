import struct
import math
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Optional


class LSBSteganography:
    def __init__(self, marker: bytes = b'STEG_END'):
        self.marker = marker
        self.service_bits = 32  # 32 бита для длины сообщения

    def _calculate_capacity(self, image: Image.Image) -> int:
        """Вычисляет максимальную емкость изображения в битах"""
        width, height = image.size
        return width * height * 3  # 3 канала на пиксель

    def _pixel_traversal_order(self, width: int, height: int) -> List[Tuple[int, int, int]]:
        """
        Порядок обхода: построчно слева направо, сверху вниз.
        Для каждого пикселя: каналы в порядке R, G, B.
        """
        order = []
        for y in range(height):
            for x in range(width):
                for channel in range(3):  # R, G, B
                    order.append((x, y, channel))
        return order

    def encode(self, image_path: str, message: str, output_path: str) -> Image.Image:
        """Встраивает сообщение в изображение"""
        # Загрузка изображения
        img = Image.open(image_path).convert('RGB')
        width, height = img.size

        # Преобразование сообщения в байты
        message_bytes = message.encode('utf-8')

        # Расчет необходимых битов
        need_bits = self.service_bits + 8 * len(message_bytes) + len(self.marker) * 8

        # Проверка емкости
        capacity = self._calculate_capacity(img)
        if need_bits > capacity:
            raise ValueError(f"Сообщение слишком большое. Доступно: {capacity} бит, нужно: {need_bits} бит")

        # Создание массива пикселей
        pixels = np.array(img)

        # Получение порядка обхода
        traversal_order = self._pixel_traversal_order(width, height)

        # Подготовка данных для встраивания
        data_to_embed = struct.pack('>I', len(message_bytes)) + message_bytes + self.marker
        bit_array = []
        for byte in data_to_embed:
            for i in range(7, -1, -1):
                bit_array.append((byte >> i) & 1)

        # Встраивание данных
        bit_index = 0
        for x, y, channel in traversal_order:
            if bit_index >= len(bit_array):
                break

            # Замена младшего бита
            old_value = pixels[y, x, channel]
            new_value = (old_value & 0xFE) | bit_array[bit_index]
            pixels[y, x, channel] = new_value
            bit_index += 1

        # Сохранение результата
        result_img = Image.fromarray(pixels)
        result_img.save(output_path)
        return result_img

    def decode(self, image_path: str) -> str:
        """Извлекает сообщение из изображения"""
        # Загрузка изображения
        img = Image.open(image_path).convert('RGB')
        width, height = img.size
        pixels = np.array(img)

        # Получение порядка обхода
        traversal_order = self._pixel_traversal_order(width, height)

        # Извлечение битов
        extracted_bits = []
        for x, y, channel in traversal_order:
            pixel_value = pixels[y, x, channel]
            extracted_bits.append(pixel_value & 1)

        # Преобразование битов в байты
        bytes_list = []
        for i in range(0, len(extracted_bits), 8):
            if i + 8 > len(extracted_bits):
                break
            byte_val = 0
            for j in range(8):
                byte_val = (byte_val << 1) | extracted_bits[i + j]
            bytes_list.append(byte_val)

        # Извлечение длины сообщения
        if len(bytes_list) < 4:
            raise ValueError("Недостаточно данных для извлечения длины сообщения")

        message_length = struct.unpack('>I', bytes(bytes_list[:4]))[0]

        # Извлечение сообщения
        message_start = 4
        message_end = message_start + message_length
        marker_start = message_end

        if len(bytes_list) < marker_start + len(self.marker):
            raise ValueError("Недостаточно данных для извлечения сообщения и маркера")

        message_bytes = bytes(bytes_list[message_start:message_end])
        marker_bytes = bytes(bytes_list[marker_start:marker_start + len(self.marker)])

        # Проверка маркера
        if marker_bytes != self.marker:
            raise ValueError("Маркер конца сообщения не найден")

        return message_bytes.decode('utf-8')


# Пример использования
if __name__ == "__main__":
    # Инициализация
    stego = LSBSteganography()

    # Тестовые данные
    original_image = "checkerboard.png"  # Замените на путь к вашему изображению
    secret_message = "сообщение"

    # Встраивание
    try:
        stego_image = stego.encode(original_image, secret_message, "stego_image.png")
        print("Сообщение успешно встроено")
    except Exception as e:
        print(f"Ошибка при встраивании: {e}")

    # Извлечение
    try:
        extracted_message = stego.decode("stego_image.png")
        print(f"Извлеченное сообщение: {extracted_message}")
    except Exception as e:
        print(f"Ошибка при извлечении: {e}")