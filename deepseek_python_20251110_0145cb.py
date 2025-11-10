from PIL import Image
import numpy as np

class LSBSteganography:
    def __init__(self):
        self.encoding_order = 'RGB'  # Порядок каналов для кодирования
        self.pixel_order = 'row'     # Порядок обхода пикселей: 'row' или 'column'
    
    def set_encoding_parameters(self, channel_order='RGB', pixel_order='row'):
        """Установка параметров кодирования"""
        self.encoding_order = channel_order
        self.pixel_order = pixel_order
    
    def _text_to_bits(self, text):
        """Преобразование текста в битовую последовательность"""
        # Сначала кодируем длину текста (4 байта = 32 бита)
        text_length = len(text)
        length_bits = format(text_length, '032b')
        
        # Затем кодируем сам текст в UTF-8
        text_bytes = text.encode('utf-8')
        text_bits = ''.join(format(byte, '08b') for byte in text_bytes)
        
        return length_bits + text_bits
    
    def _bits_to_text(self, bits):
        """Преобразование битовой последовательности в текст"""
        # Извлекаем длину текста (первые 32 бита)
        length_bits = bits[:32]
        text_length = int(length_bits, 2)
        
        # Извлекаем текст
        text_bits = bits[32:32 + text_length * 8]
        
        # Преобразуем биты в байты и затем в текст
        text_bytes = bytearray()
        for i in range(0, len(text_bits), 8):
            byte_bits = text_bits[i:i+8]
            if len(byte_bits) == 8:
                text_bytes.append(int(byte_bits, 2))
        
        return text_bytes.decode('utf-8', errors='ignore')
    
    def _get_pixel_iterator(self, image_array):
        """Генератор для обхода пикселей в заданном порядке"""
        height, width, channels = image_array.shape
        
        if self.pixel_order == 'row':
            # Обход по строкам
            for y in range(height):
                for x in range(width):
                    yield image_array[y, x]
        elif self.pixel_order == 'column':
            # Обход по столбцам
            for x in range(width):
                for y in range(height):
                    yield image_array[y, x]
        else:
            raise ValueError("Неизвестный порядок пикселей")
    
    def _modify_pixel_iterator(self, image_array, new_pixels):
        """Модифицирует пиксели изображения в заданном порядке"""
        height, width, channels = image_array.shape
        new_array = image_array.copy()
        pixel_iter = iter(new_pixels)
        
        if self.pixel_order == 'row':
            for y in range(height):
                for x in range(width):
                    new_array[y, x] = next(pixel_iter)
        elif self.pixel_order == 'column':
            for x in range(width):
                for y in range(height):
                    new_array[y, x] = next(pixel_iter)
        
        return new_array
    
    def _embed_bits_in_pixel(self, pixel, bits, channel_order):
        """Встраивает биты в пиксель"""
        new_pixel = pixel.copy()
        bit_index = 0
        
        for channel_name in channel_order:
            if bit_index >= len(bits):
                break
                
            channel_index = {'R': 0, 'G': 1, 'B': 2}[channel_name]
            # Заменяем младший бит
            new_pixel[channel_index] = (pixel[channel_index] & 0xFE) | int(bits[bit_index])
            bit_index += 1
        
        return new_pixel, bit_index
    
    def _extract_bits_from_pixel(self, pixel, channel_order, num_bits):
        """Извлекает биты из пикселя"""
        bits = []
        
        for channel_name in channel_order:
            if len(bits) >= num_bits:
                break
                
            channel_index = {'R': 0, 'G': 1, 'B': 2}[channel_name]
            # Извлекаем младший бит
            bit = pixel[channel_index] & 1
            bits.append(str(bit))
        
        return ''.join(bits)
    
    def calculate_max_capacity(self, image_path):
        """Рассчитывает максимальную емкость изображения в битах"""
        with Image.open(image_path) as img:
            img_array = np.array(img)
            height, width, channels = img_array.shape
            
            # Каждый пиксель может хранить до 3 бит (по одному на канал)
            max_bits = height * width * len(self.encoding_order)
            
            # Учитываем, что 32 бита занимает длина сообщения
            available_bits = max_bits - 32
            
            return available_bits // 8  # Возвращаем в байтах
    
    def embed_text(self, image_path, text, output_path):
        """Встраивает текст в изображение"""
        
        # Проверяем, помещается ли текст в изображение
        max_capacity = self.calculate_max_capacity(image_path)
        if len(text) > max_capacity:
            raise ValueError(f"Текст слишком длинный. Максимальная длина: {max_capacity} байт")
        
        # Открываем и конвертируем изображение
        with Image.open(image_path) as img:
            if img.mode != 'RGB':
                img = img.convert('RGB')
            image_array = np.array(img)
        
        # Преобразуем текст в биты
        bits = self._text_to_bits(text)
        total_bits = len(bits)
        
        print(f"Встраивание {len(text)} символов ({total_bits} бит)")
        
        # Встраиваем биты в изображение
        modified_pixels = []
        bit_index = 0
        bits_embedded = 0
        
        for pixel in self._get_pixel_iterator(image_array):
            if bit_index >= total_bits:
                # Все биты встроены, просто копируем оставшиеся пиксели
                modified_pixels.append(pixel)
                continue
            
            # Определяем, сколько бит можно встроить в этот пиксель
            bits_to_embed = min(len(self.encoding_order), total_bits - bit_index)
            current_bits = bits[bit_index:bit_index + bits_to_embed]
            
            # Встраиваем биты в пиксель
            new_pixel, bits_used = self._embed_bits_in_pixel(pixel, current_bits, self.encoding_order)
            modified_pixels.append(new_pixel)
            bit_index += bits_used
            bits_embedded += bits_used
        
        # Создаем модифицированное изображение
        modified_array = self._modify_pixel_iterator(image_array, modified_pixels)
        result_image = Image.fromarray(modified_array.astype(np.uint8))
        result_image.save(output_path)
        
        print(f"Успешно встроено {bits_embedded} бит")
        print(f"Изображение сохранено как: {output_path}")
        
        return result_image
    
    def extract_text(self, image_path):
        """Извлекает текст из изображения"""
        
        # Открываем изображение
        with Image.open(image_path) as img:
            if img.mode != 'RGB':
                img = img.convert('RGB')
            image_array = np.array(img)
        
        # Сначала извлекаем длину сообщения (32 бита)
        length_bits = []
        bits_extracted = 0
        
        for pixel in self._get_pixel_iterator(image_array):
            if len(length_bits) >= 32:
                break
            
            extracted = self._extract_bits_from_pixel(pixel, self.encoding_order, 32 - len(length_bits))
            length_bits.append(extracted)
            bits_extracted += len(extracted)
        
        length_bits_str = ''.join(length_bits)[:32]
        text_length = int(length_bits_str, 2)
        
        print(f"Длина сообщения: {text_length} символов")
        
        # Теперь извлекаем сам текст
        total_text_bits = text_length * 8
        text_bits = []
        bits_extracted = 0
        
        # Пропускаем пиксели, использованные для длины
        pixel_iterator = self._get_pixel_iterator(image_array)
        for _ in range((32 + len(self.encoding_order) - 1) // len(self.encoding_order)):
            next(pixel_iterator)
        
        # Извлекаем текст
        for pixel in pixel_iterator:
            if len(text_bits) >= total_text_bits:
                break
            
            extracted = self._extract_bits_from_pixel(pixel, self.encoding_order, 
                                                     total_text_bits - len(text_bits))
            text_bits.append(extracted)
            bits_extracted += len(extracted)
        
        text_bits_str = ''.join(text_bits)[:total_text_bits]
        
        # Преобразуем биты в текст
        extracted_text = self._bits_to_text('0' * 32 + text_bits_str)  # Добавляем фиктивные биты длины
        
        print(f"Извлечено {bits_extracted} бит текста")
        
        return extracted_text

# Пример использования
def main():
    stego = LSBSteganography()
    
    # Настройка параметров кодирования
    stego.set_encoding_parameters(channel_order='RGB', pixel_order='row')
    
    # Пример 1: Встраивание и извлечение текста
    try:
        # Встраивание текста
        input_image = "input.png"  # Замените на путь к вашему изображению
        output_image = "output.png"
        secret_text = "Это секретное сообщение, спрятанное в изображении! 🔐"
        
        print("=== Встраивание текста ===")
        embedded_image = stego.embed_text(input_image, secret_text, output_image)
        
        print("\n=== Извлечение текста ===")
        extracted_text = stego.extract_text(output_image)
        print(f"Извлеченный текст: {extracted_text}")
        
        # Проверка
        if secret_text == extracted_text:
            print("✅ Текст успешно восстановлен!")
        else:
            print("❌ Ошибка при восстановлении текста")
            
    except Exception as e:
        print(f"Ошибка: {e}")
    
    # Пример 2: Проверка емкости
    print("\n=== Проверка емкости ===")
    try:
        capacity = stego.calculate_max_capacity(input_image)
        print(f"Максимальная емкость изображения: {capacity} байт")
    except Exception as e:
        print(f"Ошибка при расчете емкости: {e}")

if __name__ == "__main__":
    main()