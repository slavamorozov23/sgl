# from transformers import T5Tokenizer # Больше не нужно
from pprint import pprint
import tokenizer_utils # Импортируем наш новый модуль

# # Загрузка токенизатора # Больше не нужно
# tokenizer = T5Tokenizer.from_pretrained("t5-base")

# Пример текста
text = "Пример текста для токенизации: 2 + 3 = 5\nНовая строка здесь."

# Получаем имя используемого токенизатора (из config.py через tokenizer_utils)
tokenizer_name = tokenizer_utils.get_tokenizer_name()
print(f"Используемый токенизатор (из config): {tokenizer_name}")

# Токенизация текста с использованием функции из tokenizer_utils
print("\nТокенизация текста...")
tokens = tokenizer_utils.tokenize(text)

# Вывод списка токенов
print("\nСписок токенов:")
pprint(tokens, width=1)  # Красивый вертикальный вывод

# Дополнительно: проверим encode и decode
print("\nПроверка encode/decode:")
encoded_ids = tokenizer_utils.encode(text) # По умолчанию add_special_tokens=False
print(f"Encoded IDs: {encoded_ids}")
decoded_text = tokenizer_utils.decode(encoded_ids) # По умолчанию skip_special_tokens=False
print(f"Decoded Text: '{decoded_text}'")

print(f"\nРазмер словаря: {tokenizer_utils.get_vocab_size()}")
print(f"ID паддинг-токена: {tokenizer_utils.get_pad_token_id()}")
print(f"ID EOS-токена: {tokenizer_utils.get_eos_token_id()}")