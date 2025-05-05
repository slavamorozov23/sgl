# ultrathin_graph_math_model/tokenizer_utils.py
import os
import re
from transformers import AutoTokenizer, PreTrainedTokenizerFast
from rich.console import Console
import config  # Импортируем ваш конфиг для доступа к TOKENIZER_NAME

console = Console()

_tokenizer = None
_tokenizer_name = None

# Глобальные переменные для кэширования результатов проверки
_space_encodes_non_empty = None
_newline_encodes_non_empty = None


def _initialize_tokenizer():
    """
    Инициализирует синглтон токенизатора на основе config.TOKENIZER_NAME.
    Автоматически настраивает pad_token, если он отсутствует.
    """
    global _tokenizer, _tokenizer_name
    if _tokenizer is not None and _tokenizer_name == config.TOKENIZER_NAME:
        return _tokenizer # Уже инициализирован с тем же именем

    _tokenizer_name = config.TOKENIZER_NAME
    # console.log(f"Initializing tokenizer: {_tokenizer_name}...")
    try:
        _tokenizer = AutoTokenizer.from_pretrained(_tokenizer_name, use_fast=True)

        # Настройка pad_token, если отсутствует
        if _tokenizer.pad_token_id is None:
            if _tokenizer.eos_token_id is not None:
                console.log(f"[yellow]Tokenizer '{_tokenizer_name}' lacks pad token, using EOS token (ID: {_tokenizer.eos_token_id}) as pad token.[/yellow]")
                _tokenizer.pad_token = _tokenizer.eos_token
                # _tokenizer.pad_token_id = _tokenizer.eos_token_id # AutoTokenizer обычно делает это сам при установке pad_token
            else:
                # Если нет ни pad, ни eos, добавляем новый pad токен (менее предпочтительно)
                console.log(f"[yellow]Tokenizer '{_tokenizer_name}' lacks both pad and EOS tokens. Adding a new pad token '[PAD]'.[/yellow]")
                _tokenizer.add_special_tokens({'pad_token': '[PAD]'})
                # Важно: После добавления токена может потребоваться model.resize_token_embeddings(len(tokenizer))
                # Это нужно будет учесть при рефакторинге модели. Пока оставляем так.

        # console.log(f"Tokenizer '{_tokenizer_name}' initialized. Vocab size: {_tokenizer.vocab_size}, Pad token ID: {_tokenizer.pad_token_id}")

    except Exception as e:
        console.print(f"[bold red]Error initializing tokenizer '{_tokenizer_name}': {e}[/]")
        _tokenizer = None # Сбрасываем в случае ошибки
        raise # Перевыбрасываем исключение

    # Проверяем кодирование пробела и новой строки после инициализации
    if _tokenizer:
        check_space_encoding()
        check_newline_encoding()

    return _tokenizer

def check_space_encoding():
    """Проверяет, кодируется ли пробел в непустой список токенов."""
    global _space_encodes_non_empty
    if _space_encodes_non_empty is None:
        try:
            # Используем внутренний вызов encode, чтобы избежать рекурсии при первой проверке
            encoded = get_tokenizer().encode(' ', add_special_tokens=False)
            _space_encodes_non_empty = bool(encoded)
        except Exception as e:
            console.print(f"[ERROR tokenizer_utils] Failed to check space encoding: {e}")
            _space_encodes_non_empty = True # Предполагаем, что кодируется, чтобы избежать ненужных замен при ошибке
    return _space_encodes_non_empty

def check_newline_encoding():
    """Проверяет, кодируется ли новая строка в непустой список токенов."""
    global _newline_encodes_non_empty
    if _newline_encodes_non_empty is None:
        try:
            # Используем внутренний вызов encode
            encoded = get_tokenizer().encode('\n', add_special_tokens=False)
            _newline_encodes_non_empty = bool(encoded)
        except Exception as e:
            console.print(f"[ERROR tokenizer_utils] Failed to check newline encoding: {e}")
            _newline_encodes_non_empty = True # Предполагаем, что кодируется
    return _newline_encodes_non_empty

def prepare_text_for_encoding(text: str) -> str:
    """Подготавливает текст к кодированию, заменяя пробелы и новые строки при необходимости."""
    # Убедимся, что проверки были выполнены хотя бы раз
    check_space_encoding()
    # check_newline_encoding() # Больше не нужна проверка для \n, так как всегда заменяем
    # Выполняем замену только если проверка показала, что символ не кодируется
    if not _space_encodes_non_empty:
        text = text.replace(' ', '_')
    # --- ИЗМЕНЕНО: Всегда заменяем \n на _ ---
    text = text.replace('\n', '_')
    # --- КОНЕЦ ИЗМЕНЕНИЯ ---
    return text

def get_tokenizer() -> PreTrainedTokenizerFast:
    """
    Возвращает инициализированный синглтон экземпляр токенизатора.
    """
    if _tokenizer is None:
        _initialize_tokenizer()
    if _tokenizer is None:
         raise RuntimeError("Tokenizer could not be initialized.") # Добавим проверку
    return _tokenizer

# --- Функции-обертки ---

def tokenize(text: str, **kwargs) -> list[str]:
    """Токенизирует текст."""
    return get_tokenizer().tokenize(text, **kwargs)

def encode(text: str, **kwargs) -> list[int]:
    """Кодирует текст в список ID токенов."""
    # Убедимся, что add_special_tokens=False по умолчанию, если не указано иное,
    # так как многие части кода полагались на это.
    kwargs.setdefault('add_special_tokens', False)
    # --- ДОБАВИТЬ ВЫЗОВ prepare_text_for_encoding ---
    prepared_text = prepare_text_for_encoding(text)
    # --- КОНЕЦ ИЗМЕНЕНИЯ ---
    # --- УБРАН ЛОГ ИЗ ПРЕДЫДУЩЕГО ШАГА ---
    # tokenizer_instance = get_tokenizer()
    # print(f"[DIAG tokenizer_utils.encode] Input text: {repr(text)}, kwargs: {kwargs}") # Лог 1
    # result = tokenizer_instance.encode(text, **kwargs)
    # print(f"[DIAG tokenizer_utils.encode] Output IDs: {result}") # Лог 2
    # --- КОНЕЦ УДАЛЕНИЯ ЛОГА ---
    return get_tokenizer().encode(prepared_text, **kwargs) # Используем prepared_text

def decode(token_ids, **kwargs) -> str:
    """Декодирует список ID токенов в текст."""
    # Убедимся, что skip_special_tokens=False по умолчанию, если не указано иное.
    kwargs.setdefault('skip_special_tokens', False)
    return get_tokenizer().decode(token_ids, **kwargs)

def get_vocab_size() -> int:
    """Возвращает размер словаря токенизатора."""
    return get_tokenizer().vocab_size

def get_pad_token_id() -> int | None:
    """Возвращает ID паддинг-токена."""
    return get_tokenizer().pad_token_id

def get_eos_token_id() -> int | None:
    """Возвращает ID EOS-токена."""
    return get_tokenizer().eos_token_id

def get_user_token_id() -> int | None:
    """Возвращает ID токена пользователя (если определен в config)."""
    tok = get_tokenizer()
    user_token = getattr(config, 'USER_TOKEN', None)
    if user_token:
        try:
            user_id = tok.convert_tokens_to_ids(user_token)
            if user_id == tok.unk_token_id:
                 console.print(f"[bold red]Error: USER_TOKEN '{user_token}' maps to UNK token in tokenizer vocabulary.[/]")
                 return None # Возвращаем None, если токен UNK
            return user_id
        except KeyError:
            console.print(f"[bold red]Error: USER_TOKEN '{user_token}' not found in tokenizer vocabulary.[/]")
            return None # Возвращаем None при ошибке
    return None

def get_assistant_token_id() -> int | None:
    """Возвращает ID токена ассистента (если определен в config)."""
    tok = get_tokenizer()
    assistant_token = getattr(config, 'ASSISTANT_TOKEN', None)
    if assistant_token:
        try:
            assistant_id = tok.convert_tokens_to_ids(assistant_token)
            if assistant_id == tok.unk_token_id:
                 console.print(f"[bold red]Error: ASSISTANT_TOKEN '{assistant_token}' maps to UNK token in tokenizer vocabulary.[/]")
                 return None # Возвращаем None, если токен UNK
            return assistant_id
        except KeyError:
            console.print(f"[bold red]Error: ASSISTANT_TOKEN '{assistant_token}' not found in tokenizer vocabulary.[/]")
            return None # Возвращаем None при ошибке
    return None

def get_token_id(token_str: str) -> int:
    """Конвертирует строку токена в ID."""
    return get_tokenizer().convert_tokens_to_ids(token_str)

def get_tokenizer_name() -> str:
    """Возвращает имя используемого токенизатора."""
    get_tokenizer() # Убедимся, что инициализирован
    return _tokenizer_name

def get_safe_tokenizer_name() -> str:
    """Возвращает имя токенизатора, безопасное для использования в именах файлов."""
    name = get_tokenizer_name()
    return re.sub(r'[\\/*?:"<>|]', '_', name)

# Можно добавить другие обертки по мере необходимости
