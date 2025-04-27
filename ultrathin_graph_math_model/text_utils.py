import re
from typing import Union, List, Tuple, Optional


def clean_text(text: Union[str, bytes],
               prefix_pattern: Optional[str] = r"^b['\"](.*?)['\"]$",
               prefix_flags: int = re.DOTALL,
               escape_replacements: Optional[List[Tuple[str, str]]] = None,
               remove_real_linebreaks: bool = True,
               strip_whitespace: bool = True,
               lowercase: bool = False,
               custom_replacements: Optional[List[Tuple[str, str]]] = None
              ) -> str:
    """
    Clean text by applying several optional steps:
    1. Decode bytes to string using UTF-8 or Latin-1.
    2. Remove a prefix/suffix matching prefix_pattern.
    3. Replace escape sequences (e.g., literal "\\n", "\\r").
    4. Optionally remove real newlines and carriage returns.
    5. Apply additional custom replacements.
    6. Strip leading/trailing whitespace and optionally lowercase result.
    """
    # Decode bytes
    if isinstance(text, bytes):
        for enc in ('utf-8', 'latin-1'):
            try:
                text = text.decode(enc)
                break
            except UnicodeDecodeError:
                continue
    else:
        text = str(text)

    # Remove prefix pattern
    if prefix_pattern:
        match = re.match(prefix_pattern, text, prefix_flags)
        if match:
            text = match.group(1)

    # Default escape replacements
    if escape_replacements is None:
        escape_replacements = [('\\n', ''), ('\\r', '')]
    for old, new in escape_replacements:
        text = text.replace(old, new)

    # Custom replacements
    if custom_replacements:
        for old, new in custom_replacements:
            text = text.replace(old, new)

    # Remove real linebreaks
    if remove_real_linebreaks:
        text = text.replace('\n', '').replace('\r', '')

    # Strip whitespace
    if strip_whitespace:
        text = text.strip()

    # Lowercase
    if lowercase:
        text = text.lower()

    return text