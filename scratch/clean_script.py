import os

file_path = r"c:\Users\vibha\OneDrive\Desktop\model Experiment\direction2_idd.py"

with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()

# Replace common problematic characters
content = content.replace('×', 'x')
content = content.replace('→', '->')
content = content.replace('—', '-')
content = content.replace('…', '...')

# Strip the decorative lines
# ════════════════════════════════════════════════════════════════════════════
# ────────────────────────────────────────────────────────────────────────────
import re
content = re.sub(r'[═─]{10,}', lambda m: '=' * len(m.group(0)), content)
content = re.sub(r'──', '--', content)

# Remove any remaining non-ASCII characters
def strip_non_ascii(text):
    return "".join(i if ord(i) < 128 else " " for i in text)

content = strip_non_ascii(content)

with open(file_path, 'w', encoding='utf-8') as f:
    f.write(content)

print("Cleanup complete.")
