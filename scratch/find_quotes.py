import os
content = open(r'c:\Users\vibha\OneDrive\Desktop\model Experiment\direction2_idd.py', encoding='utf-8').read()
for i, line in enumerate(content.split('\n')):
    if '"""' in line:
        print(f"Line {i+1}: {line.strip()}")
