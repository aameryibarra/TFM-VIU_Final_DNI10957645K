import json, sys

with open('Codigo-TFM_Ariel_Mery_Ibarra.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

def merge_split_strings(source_lines):
    """Merge consecutive source array elements that form broken string literals."""
    result = []
    i = 0
    while i < len(source_lines):
        line = source_lines[i]
        stripped = line.rstrip('\n')
        # Detect an unterminated print( opening: ends with f" or just "
        if (stripped == 'print(f"' or stripped == 'print("') and i + 1 < len(source_lines):
            next_line = source_lines[i + 1]
            # Replace with: print() on its own line + print("content")
            content_line = next_line  # e.g. 'Fraude en train...") \n' or '=== COMPARATIVA ===")\n'
            result.append('print()\n')
            result.append('print(' + stripped[6:] + content_line)  # strip 'print(' prefix, keep f" or "
            i += 2
            continue
        result.append(line)
        i += 1
    return result

# Fix cells 42, 62, 63
for cell_idx in [42, 62, 63]:
    original = nb['cells'][cell_idx]['source']
    fixed = merge_split_strings(original)
    nb['cells'][cell_idx]['source'] = fixed

    src_check = ''.join(nb['cells'][cell_idx]['source'])
    try:
        compile(src_check, f'cell_{cell_idx}', 'exec')
        print(f'Cell [{cell_idx}] OK')
    except SyntaxError as e:
        print(f'Cell [{cell_idx}] STILL BROKEN line {e.lineno}: {e}')
        lines = src_check.split('\n')
        for j in range(max(0, e.lineno-2), min(len(lines), e.lineno+3)):
            print(f'  {j+1}: {repr(lines[j])}')

# Final check: all code cells
print('\n--- Final syntax check ---')
errors = 0
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] != 'code':
        continue
    src = ''.join(cell['source'])
    # Skip magic commands (not valid Python but fine in Jupyter)
    src_no_magic = '\n'.join(
        l if not l.strip().startswith('%') else f'pass  # magic: {l.strip()}'
        for l in src.split('\n')
    )
    try:
        compile(src_no_magic, f'cell_{i}', 'exec')
    except SyntaxError as e:
        print(f'  Cell [{i}] line {e.lineno}: {e}')
        errors += 1

if errors == 0:
    print('  All code cells syntax-clean')

with open('Codigo-TFM_Ariel_Mery_Ibarra.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print('\nNotebook saved.')
