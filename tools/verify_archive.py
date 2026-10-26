"""Verify archived research bytes without modifying or executing the study."""
from pathlib import Path
import hashlib
import sys

root = Path(__file__).resolve().parents[1] / 'research'
failed = []
count = 0
for line in (root / 'MANIFEST_SHA256.txt').read_text(encoding='utf-8').splitlines():
    expected, relative = line.split('  ', 1)
    path = root / relative
    if not path.resolve().is_relative_to(root.resolve()):
        failed.append(relative + ': invalid manifest path')
        continue
    if not path.is_file():
        failed.append(relative + ': missing')
    elif hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        failed.append(relative + ': hash mismatch')
    count += 1
if failed:
    print('\n'.join(failed))
    sys.exit(1)
print(f'PASS: {count} archived research files match the SHA-256 manifest.')
