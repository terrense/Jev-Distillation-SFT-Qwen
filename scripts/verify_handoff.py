"""Check public handoff safety and the recovered Word file. Never makes API calls."""
import hashlib
import io
import json
from pathlib import Path
import re
import zipfile
import argparse
import subprocess

root = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument('--staged', action='store_true')
args = parser.parse_args()
patterns = [re.compile(r'(?<![A-Za-z0-9_])sk_[A-Za-z0-9_-]{20,}'), re.compile(r'(?<![A-Za-z0-9_])sk-proj-[A-Za-z0-9_-]{20,}')]
found = []
for p in root.rglob('*'):
    if not p.is_file() or any(x in p.parts for x in ['.git', '__pycache__', '.venv', 'venv']):
        continue
    if p.name.startswith('.env') and p.name != '.env.example':
        found.append(str(p.relative_to(root)))
    if p.suffix in {'.py', '.json', '.jsonl', '.md', '.yaml', '.yml', '.txt', '.bib'}:
        text = p.read_text(encoding='utf-8-sig')
        if any(pattern.search(text) for pattern in patterns):
            found.append(str(p.relative_to(root)))
if found:
    raise SystemExit('Credential-like material found in: ' + ', '.join(found))
receipt = json.loads((root/'audit/restored_binary.json').read_text(encoding='utf-8'))
archive_path = 'archive/originals/CARE_Distill_Research_Package.zip'
archive_bytes = (root/archive_path).read_bytes()
if args.staged or hashlib.sha256(archive_bytes).hexdigest() != receipt['archive_sha256']:
    ref = ':'+archive_path if args.staged else 'HEAD:'+archive_path
    r = subprocess.run(['git','show',ref],cwd=root,capture_output=True)
    if r.returncode:raise SystemExit('Original archive Git object missing; run scripts/restore_docx_index.py.')
    archive_bytes=r.stdout
if hashlib.sha256(archive_bytes).hexdigest()!=receipt['archive_sha256']:
    raise SystemExit('Original archive fingerprint does not match the owner-provided source.')
data = (root/receipt['path']).read_bytes()
source = 'working_copy'
if args.staged or data.startswith(b'%TSD'):
    ref = ':' + receipt['path'] if args.staged else 'HEAD:' + receipt['path']
    result = subprocess.run(['git', 'show', ref], cwd=root, capture_output=True)
    if result.returncode:
        raise SystemExit('Protected working copy; stage the valid original using scripts/restore_docx_index.py first.')
    data = result.stdout
    source = 'staged_git_object' if args.staged else 'committed_git_object_local_copy_protected'
if hashlib.sha256(data).hexdigest() != receipt['sha256']:
    raise SystemExit('Restored DOCX hash mismatch. Restore original archive member; never commit a protected %TSD copy.')
with zipfile.ZipFile(io.BytesIO(data)) as document:
    assert 'word/document.xml' in document.namelist()
print(json.dumps({'credential_scan': 'passed', 'restored_docx': 'valid', 'verified_source': source, 'live_api_calls': 0}))
