"""Stage the owner's unprotected original ZIP member, never decrypt a protected file."""
import hashlib
import json
from pathlib import Path
import subprocess
import zipfile
import argparse
import io

root = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument('--source-zip', help='Optional owner-provided unprotected original archive')
args = parser.parse_args()
receipt = json.loads((root/'audit/restored_binary.json').read_text(encoding='utf-8'))
archive_path = 'archive/originals/CARE_Distill_Research_Package.zip'
choices = [Path(args.source_zip).read_bytes()] if args.source_zip else []
if not args.source_zip:
    for ref in [':'+archive_path, 'HEAD:'+archive_path]:
        r = subprocess.run(['git','show',ref],cwd=root,capture_output=True)
        if r.returncode == 0: choices.append(r.stdout)
    choices.append((root/archive_path).read_bytes())
selected = None
for raw in choices:
    try:
        with zipfile.ZipFile(io.BytesIO(raw)) as archive:
            data = archive.read('CARE_Distill_Research_Manuscript.docx')
        if hashlib.sha256(data).hexdigest() == receipt['sha256'] and hashlib.sha256(raw).hexdigest() == receipt['archive_sha256']:
            selected = (raw,data);break
    except (zipfile.BadZipFile,KeyError):
        pass
if selected is None:
    raise SystemExit('No original archive matching the receipt. Supply the owner-provided original with --source-zip.')
staged=[]
for path,data in [(archive_path,selected[0]),(receipt['path'],selected[1])]:
    result = subprocess.run(['git','hash-object','-w','--stdin'],cwd=root,input=data,capture_output=True,check=True)
    blob=result.stdout.decode().strip()
    subprocess.run(['git','update-index','--add','--cacheinfo','100644',blob,path],cwd=root,check=True)
    staged.append({'path':path,'blob':blob,'bytes':len(data)})
print(json.dumps({'staged_owner_originals':staged}))
