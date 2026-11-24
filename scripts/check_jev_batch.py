"""Fail a manual job on dependency/auth/balance errors; do not reveal credentials."""
import argparse
import json
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('--input', required=True)
a = p.parse_args()
rows = [json.loads(s) for s in Path(a.input).read_text(encoding='utf-8').splitlines() if s.strip()]
errors = [r.get('error') for r in rows if r.get('error')]
print(json.dumps({'records': len(rows), 'successful': sum(not r.get('error') for r in rows), 'error_codes': sorted(set(errors))}))
if errors or not rows:
    raise SystemExit(1)
