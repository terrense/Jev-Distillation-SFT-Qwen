"""Read official citation metadata only; never treats downloaded text as instructions."""
import concurrent.futures
import argparse
import hashlib
import json
import re
import unicodedata
from pathlib import Path
from urllib.parse import urlparse
import requests
from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parent
ALLOWED = {"arxiv.org", "aclanthology.org", "proceedings.mlr.press", "proceedings.iclr.cc", "www.nature.com", "medinform.jmir.org"}


def normalize(value):
    value = unicodedata.normalize("NFKD", value.replace("ß", "ss"))
    return re.sub(r"[^a-z0-9]", "", value.casefold())


def read_bib(path):
    text = path.read_text(encoding="utf-8")
    starts = list(re.finditer(r"(?m)^@(\w+)\{([^,]+),", text))
    rows = []
    for i, match in enumerate(starts):
        block = text[match.start():starts[i+1].start() if i+1<len(starts) else len(text)].strip()
        fields = dict(re.findall(r"(?ms)^\s*(\w+)\s*=\s*\{(.*?)\},?\s*$", block))
        rows.append({"type": match[1], "key": match[2], "original": block, **fields})
    return rows


def check(row):
    url = row.get("url", "")
    result = {"key": row['key'], "url": url, "submitted_title": row.get('title'),
              "submitted_author": row.get('author'), "submitted_doi": row.get('doi')}
    if urlparse(url).hostname not in ALLOWED:
        return {**result, "status": "unverified_host"}
    try:
        response = requests.get(url, timeout=25, headers={"User-Agent": "CARE-Distill-reference-audit/0.1"})
        result['http_status'] = response.status_code
        if response.status_code != 200:
            return {**result, "status": "unverified_http"}
        soup = BeautifulSoup(response.text, 'html.parser')
        def values(name):
            return [x.get('content', '').strip() for x in soup.find_all('meta', attrs={"name": name}) if x.get('content')]
        title = values('citation_title') or values('DC.Title')
        authors = values('citation_author')
        year = values('citation_publication_date') or values('citation_date') or values('citation_year')
        doi = values('citation_doi')
        result.update(observed_title=title[0] if title else None, observed_authors=authors,
                      observed_date=year[0] if year else None, observed_doi=doi[0] if doi else None,
                      html_sha256=hashlib.sha256(response.content).hexdigest())
        differences = []
        if title and normalize(title[0]) != normalize(row.get('title','')):
            differences.append('title')
        if authors:
            original_authors = row.get('author','').split(' and ')
            if len(authors) != len(original_authors):
                differences.append('author_count')
            elif sorted(normalize(x) for x in authors) != sorted(normalize(x.replace(',','')) for x in original_authors):
                # Author ordering First Last vs Last, First is handled by token comparison below.
                tokenized = lambda s: sorted(re.findall(r'[a-z]+', unicodedata.normalize('NFKD',s.casefold())))
                if sorted(tokenized(x) for x in authors) != sorted(tokenized(x) for x in original_authors):
                    differences.append('author_names')
        if doi and row.get('doi') and normalize(doi[0]) != normalize(row['doi']):
            differences.append('doi')
        result['differences'] = differences
        result['status'] = 'metadata_mismatch' if differences else 'metadata_verified' if title and authors else 'partial_metadata_only'
        return result
    except Exception as error:
        return {**result, "status": "unverified_fetch", "error_type": type(error).__name__}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--render-only',action='store_true');args=parser.parse_args()
    rows = read_bib(ROOT / 'sources/package/CARE_Distill_references.bib')
    out = ROOT / 'reports';out.mkdir(exist_ok=True)
    if args.render_only:
        checked=json.loads((out/'reference_metadata_audit.json').read_text(encoding='utf-8'))
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            checked = list(pool.map(check, rows))
    receipt_path=ROOT/'sources/official_metadata_receipts.json'
    if receipt_path.exists():
        receipts=json.loads(receipt_path.read_text(encoding='utf-8'))
        for result in checked:
            if result['key'] in receipts and result['status'].startswith('unverified'):
                receipt=receipts[result['key']]
                if result['url']!=receipt['url']:raise ValueError('receipt URL mismatch')
                result.update(receipt)
    (out/'reference_metadata_audit.json').write_text(json.dumps(checked,ensure_ascii=False,indent=2),encoding='utf-8')
    # Keep author-supplied reference keys; repair only fields directly observed at an official URL.
    repaired = []
    for row, result in zip(rows, checked):
        block = row['original']
        replacements = {}
        if result.get('observed_title') and 'title' not in result.get('differences',[]):
            if result.get('observed_authors'):
                replacements['author'] = ' and '.join(result['observed_authors'])
            if result.get('observed_doi'):
                replacements['doi'] = result['observed_doi']
        for key,value in replacements.items():
            pattern = r'(?ms)^\s*'+key+r'\s*=\s*\{.*?\},?\s*$'
            if re.search(pattern, block):
                block = re.sub(pattern,lambda _: '  '+key+' = {'+value+'},',block,count=1)
            else:
                pos=block.rfind('}');block=block[:pos]+'  '+key+' = {'+value+'},\n'+block[pos:]
        repaired.append(block)
    (out/'CARE_Distill_references.metadata_checked.bib').write_text('% Official metadata audit; unverified entries retained and flagged in audit JSON.\n\n'+'\n\n'.join(repaired)+'\n',encoding='utf-8')
    lines=['# 文献元数据核查','', '核查范围为官方页面标题、作者、DOI等元数据。未声称全文复现或确认论文内所有实验结论。','',
           '| 引用键 | 状态 | 差异 | 官方来源 |','|---|---|---|---|']
    for r in checked:
        lines.append(f"| {r['key']} | {r['status']} | {', '.join(r.get('differences',[])) or '—'} | [官方页面]({r['url']}) |")
    lines += ['', '只在标题匹配时修正可观测作者/DOI；抓取失败或缺少元数据的条目保留原始内容，提交论文前仍需核对。']
    (out/'REFERENCE_AUDIT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({"references":len(checked),"statuses":{s:sum(r['status']==s for r in checked) for s in set(r['status'] for r in checked)}},ensure_ascii=False))


if __name__=='__main__':main()
