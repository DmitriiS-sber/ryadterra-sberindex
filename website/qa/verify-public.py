from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import subprocess, hashlib, json, re
from html.parser import HTMLParser
from urllib.parse import urljoin, urlsplit

ROOT=Path(__file__).resolve().parent
DIST=ROOT.parent/'dist'
OUT=ROOT/'http-artifacts'
OUT.mkdir(exist_ok=True)
BASE='https://ryadterra.vercel.app/'

def fetch(relative):
    url=urljoin(BASE,relative)
    key=hashlib.sha1(url.encode()).hexdigest()
    body=OUT/(key+'.bin');headers=OUT/(key+'.headers')
    r=subprocess.run(['curl','-q','--silent','--show-error','--location','--max-redirs','5','--max-time','30','--proto','=https','--proto-redir','=https','--dump-header',str(headers),'--output',str(body),'--write-out','%{json}',url],capture_output=True,text=True)
    if r.returncode:return {'path':relative,'url':url,'error':r.stderr.strip(),'curl_exit':r.returncode}
    info=json.loads(r.stdout)
    final=info['url_effective']
    for value in [url,final]:
        assert 'gpt' not in value.lower(),value
        assert not re.search('senyush|сeнюш|dmitr|dmitri|limona|455|5017|сенюш|дмитр',value,re.I),value
        assert urlsplit(value).hostname=='ryadterra.vercel.app',value
    raw=body.read_bytes();expected=(DIST/relative).read_bytes() if relative else (DIST/'index.html').read_bytes()
    return {'path':relative or '/', 'status':info['http_code'],'url':final,'redirects':info['num_redirects'],'bytes':len(raw),'content_type':info.get('content_type'),'tls_verified':info.get('ssl_verify_result')==0,'matches_source':raw==expected,'sha256':hashlib.sha256(raw).hexdigest()}

paths=['']+[p.relative_to(DIST).as_posix() for p in DIST.rglob('*') if p.is_file() and p.name!='index.html']
results=[]
with ThreadPoolExecutor(max_workers=6) as pool:
    for f in as_completed([pool.submit(fetch,x) for x in paths]):results.append(f.result())
results.sort(key=lambda x:x['path'])
failures=[x for x in results if x.get('status')!=200 or not x.get('matches_source') or not x.get('tls_verified')]
report={'checked_at':datetime.now(timezone.utc).isoformat(),'authentication':'curl -q; no Authorization headers, cookies or bypass token','base_url':BASE,'files':len(results),'status':'passed' if not failures else 'failed','results':results}
(OUT/'http-results.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
print(json.dumps({'status':report['status'],'files':len(results),'failures':failures,'base_url':BASE},ensure_ascii=False))
assert not failures,failures
