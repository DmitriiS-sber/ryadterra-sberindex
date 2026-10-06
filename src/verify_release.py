"""Regression tests for revision selection and as-of non-anticipation."""
from pathlib import Path
import json
import pandas as pd
from asof import asof_panel
ROOT = Path(__file__).resolve().parents[1]
def main():
    cutoff = '2026-04-02T00:00:00Z'
    def row(value, published, retrieved, vintage):
        return dict(territory_id=1, observation_month='2026-01', value=value,
                    published_at=published, retrieved_at=retrieved, vintage_id=vintage)
    old = row(100., '2026-02-01T00:00:00Z', '2026-04-01T00:00:00Z', 'v1')
    new = row(200., '2026-03-01T00:00:00Z', '2026-03-02T00:00:00Z', 'v2')
    frame = pd.DataFrame([old, new]); checks = []
    def check(name, condition):
        if not condition: raise AssertionError(name)
        checks.append(dict(name=name, status='passed'))
    def rejected(name, source):
        try: asof_panel(source, cutoff)
        except ValueError: check(name, True)
        else: check(name, False)
    base = asof_panel(frame, cutoff)
    check('Late retrieval of an old revision cannot replace a newer revision', base.iloc[0,0] == 200.)
    check('Input row order does not affect revision selection', base.equals(asof_panel(frame.iloc[::-1], cutoff)))
    future = pd.DataFrame([row(999., '2026-04-03T00:00:00Z', '2026-04-04T00:00:00Z','v3')])
    check('Future publication does not alter an as-of panel', base.equals(asof_panel(pd.concat([frame,future]),cutoff)))
    late = pd.DataFrame([row(888.,'2026-03-15T00:00:00Z','2026-04-05T00:00:00Z','v4')])
    check('Unretrieved revision cannot enter an as-of panel',base.equals(asof_panel(pd.concat([frame,late]),cutoff)))
    rejected('Ambiguous publication ordering fails closed',pd.concat([frame,pd.DataFrame([dict(new,value=300.,vintage_id='v5')])]))
    unknown=frame.copy();unknown.loc[0,'published_at']=None
    rejected('Unknown publication time is rejected',unknown)
    rejected('Duplicate key within vintage is rejected',pd.concat([frame,frame.iloc[:1]]))
    backwards=frame.copy();backwards.loc[0,'retrieved_at']='2026-01-15T00:00:00Z'
    rejected('Retrieval before publication is rejected',backwards)
    from prospective import code_hashes, LOCK
    if LOCK.exists():
        check('Release code matches its prospective lock',json.loads(LOCK.read_text())['code_sha256']==code_hashes())
    output=ROOT/'results/release_verification.json'
    output.write_text(json.dumps(dict(status='passed',checks=checks,scope='Synthetic revision-registry regression tests; not evidence of real publication history.'),ensure_ascii=False,indent=2))
    print(json.dumps(dict(status='passed',checks=len(checks),output=str(output))))
if __name__=='__main__':main()
