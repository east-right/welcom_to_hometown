import asyncio, os, re, math, json, sys
from dotenv import load_dotenv
load_dotenv(r'C:\VSCode\wh_poc\.env'); os.environ['WH_LLM_BACKEND']='gemini'
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # service_v5 (wh_poc 임포트용)
from wh_poc.tools import collect_places, RERANK_SNIPPETS
from wh_poc.clients import daum_blog, llm_async, embed_texts
from wh_poc.core import load_json, set_verbose
from wh_poc import prompts
set_verbose(False)
KEEP=15
def norm(s): return re.sub(r"[\s\-_()·.,'\"]",'',str(s)).lower()
def toks(s): return set(re.findall(r"[가-힣a-z0-9]+", str(s).lower()))
def cos(a,b):
    d=sum(x*y for x,y in zip(a,b)); na=math.sqrt(sum(x*x for x in a)); nb=math.sqrt(sum(x*x for x in b))
    return d/(na*nb) if na and nb else 0.0
METHODS=['no_rerank','스니펫수','키워드겹침','raw_cos','HyDE_single','HyDE_multi']

async def gather_snips(query, cc, region):
    ctx=type('C',(),{})(); ctx.state={}
    await collect_places(query, ctx, category_code=cc)
    places=load_json(ctx.state['last_places_file'])
    qs=[f"{region} {p['name']}".strip() for p in places]
    raw=await asyncio.gather(*[daum_blog(x,display=10,sort=s) for x in qs for s in ('accuracy','recency')])
    hits={}
    for i,p in enumerate(places):
        m=raw[i*2][0]+raw[i*2+1][0]; seen=set(); t=[]
        for b in m:
            if b['link'] in seen: continue
            seen.add(b['link'])
            if p['name'] in b['title'] or p['name'] in b['description']: t.append(b)
        hits[p['name']]=t[:RERANK_SNIPPETS]
    return [p['name'] for p in places], hits

async def one(case, agg, Nbox):
    q=case['user_query']; answers=case['answer']
    names, hits = await gather_snips(case['query'], case['cc'], case['region'])
    concat={n:" ".join(f"{b['title']} {b['description']}" for b in hits[n]).strip() for n in names}
    have=[n for n in names if concat[n]]
    hypo=(await llm_async(prompts.HYDE_REVIEW.format(user_query=q))).strip() or q
    try:
        facets=[f for f in json.loads(await llm_async(prompts.HYDE_MULTI.format(user_query=q), json_mode=True)) if isinstance(f,str) and f.strip()]
    except Exception: facets=[hypo]
    facets=facets or [hypo]
    vecs=await embed_texts([q, hypo]+facets+[concat[n] for n in have])
    qv,hv=vecs[0],vecs[1]; fv=vecs[2:2+len(facets)]; mv={n:vecs[2+len(facets)+i] for i,n in enumerate(have)}
    qtok=toks(q)
    def sc(m,n):
        if n not in mv: return -1
        if m=='raw_cos': return cos(qv,mv[n])
        if m=='HyDE_single': return cos(hv,mv[n])
        if m=='HyDE_multi': return max(cos(f,mv[n]) for f in fv)
        return 0
    orders={'no_rerank':names,
            '스니펫수':sorted(names,key=lambda n:-len(hits[n])),
            '키워드겹침':sorted(names,key=lambda n:-len(qtok & toks(concat[n])))}
    for m in ('raw_cos','HyDE_single','HyDE_multi'): orders[m]=sorted(names,key=lambda n:-sc(m,n))
    rank={m:{n:i+1 for i,n in enumerate(o)} for m,o in orders.items()}
    line=[]
    for a in answers:
        Nbox['_']=Nbox.get('_',0)+1
        n=next((x for x in names if norm(a) in norm(x) or norm(x) in norm(a)),None)
        if not n: line.append(f"{a}=미수집"); continue
        marks=''
        for m in METHODS:
            ok=rank[m][n]<=KEEP; agg[m]=agg.get(m,0)+ok; marks+='✅' if ok else '·'
        line.append(f"{a}[{marks}]")
    print(f"[{case['question'][:20]:22}] "+" ".join(line))

HERE=os.path.dirname(os.path.abspath(__file__))          # bench_rerank/
V5=os.path.dirname(HERE)                                  # service_v5/

async def main():
    cases=json.load(open(os.path.join(HERE,'inputs.json'),encoding='utf-8'))
    gold={c['question']:c['answer'] for c in json.load(open(os.path.join(V5,'test.json'),encoding='utf-8'))}
    for c in cases: c['answer']=gold.get(c['question'],[])
    agg={}; Nbox={}
    print(f"표기 [{'/'.join(METHODS)}]  ✅=정답 top{KEEP} 진입\n")
    for c in cases:
        try: await one(c, agg, Nbox)
        except Exception as e: print(f"[{c['question']}] ERROR {e}")
    N=Nbox['_']
    print(f"\n==== 정답단위 리랭크 Recall@{KEEP}, N={N} (미수집 포함 분모) ====")
    for m in METHODS: print(f"  {m:12}: {agg.get(m,0):2}/{N} = {agg.get(m,0)/N*100:.1f}%")
asyncio.run(main())
