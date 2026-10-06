import asyncio, os, re, json, math, statistics, sys
from dotenv import load_dotenv
load_dotenv(r'C:\VSCode\wh_poc\.env'); os.environ['WH_LLM_BACKEND']='gemini'
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wh_poc.tools import collect_places, RERANK_SNIPPETS, INFO_PER_PLACE, INFO_BODY_CHARS
from wh_poc.clients import daum_blog, llm_async, embed_texts, scrape_urls
from wh_poc.core import FAIL_PREFIX, load_json, set_verbose
from wh_poc import prompts
set_verbose(False)
HERE=os.path.dirname(os.path.abspath(__file__))
KEEPS=[15,20]; FINAL_K=8; RUNS=5
def norm(s): return re.sub(r"[\s\-_()·.,'\"]",'',str(s)).lower()
def match(a,names): return next((n for n in names if norm(a) in norm(n) or norm(n) in norm(a)),None)
def cos(a,b):
    d=sum(x*y for x,y in zip(a,b)); na=math.sqrt(sum(x*x for x in a)); nb=math.sqrt(sum(x*x for x in b))
    return d/(na*nb) if na and nb else 0.0
def fresh(): c=type('C',(),{})(); c.state={}; return c

async def collect(case):
    ctx=fresh(); await collect_places(case['query'], ctx, category_code=case['cc'])
    places=[p['name'] for p in load_json(ctx.state['last_places_file'])]
    meta={p['name']:p for p in load_json(ctx.state['last_places_file'])}
    qs=[f"{case['region']} {n}".strip() for n in places]
    raw=await asyncio.gather(*[daum_blog(x,display=10,sort=s) for x in qs for s in ('accuracy','recency')])
    hits={}
    for i,n in enumerate(places):
        m=raw[i*2][0]+raw[i*2+1][0]; seen=set(); t=[]
        for b in m:
            if b['link'] in seen: continue
            seen.add(b['link'])
            if n in b['title'] or n in b['description']: t.append(b)
        hits[n]=t[:RERANK_SNIPPETS]
    return places, hits, meta

async def rerank(places, hits, uq):
    concat={p:" ".join(f"{b['title']} {b['description']}" for b in hits[p]).strip() for p in places}
    have=[p for p in places if concat[p]]
    try: facets=[f for f in json.loads(await llm_async(prompts.HYDE_MULTI.format(user_query=uq), json_mode=True)) if isinstance(f,str) and f.strip()]
    except Exception: facets=[]
    if not facets: facets=[(await llm_async(prompts.HYDE_REVIEW.format(user_query=uq))).strip() or uq]
    vecs=await embed_texts(facets+[concat[p] for p in have])
    fv=vecs[:len(facets)]; mv={p:vecs[len(facets)+i] for i,p in enumerate(have)}
    score={p:(max(cos(f,mv[p]) for f in fv) if p in mv else -1) for p in places}
    return sorted(places, key=lambda p:-score[p])

async def summarize(kept, hits, uq):
    links,owner=[],[]
    for p in kept:
        for b in hits[p][:INFO_PER_PLACE]:
            links.append(b['link']); owner.append(p)
    scraped=await scrape_urls(links)
    bodies={p:[] for p in kept}
    for name,s in zip(owner,scraped):
        if s['text'].startswith(FAIL_PREFIX) or s.get('sponsored'): continue
        bodies[name].append(s['text'])
    async def one(name,texts):
        joined="\n\n---\n\n".join(f"[후기 {i+1}]\n{t[:INFO_BODY_CHARS]}" for i,t in enumerate(texts))
        p=prompts.SUMMARIZE_INFO.format(user_query=uq,place_name=name,n=len(texts))+"\n"+joined
        try: return json.loads(await llm_async(p,json_mode=True))
        except Exception: return {"strengths":[],"summary":""}
    names=[p for p in kept if bodies[p]]
    rs=await asyncio.gather(*[one(n,bodies[n]) for n in names])
    return {n:r for n,r in zip(names,rs)}

async def score_top8(summaries, uq):
    if not summaries: return []
    items="\n".join(f"- {n}: {v.get('summary','')} [강점] {' / '.join(v.get('strengths',[]))}" for n,v in summaries.items())
    try: sc=json.loads(await llm_async(prompts.SELECT_SCORE.format(user_query=uq, items=items), json_mode=True))
    except Exception: sc={}
    def s(n):
        x=sc.get(n); return x.get('score',0) if isinstance(x,dict) else (x or 0)
    return sorted(summaries, key=lambda n:-s(n))[:FINAL_K]

async def main():
    cases=json.load(open(os.path.join(HERE,'bench_rerank','inputs.json'),encoding='utf-8'))
    gold={c['question']:c['answer'] for c in json.load(open(os.path.join(HERE,'test.json'),encoding='utf-8'))}
    N=sum(len(gold.get(c['question'],[])) for c in cases)
    rr={K:[] for K in KEEPS}; fin={K:[] for K in KEEPS}
    for run in range(1,RUNS+1):
        rr_hit={K:0 for K in KEEPS}; fin_hit={K:0 for K in KEEPS}
        for case in cases:
            places, hits, meta = await collect(case)
            ranked = await rerank(places, hits, case['user_query'])
            rankof={n:i+1 for i,n in enumerate(ranked)}
            ans=gold.get(case['question'],[])
            for K in KEEPS:
                # 리랭크 recall @K (같은 순위, 컷만)
                for a in ans:
                    n=match(a,ranked)
                    if n and rankof[n]<=K: rr_hit[K]+=1
                # 최종 recall: top K 크롤·요약 → 스코어 → top8
                summ=await summarize(ranked[:K], hits, case['user_query'])
                top8=await score_top8(summ, case['user_query'])
                for a in ans:
                    if match(a, top8): fin_hit[K]+=1
        for K in KEEPS:
            rr[K].append(rr_hit[K]); fin[K].append(fin_hit[K])
        print(f"RUN {run}: " + " | ".join(f"K{K} 리랭크 {rr_hit[K]}/{N}={rr_hit[K]/N*100:.0f}% 최종 {fin_hit[K]}/{N}={fin_hit[K]/N*100:.0f}%" for K in KEEPS))
    print(f"\n==== 평균 (정답 {N}개, {RUNS}회) ====")
    for K in KEEPS:
        rm=statistics.mean(rr[K]); fm=statistics.mean(fin[K])
        print(f"  KEEP={K}: 리랭크 {rm/N*100:.1f}% ({rm:.1f}/{N})  |  최종 {fm/N*100:.1f}% ({fm:.1f}/{N})")
asyncio.run(main())
