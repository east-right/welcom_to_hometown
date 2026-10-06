"""리랭커 ablation — 두 질문에 한 번의 라이브 실행으로 답한다.

Q1 (Judge 기여도): HyDE_multi rerank@8 (Judge 없음) vs rerank top15→요약→Judge→top8.
     델타가 0이면 Judge의 지연·비용은 낭비.
Q2 (facet 개수 + max vs mean): 같은 3개 facet 임베딩으로
     - 각 facet 단독(전통/인기/미디어)  → "1개론 왜 부족한가"
     - mean3 vs max3                    → "왜 max인가"
     추가 임베딩/LLM 비용 0 (Q1 요약·스코어만 비용 발생).

주의: Gemini 크레딧 필요 (임베딩+LLM). 크레딧 충전 후 실행.
"""
import asyncio, os, re, math, json, sys, statistics
from dotenv import load_dotenv
load_dotenv(r'C:\VSCode\wh_poc\.env'); os.environ['WH_LLM_BACKEND']='gemini'
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wh_poc.tools import collect_places, RERANK_SNIPPETS, INFO_PER_PLACE, INFO_BODY_CHARS
from wh_poc.clients import daum_blog, llm_async, embed_texts, scrape_urls
from wh_poc.core import FAIL_PREFIX, load_json, set_verbose
from wh_poc import prompts
set_verbose(False)
HERE=os.path.dirname(os.path.abspath(__file__))
KEEP=15; FINAL_K=8; RUNS=int(os.getenv("ABL_RUNS","1"))

def norm(s): return re.sub(r"[\s\-_()·.,'\"]",'',str(s)).lower()
def match(a,names): return next((n for n in names if norm(a) in norm(n) or norm(n) in norm(a)),None)
def cos(a,b):
    d=sum(x*y for x,y in zip(a,b)); na=math.sqrt(sum(x*x for x in a)); nb=math.sqrt(sum(x*x for x in b))
    return d/(na*nb) if na and nb else 0.0
def fresh(): c=type('C',(),{})(); c.state={}; return c

# facet 단독은 개수에 맞춰 동적으로. 3개 기준 라벨.
FACET_LABELS=['f_전통','f_인기','f_미디어']

async def collect(case):
    ctx=fresh(); await collect_places(case['query'], ctx, category_code=case['cc'])
    places=[p['name'] for p in load_json(ctx.state['last_places_file'])]
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
    return places, hits

async def embed_all(places, hits, uq):
    """query + single-HyDE + 3 facets + 매장 concat 벡터를 한 배치로."""
    concat={n:" ".join(f"{b['title']} {b['description']}" for b in hits[n]).strip() for n in places}
    have=[n for n in places if concat[n]]
    single=(await llm_async(prompts.HYDE_REVIEW.format(user_query=uq))).strip() or uq
    try: facets=[f for f in json.loads(await llm_async(prompts.HYDE_MULTI.format(user_query=uq), json_mode=True)) if isinstance(f,str) and f.strip()]
    except Exception: facets=[]
    facets=facets or [single]
    vecs=await embed_texts([uq, single]+facets+[concat[n] for n in have])
    qv, sv = vecs[0], vecs[1]
    fv = vecs[2:2+len(facets)]
    mv = {n: vecs[2+len(facets)+i] for i,n in enumerate(have)}
    return concat, have, qv, sv, fv, mv

def rankers(places, qv, sv, fv, mv):
    """같은 임베딩에서 모든 변형의 순위를 만든다 (추가 비용 0)."""
    def cs(vec,n): return cos(vec,mv[n]) if n in mv else -1
    methods={}
    methods['raw_cos']   = {n: cs(qv,n) for n in places}
    methods['hyde_single']= {n: cs(sv,n) for n in places}
    for i,fvec in enumerate(fv):
        lbl=FACET_LABELS[i] if i<len(FACET_LABELS) else f'f_{i}'
        methods[lbl]={n:(cos(fvec,mv[n]) if n in mv else -1) for n in places}
    methods['multi_mean']= {n:(statistics.mean(cos(f,mv[n]) for f in fv) if n in mv else -1) for n in places}
    methods['multi_max'] = {n:(max(cos(f,mv[n]) for f in fv) if n in mv else -1) for n in places}
    orders={m:sorted(places, key=lambda n:-sc[n]) for m,sc in methods.items()}
    return orders

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

async def judge_top8(summaries, uq):
    if not summaries: return []
    items="\n".join(f"- {n}: {v.get('summary','')} [강점] {' / '.join(v.get('strengths',[]))}" for n,v in summaries.items())
    try: sc=json.loads(await llm_async(prompts.SELECT_SCORE.format(user_query=uq, items=items), json_mode=True))
    except Exception: sc={}
    def s(n):
        x=sc.get(n); return x.get('score',0) if isinstance(x,dict) else (x or 0)
    return sorted(summaries, key=lambda n:-s(n))[:FINAL_K]

RERANK_METHODS=['raw_cos','hyde_single','f_전통','f_인기','f_미디어','multi_mean','multi_max']

async def main():
    cases=json.load(open(os.path.join(HERE,'bench_rerank','inputs.json'),encoding='utf-8'))
    gold={c['question']:c['answer'] for c in json.load(open(os.path.join(HERE,'test.json'),encoding='utf-8'))}
    for c in cases: c['answer']=gold.get(c['question'],[])
    N=sum(len(c['answer']) for c in cases)

    # 누적: 리랭크 @8/@15 = 방법별, 최종 Judge @8
    r8={m:[] for m in RERANK_METHODS}; r15={m:[] for m in RERANK_METHODS}
    base8=[]  # multi_max rerank@8 (Judge 없는 베이스라인)
    judge8=[] # rerank top15 → 요약 → Judge → top8
    for run in range(1,RUNS+1):
        h8={m:0 for m in RERANK_METHODS}; h15={m:0 for m in RERANK_METHODS}; b8=0; j8=0
        for case in cases:
            places, hits = await collect(case)
            concat, have, qv, sv, fv, mv = await embed_all(places, hits, case['user_query'])
            orders=rankers(places, qv, sv, fv, mv)
            rankof={m:{n:i+1 for i,n in enumerate(o)} for m,o in orders.items()}
            ans=case['answer']
            for a in ans:
                for m in RERANK_METHODS:
                    n=match(a, orders[m])
                    if n and rankof[m][n]<=8:  h8[m]+=1
                    if n and rankof[m][n]<=KEEP: h15[m]+=1
                # multi_max rerank@8 베이스라인
                n=match(a, orders['multi_max'])
                if n and rankof['multi_max'][n]<=8: b8+=1
            # Judge 스테이지: production 경로 = multi_max top15 → 요약 → Judge → top8
            summ=await summarize(orders['multi_max'][:KEEP], hits, case['user_query'])
            top8=await judge_top8(summ, case['user_query'])
            for a in ans:
                if match(a, top8): j8+=1
            print(f"  run{run} [{case['question'][:16]:18}] multi_max@8={sum(1 for a in ans if (lambda n:n and rankof['multi_max'][n]<=8)(match(a,orders['multi_max'])))}  judge@8={sum(1 for a in ans if match(a,top8))} /{len(ans)}")
        for m in RERANK_METHODS: r8[m].append(h8[m]); r15[m].append(h15[m])
        base8.append(b8); judge8.append(j8)
        print(f"RUN {run} 완료: multi_max rerank@8={b8}/{N}  Judge@8={j8}/{N}")

    print(f"\n{'='*56}\n리랭크 Recall (정답 {N}개, {RUNS}회 평균)\n{'='*56}")
    print(f"{'method':14} {'@8':>10} {'@15':>10}")
    for m in RERANK_METHODS:
        a8=statistics.mean(r8[m]); a15=statistics.mean(r15[m])
        print(f"{m:14} {a8/N*100:8.1f}% {a15/N*100:8.1f}%")
    b=statistics.mean(base8); j=statistics.mean(judge8)
    print(f"\n{'='*56}\nQ1 Judge 기여도 (top8, 같은 컬렉션)\n{'='*56}")
    print(f"  HyDE_multi rerank@8 (Judge X): {b/N*100:.1f}%  ({b:.1f}/{N})")
    print(f"  rerank15→요약→Judge→top8     : {j/N*100:.1f}%  ({j:.1f}/{N})")
    print(f"  Δ Judge 기여                  : {(j-b)/N*100:+.1f}%p")

asyncio.run(main())
