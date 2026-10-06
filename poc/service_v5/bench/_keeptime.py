import asyncio, os, time, json, sys
from dotenv import load_dotenv
load_dotenv(r'C:\VSCode\wh_poc\.env'); os.environ['WH_LLM_BACKEND']='gemini'
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import wh_poc.tools as T
from wh_poc.tools import collect_places, collect_info
from wh_poc.core import set_verbose
set_verbose(False)
HERE=os.path.dirname(os.path.abspath(__file__))
KEEPS=[15,20]

def fresh():
    c=type('C',(),{})(); c.state={}; return c

async def main():
    cases=json.load(open(os.path.join(HERE,'bench_rerank','inputs.json'),encoding='utf-8'))
    # 후보 많은 대표 질문
    targets=[c for c in cases if c['question']=='창원 가포 맛집 추천']
    results={K:[] for K in KEEPS}
    for case in targets:
        ctx=fresh(); await collect_places(case['query'], ctx, category_code=case['cc'])
        pf=ctx.state['last_places_file']
        print(f"\n[{case['question']}]")
        for K in KEEPS:
            T.RERANK_KEEP=K
            c2=fresh(); c2.state['last_places_file']=pf
            t0=time.perf_counter()
            await collect_info(c2, user_query=case['user_query'], region_prefix=case['region'])
            dt=time.perf_counter()-t0
            results[K].append(dt)
            print(f"   KEEP={K:2}: {dt:5.1f}s")
    print("\n==== collect_info 평균 시간 (KEEP별) ====")
    base=sum(results[KEEPS[0]])/len(results[KEEPS[0]])
    for K in KEEPS:
        avg=sum(results[K])/len(results[K])
        print(f"  KEEP={K:2}: 평균 {avg:5.1f}s  (KEEP=10 대비 +{avg-base:4.1f}s, x{avg/base:.2f})")
asyncio.run(main())
