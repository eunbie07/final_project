from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from data_utils import get_infra_score_dict

app = FastAPI()

# CORS 설정 (모든 도메인 허용, 개발용)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 애플리케이션 구동 시점에 점수 딕셔너리 메모리에 적재
infra_score = get_infra_score_dict('서울교통공사_역주소 및 전화번호_20250318.csv')

@app.get("/infra-score")
def get_score(region: str):
    score = infra_score.get(region)
    if score is None:
        return {"region": region, "score": 0, "msg": "해당 지역 데이터 없음"}
    return {"region": region, "score": score}
