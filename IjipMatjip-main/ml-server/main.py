import os
import psycopg2
import psycopg2.extras
import time
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict
from dotenv import load_dotenv

from infrastructure_analyzer import InfrastructureAnalyzer
# model_utils는 더 이상 사용하지 않으므로 import 문을 제거하거나 주석 처리합니다.
# from model_utils import train_and_save_model 

app = FastAPI()
load_dotenv()

# 전역 변수 선언
loaded_models = {}
infra_analyzer = None
neighborhood_scores_df = None
trade_df = None


def get_db_connection():
  conn = psycopg2.connect(
    host=os.getenv("DB_HOST"),
    port=os.getenv("DB_PORT"),
    dbname=os.getenv("DB_NAME"),
    user=os.getenv("DB_USER"),
    password=os.getenv("DB_PASSWORD")
  )
  return conn

@app.on_event("startup")
def startup_event():
  global infra_analyzer, neighborhood_scores_df,trade_df
  infra_analyzer = InfrastructureAnalyzer()
# Pydantic 모델 정의
class HouseInfo(BaseModel):
  region_code: str
  area: float
  floor: int
  build_year: int

class InfraQuery(BaseModel):
  latitude: float
  longitude: float
  radius_km: float

class Budget(BaseModel):
  min: int | None = None
  max: int | None = None

class SizePyeong(BaseModel):
  min: int | None = None
  max: int | None = None

class RecommendationRequest(BaseModel):
    preferences: List[str]
    budget: Budget | None = None
    size_pyeong: SizePyeong | None = None
    dong: str | None = None
    region: str | None = None # 희망 지역 필드 추가
# --- API 엔드포인트들 ---
@app.post("/predict")
def predict_price(info:HouseInfo):
  model_filename = f"real_estate_model_{info.region_code}.pkl"
  if model_filename in loaded_models:
    model = loaded_models[model_filename]
  else:
    if not os.path.exists(model_filename):
      success = train_and_save_model(info.region_code)
    if not success:
      raise HTTPException(status_code=404, detail=f"{info.region_code} 지역의 데이터를 찾을 수 없거나 모델을 생성할 수 없습니다.")
    model = joblib.load(model_filename)
    loaded_models[model_filename] = model
    print(f"'{model_filename}' 모델을 로드했습니다.")
  
  current_year = time.localtime().tm_year
  age = current_year - info.build_year
  input_data = pd.DataFrame([[info.area, info.floor, age]], columns=['area','floor','age'])
  prediction = model.predict(input_data)
  return {"predicted_price": prediction[0]}

@app.post("/infrastructure")
def get_nearby_infrastructure(query: InfraQuery):
    if not infra_analyzer:
        raise HTTPException(status_code=503, detail="인프라 분석기가 아직 준비되지 않았습니다.")
    
    # 👇 5가지 인프라를 모두 검색하여 반환하도록 수정
    return {
        "schools": infra_analyzer.find_nearby(query.latitude, query.longitude, query.radius_km, 'school'),
        "subways": infra_analyzer.find_nearby(query.latitude, query.longitude, query.radius_km, 'subway'),
        "hospitals": infra_analyzer.find_nearby(query.latitude, query.longitude, query.radius_km, 'hospital'),
        "marts": infra_analyzer.find_nearby(query.latitude, query.longitude, query.radius_km, 'mart'),
        "parks": infra_analyzer.find_nearby(query.latitude, query.longitude, query.radius_km, 'park')
    }

@app.post("/recommend/neighborhood")
def recommend_neighborhood_and_properties(request: RecommendationRequest):
    conn = get_db_connection()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
            # 1. 최적의 동네 추천 (상위 5개)
            score_clauses = [f"{pref}_score" for pref in request.preferences if f"{pref}_score" in ['school_score', 'subway_score', 'price_score']]
            total_score_sql = " + ".join(score_clauses) if score_clauses else "school_score + subway_score + price_score"
            
            where_clauses_dong = ["latitude IS NOT NULL AND longitude IS NOT NULL"]
            params_dong = []
            if request.region:
                where_clauses_dong.append("sigungu_name = %s")
                params_dong.append(request.region)
            if request.budget and request.budget.min is not None:
                where_clauses_dong.append("avg_price >= %s")
                params_dong.append(request.budget.min)
            if request.budget and request.budget.max is not None:
                where_clauses_dong.append("avg_price <= %s")
                params_dong.append(request.budget.max)
            
            where_sql_dong = "WHERE " + " AND ".join(where_clauses_dong)
            dong_query = f"SELECT *, ({total_score_sql}) AS total_score FROM neighborhood_scores {where_sql_dong} ORDER BY total_score DESC LIMIT 5;"
            cur.execute(dong_query, tuple(params_dong))
            recommended_dongs = [dict(row) for row in cur.fetchall()]

            # 2. 추천된 동네 내에서 매물 검색
            properties = []
            recommended_dong_names = [d['dong'] for d in recommended_dongs]
            if recommended_dong_names:
                where_clauses_prop = ["dong_name IN %s"]
                params_prop = [tuple(recommended_dong_names)]
                if request.budget and request.budget.min is not None:
                    where_clauses_prop.append("price >= %s")
                    params_prop.append(request.budget.min)
                if request.budget and request.budget.max is not None:
                    where_clauses_prop.append("price <= %s")
                    params_prop.append(request.budget.max)
                if request.size_pyeong and request.size_pyeong.min is not None:
                    where_clauses_prop.append("area >= %s")
                    params_prop.append(request.size_pyeong.min * 3.305785)
                if request.size_pyeong and request.size_pyeong.max is not None:
                    where_clauses_prop.append("area <= %s")
                    params_prop.append(request.size_pyeong.max * 3.305785)
                
                where_sql_prop = "WHERE " + " AND ".join(where_clauses_prop)
                prop_query = f"SELECT * FROM transactions {where_sql_prop} LIMIT 10;"
                cur.execute(prop_query, tuple(params_prop))
                
                raw_properties = [dict(row) for row in cur.fetchall()]
                for prop in raw_properties:
                    properties.append({
                        "name": prop.get('complex_name'), "dong_name": prop.get('dong_name'),
                        "price": prop.get('price'), "predicted_price": prop.get('price'),
                        "size_m2": prop.get('area'), "floor": prop.get('floor'),
                        "build_year": prop.get('build_year'), "latitude": prop.get('latitude'),
                        "longitude": prop.get('longitude')
                    })
            
            return {"neighborhoods": recommended_dongs, "properties": properties}
    except Exception as e:
        print(f"추천 API 처리 중 오류 발생: {e}")
        raise HTTPException(status_code=500, detail="추천 정보를 가져오는 데 실패했습니다.")
    finally:
        if conn:
            conn.close()
