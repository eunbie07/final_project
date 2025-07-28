import os
import requests
import pandas as pd
from io import StringIO
import time
import joblib
from sklearn.ensemble import RandomForestRegressor
from dotenv import load_dotenv

# 함수가 region_code를 인자로 받도록 수정
def train_and_save_model(region_code: str):
  print(f"{region_code} 지역의 데이터 로딩 및 모델 학습을 시작합니다.")
  load_dotenv()
  api_key = os.getenv("API_KEY")
  
  if not api_key:
    print("오류 : API_KEY를 찾을 수 없습니다.")
    return False 
  url = "https://apis.data.go.kr/1613000/RTMSDataSvcAptTradeDev/getRTMSDataSvcAptTradeDev"
  params = {
    "serviceKey": api_key,
    "LAWD_CD" : region_code, 
    "DEAL_YMD": "202306",
    "numOfRows": "200"
  }
  response = requests.get(url, params=params)
  
  if response.status_code == 200:
    # 데이터가 없는 경우 확인하는 코드
    if '<totalCount>0</totalCount>' in response.text:
      print(f"오류: {region_code} 지역, {params['DEAL_YMD']} 날짜에 데이터가 없습니다.")
      return False
    df = pd.read_xml(StringIO(response.text), xpath=".//item")
    if df.empty:
      print(f"오류: {region_code} 지역에 데이터가 없습니다.")
      return False
    
    print (f"len(df)개의 데이터를 수집했습니다.")
    
    df_processed = df[['dealAmount','buildYear', 'excluUseAr', 'floor']].copy()
    df_processed.columns = ['price','build_year','area','floor']
    
    df_processed['price'] = df_processed['price'].str.replace(' ','').str.replace(',','').astype(int)
    df_processed['area'] = pd.to_numeric(df_processed['area'])
    df_processed['floor'] = pd.to_numeric(df_processed['floor'])
    df_processed['build_year'] = pd.to_numeric(df_processed['build_year'])
    current_year = time.localtime().tm_year
    df_processed['age'] = current_year - df_processed['build_year']
    
    features = ['area', 'floor', 'age']
    target = 'price'
    X_train = df_processed[features]
    y_train = df_processed[target]
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train,y_train)
    
    #모델 파일 이름에 region_code 포함
    model_filename = f'real_estate_model_{region_code}.pkl'
    joblib.dump(model, model_filename)
    print(f"모델 학습 완료! '{model_filename}' 파일로 저장되었습니다.")
    return True 
  else:
    print(f"API 호출 실패! 응답 코드 : {response.status.status_code}")
    return False
        
