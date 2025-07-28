import pandas as pd
from pyproj import Transformer

def transform_coords(x, y, from_epsg, to_epsg='epsg:4326'):
  try:
    transformer = Transformer.from_crs(f"epsg:{from_epsg}", to_epsg, always_xy=True)
    lng, lat = transformer.transform(x, y)
    return lat, lng
  except Exception as e:
    return None, None
  
def main():
  print("좌표 변환을 시작합니다...")
  
  # 병원 데이터 변환
  try:
    hospital_df = pd.read_csv('../datas/hospitals.csv', encoding='utf-8-sig')
    hospital_df[['latitude', 'longitude']] = hospital_df.apply(lambda row: transform_coords(row['경도'], row['위도'], from_epsg=5179), axis=1, result_type='expand')
    hospital_df = hospital_df.rename(columns={'사업장명':'name'})
    hospital_cleaned = hospital_df[['name', 'latitude', 'longitude']].dropna()
    hospital_cleaned.to_csv('../datas/hospitals_cleaned.csv', index=False, encoding='utf-8-sig')
    print("-> 병원 데이터 변환 및 저장 완료")
  except Exception as e:
    print(f"병원 데이터 처리 실패: {e}")
    
  # 마트 데이터 변환
  try:
    mart_df = pd.read_csv('../datas/marts.csv', encoding='utf-8-sig')
    mart_df[['latitude', 'longitude']] = mart_df.apply(lambda row: transform_coords(row['좌표정보x(epsg5174)'], row['좌표정보y(epsg5174)'], from_epsg=5174), axis=1, result_type='expand')
    mart_df = mart_df.rename(columns={'사업장명':'name'})
    mart_cleaned = mart_df[['name', 'latitude', 'longitude']].dropna()
    mart_cleaned.to_csv('../datas/marts_cleaned.csv', index=False, encoding='utf-8-sig')
    print("-> 마트 데이터 변환 및 저장 완료")
  except Exception as e:
    print(f"마트 데이터 처리 실패: {e}")
    
  print("좌표 변환 완료")
  
if __name__  == "__main__":
  main()