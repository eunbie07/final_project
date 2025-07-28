import pandas as pd
import json

def create_region_json():
  try:
    # 1. 동네 좌표 읽기
    df = pd.read_csv('../datas/encoding_dong_code.csv', encoding='utf-8-sig')
    
    # 2. 시도/시군구 데이터만 추출 및 중복 제거
    regions_df = df[['시도명','시군구명']].drop_duplicates().sort_values(by=['시도명','시군구명'])
    
    # 3. JSON 구조로 데이터 가공
    region_data = {}
    for index, row in regions_df.iterrows():
      sido = row['시도명']
      sigungu = row['시군구명']
      if sido not in region_data:
        region_data[sido] = []
      # 시군구 이름 있고, 리스트에 없는 경우에만 추가
      if sigungu and sigungu not in region_data[sido]:
        region_data[sido].append(sigungu)
        
    # 4. JSON 파일로 저장
    output_path = '../../frontend/src/data/regions.json' # frontend 폴더에 넣음
    with open(output_path, 'w', encoding='utf-8-sig') as f:
      json.dump(region_data, f, ensure_ascii=False, indent=4)
    print(f"지역 데이터가 '{output_path}'에 성공적으로 생성되었습니다.")
    
  except Exception as e:
    print(f"오류 발생: {e}")

if __name__ == '__main__':
  create_region_json()