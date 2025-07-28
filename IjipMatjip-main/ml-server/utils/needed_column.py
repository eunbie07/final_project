import pandas as pd

# CSV 파일 불러오기 (파일 경로에 맞게 수정)
df = pd.read_csv('../datas/park.csv')

# 필요한 컬럼만 추출
columns_needed = [
    '공원구분',
    '공원명',
    '소재지도로명주소',
    '위도',   # 도로명주소 → 도로명전체주소로 보임
    '경도',
]

df_needed = df[columns_needed]

# 번호 컬럼 생성 (1번부터시작하게)
df_needed.insert(0,'번호', range(1, len(df_needed) + 1))

# 결과 확인
print(df_needed.head())

# 필요시 CSV로 저장
df_needed.to_csv('../datas/parks.csv', index=False)
