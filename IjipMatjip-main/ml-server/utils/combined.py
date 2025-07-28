import pandas as pd
import glob # 파이썬에서 특정 패턴에 맞는 파일들을 한꺼번에 찾을 때 사용.
import os

# 1. 파일이 있는 폴더 경로 지정
path ='./datas/'

# 2. 원하는 csv 파일만 선택하는 패턴을 만듦
file_pattern = os.path.join(path, "encoding_sub*.csv")

# 3. 패턴에 맞는 파일 목록을 가져옴
subway_files = glob.glob(file_pattern)
print(f"지하철 파일 목록 : {subway_files}")

# 4. 찾은 파일들을 모두 읽어와 하나의 데이터프레임으로 합침
df_list =[]
if subway_files:
  for filename in subway_files:
    df = pd.read_csv(filename, encoding='utf-8-sig')
    df_list.append(df)
  
  combined_df = pd.concat(df_list, axis=0, ignore_index=True)
  
  # 5. 합쳐진 데이터 프레임을 새로운 csv 파일로 저장
  output_filename = './datas/subway_combined.csv'
  combined_df.to_csv(output_filename, index=False, encoding='utf-8-sig')
  
  print(f"\n 성공적으로 '{output_filename}'에 저장되었습니다.")
else:
  print("지정한 패턴에 맞는 파일이 없습니다.")


