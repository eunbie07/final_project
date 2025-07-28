import chardet
import pandas as pd

file_path = '../datas/parksd.csv'

# 파일의 인코딩 감지
with open(file_path, 'rb') as f: # #파일을 바이너리 모드(rb)로 열어야 합니다
  raw_data = f.read()
  result = chardet.detect(raw_data)
  detected_encoding = result['encoding']
  confidence = result['confidence']
  
# print(f"인코딩 : {detected_encoding}, 신뢰도: {confidence:.2f}")

try:
  df = pd.read_csv(file_path, encoding=detected_encoding)
  print('csv 파일 읽기 성공')
  # print(df.head()) # 잘 나오는 것을 확인했으니 지웁니다.
except UnicodeDecodeError:
  print(f"인코딩 '{detected_encoding}'으로 파이을 읽는 데 실패했습니다. 다른 인코딩을 시도합니다.")
  try:
    df = pd.read_csv(file_path, encoding='utf-8') # 인코딩 감지된게 cp949였으니 다른걸로 다시 시도
    print('csv 파일 (utf-8) 읽기 성공')
    # print(df.head()) #잘 나오는 것을 확인했으니 지웁니다.
  except Exception as e:
    print(f"utf-8로 파일을 읽는데 실패했습니다 : {e}")
    
# 깨지지 않은 DataFrame을 새 CSV 파일로 저장합니다.
output_file_path = '../datas/park.csv'

# encoding='utf-8-sig'로 저장하는 것이 핵심!
df.to_csv(output_file_path, index=False, encoding='utf-8-sig')

print(f"CSV 파일이 '{output_file_path}' 에 성공적으로 저장되었습니다.")