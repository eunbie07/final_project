import pandas as pd
import re

def extract_gu(address):
    m = re.search(r'서울특별시\s+(\S+구)', str(address))
    if m:
        return m.group(1)
    return None

def get_infra_score_dict(csv_path='서울교통공사_역주소 및 전화번호_20250318.csv'):
    df = pd.read_csv(csv_path, encoding='euc-kr')
    df['구'] = df['도로명주소'].apply(extract_gu)
    gu_counts = df['구'].value_counts().sort_values(ascending=False)
    infra_score = gu_counts.to_dict()
    return infra_score

# 아래는 테스트용 (직접 실행 시만 동작)
if __name__ == "__main__":
    scores = get_infra_score_dict()
    print(scores)
