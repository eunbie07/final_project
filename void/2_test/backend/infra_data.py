def get_infra_score(region):
    # 실제론 공공데이터 API 크롤링/DB 등에서 가져오기
    # 여기선 예시로 하드코딩
    sample_data = {
        "강남구": {"교통": 8, "마트": 6, "병원": 5, "공원": 3},
        "마포구": {"교통": 5, "마트": 7, "병원": 4, "공원": 4},
        "성동구": {"교통": 6, "마트": 5, "병원": 6, "공원": 5},
    }
    d = sample_data.get(region)
    if not d:
        return {"error": "해당 지역 데이터 없음"}
    total = sum(d.values())
    return {"region": region, "infra": d, "total": total}
