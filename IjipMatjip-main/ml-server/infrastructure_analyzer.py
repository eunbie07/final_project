import os
import psycopg2
from dotenv import load_dotenv

class InfrastructureAnalyzer:
  def __init__(self):
    print("인프라 분석기 초기화")
    load_dotenv()
    self.db_config = {
      "host": os.getenv("DB_HOST"),
      "port": os.getenv("DB_PORT"),
      "dbname": os.getenv("DB_NAME"),
      "user": os.getenv("DB_USER"),
      "password": os.getenv("DB_PASSWORD")
    }
  def find_nearby(self, lat, lng, radius_km, infra_type):
    if lat is None or lng is None: return []
    
    # 검색할 테이블 이름을 infra_type에 따라 결정
    table_map = {
      'school': 'schools',
      'subway': 'subways',
      'hospital': 'hospitals',
      'mart': 'marts',
      'park': 'parks'
    }
    table_name = table_map.get(infra_type)
    
    if not table_name: return []
    
    query = f"""
    SELECT name, latitude, longitude
    FROM {table_name}
    WHERE ST_DWithin(
      geom,
      ST_SetSRID(ST_MakePoint(%s, %s), 4326),
      %s
    );
    """
    
    results = []
    conn = None
    try:
      conn = psycopg2.connect(**self.db_config)
      with conn.cursor() as cur:
        cur.execute(query, (lng, lat, radius_km * 500))
        rows = cur.fetchall()
        for row in rows:
          results.append({
            'name': row[0],
            'latitude': row[1],
            'longitude': row[2],
          })
    except Exception as e:
      print(f"DB 쿼리 중 오류 발생 ({table_name}): {e}")
    finally:
      if conn: conn.close()
    
    return results