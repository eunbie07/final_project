import os
import pandas as pd
import psycopg2
from dotenv import load_dotenv
from tqdm import tqdm

def migrate_csv_to_db():
  load_dotenv()
  conn = None
  try:
    conn = psycopg2.connect(
      host=os.getenv("DB_HOST"),
      port=os.getenv("DB_PORT"),
      dbname=os.getenv("DB_NAME"),
      user=os.getenv("DB_USER"),
      password=os.getenv("DB_PASSWORD")
    )
    cur = conn.cursor()
    print("DB 연결 성공")
    
    # 모든 인프라 테이블을 삭제하고 새로 생성
    create_tables_sql = """
    DROP TABLE IF EXISTS schools, subways, hospitals, marts, parks;
    CREATE TABLE schools (id SERIAL PRIMARY KEY, name VARCHAR(255), latitude DOUBLE PRECISION, longitude DOUBLE PRECISION, geom GEOGRAPHY(Point, 4326));
    CREATE TABLE subways (id SERIAL PRIMARY KEY, name VARCHAR(255), latitude DOUBLE PRECISION, longitude DOUBLE PRECISION, geom GEOGRAPHY(Point, 4326));
    CREATE TABLE hospitals (id SERIAL PRIMARY KEY, name VARCHAR(255), latitude DOUBLE PRECISION, longitude DOUBLE PRECISION, geom GEOGRAPHY(Point, 4326));
    CREATE TABLE marts (id SERIAL PRIMARY KEY, name VARCHAR(255), latitude DOUBLE PRECISION, longitude DOUBLE PRECISION, geom GEOGRAPHY(Point, 4326));
    CREATE TABLE parks (id SERIAL PRIMARY KEY, name VARCHAR(255), latitude DOUBLE PRECISION, longitude DOUBLE PRECISION, geom GEOGRAPHY(Point, 4326));
    """
    cur.execute(create_tables_sql)
    print("모든 인프라 테이블 생성 완료")
    
    print("CSV 데이터 삽입을 시작합니다...")
    
    # 처리할 파일 목록 (테이블이름: 파일 경로, 이름 컬럼)
    infra_files = {
      'schools': {'path': '../datas/encoding_schools.csv', 'name_col':'name'},
      'subways': {'path': '../datas/subway_combined.csv', 'name_col':'name'},
      'hospitals': {'path': '../datas/hospitals.csv', 'name_col':'name'},
      'marts': {'path': '../datas/marts.csv', 'name_col':'name'},
      'parks': {'path': '../datas/parks.csv', 'name_col':'name'}
    }
    
    for table_name, file_info in infra_files.items():
      try:
        df = pd.read_csv(file_info['path'])
        df.dropna(subset=[file_info['name_col'],'latitude','longitude'], inplace=True)
        
        for index, row in tqdm(df.iterrows(), total=len(df), desc=f"{table_name} 데이터 삽입 중"):
          cur.execute(
            f"INSERT INTO {table_name} (name, latitude, longitude, geom) VALUES (%s, %s, %s, ST_SetSRID(ST_MakePoint(%s, %s),4326))",
            (row[file_info['name_col']], row['latitude'], row['longitude'], row['longitude'],row['latitude'])
          )
        print(f"-> {table_name} 데이터 삽입 완료")
      except FileNotFoundError:
        print(f"경고: {file_info['path']} 파일을 찾을 수 없어 건너뜁니다.")
        
    # 공간 인덱스 생성
    create_index_sql = """
    CREATE INDEX IF NOT EXISTS schools_geom_idx ON schools USING GIST (geom);
    CREATE INDEX IF NOT EXISTS schools_geom_idx ON schools USING GIST (geom);
    CREATE INDEX IF NOT EXISTS schools_geom_idx ON schools USING GIST (geom);
    CREATE INDEX IF NOT EXISTS schools_geom_idx ON schools USING GIST (geom);
    CREATE INDEX IF NOT EXISTS schools_geom_idx ON schools USING GIST (geom);
    """
    cur.execute(create_index_sql)
    print("모든 공간 인덱스 생성 완료")
    
    conn.commit()
    print("모든 인프라 데이터가 성공적으로 저장되었습니다.")
    
  except Exception as e:
    if conn: conn.rollback()
    print(f"오류 발생: {e}")
  finally:
    if conn: conn.close()
    print("데이터 베이스 연결 해제")
    
if __name__ == "__main__":
  migrate_csv_to_db()
        