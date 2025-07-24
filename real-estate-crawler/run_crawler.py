#!/usr/bin/env python3
"""
부동산 크롤러 실행 스크립트
사용법: python run_crawler.py [옵션]
"""

import argparse
import sys
import os
from datetime import datetime
import asyncio

# 프로젝트 루트를 Python 경로에 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from main import RealEstateCrawler
from advanced_crawler import AdvancedRealEstateCrawler
from config import DEFAULT_LAT, DEFAULT_LNG, DEFAULT_ZOOM

def parse_arguments():
    """명령행 인수 파싱"""
    parser = argparse.ArgumentParser(
        description='부동산 크롤러 실행 스크립트',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
사용 예시:
  python run_crawler.py                    # 기본 설정으로 크롤링
  python run_crawler.py --lat 37.5 --lng 127.0  # 특정 위치 크롤링
  python run_crawler.py --advanced         # 고급 크롤러 사용
  python run_crawler.py --output my_data   # 출력 파일명 지정
        """
    )
    
    parser.add_argument(
        '--lat', 
        type=float, 
        default=DEFAULT_LAT,
        help=f'위도 (기본값: {DEFAULT_LAT})'
    )
    
    parser.add_argument(
        '--lng', 
        type=float, 
        default=DEFAULT_LNG,
        help=f'경도 (기본값: {DEFAULT_LNG})'
    )
    
    parser.add_argument(
        '--zoom', 
        type=int, 
        default=DEFAULT_ZOOM,
        help=f'줌 레벨 (기본값: {DEFAULT_ZOOM})'
    )
    
    parser.add_argument(
        '--advanced', 
        action='store_true',
        help='고급 크롤러 사용 (비동기)'
    )
    
    parser.add_argument(
        '--output', 
        type=str, 
        default='real_estate',
        help='출력 파일명 (확장자 제외)'
    )
    
    parser.add_argument(
        '--headless', 
        action='store_true', 
        default=True,
        help='헤드리스 모드 사용 (기본값: True)'
    )
    
    parser.add_argument(
        '--delay', 
        type=float, 
        default=1.0,
        help='요청 간 딜레이 (초)'
    )
    
    parser.add_argument(
        '--timeout', 
        type=int, 
        default=30,
        help='요청 타임아웃 (초)'
    )
    
    parser.add_argument(
        '--verbose', 
        action='store_true',
        help='상세 로그 출력'
    )
    
    return parser.parse_args()

def run_basic_crawler(args):
    """기본 크롤러 실행"""
    print("🚀 기본 크롤러 시작...")
    print(f"📍 위치: 위도 {args.lat}, 경도 {args.lng}, 줌 {args.zoom}")
    print(f"⏱️  딜레이: {args.delay}초, 타임아웃: {args.timeout}초")
    print("-" * 50)
    
    try:
        crawler = RealEstateCrawler()
        
        # 크롤링 실행
        properties = crawler.crawl_ziptoss(
            lat=args.lat,
            lng=args.lng,
            zoom=args.zoom
        )
        
        if properties:
            print(f"✅ {len(properties)}개의 매물 정보를 수집했습니다.")
            
            # 데이터 저장
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            csv_filename = f"{args.output}_{timestamp}.csv"
            json_filename = f"{args.output}_{timestamp}.json"
            
            crawler.save_to_csv(properties, csv_filename)
            crawler.save_to_json(properties, json_filename)
            
            print(f"💾 데이터 저장 완료:")
            print(f"   CSV: {csv_filename}")
            print(f"   JSON: {json_filename}")
            
            # 결과 미리보기
            if args.verbose:
                print("\n📋 처음 3개 매물 미리보기:")
                for i, prop in enumerate(properties[:3], 1):
                    print(f"\n매물 {i}:")
                    for key, value in prop.items():
                        if key != 'raw_html':
                            print(f"  {key}: {value}")
        else:
            print("⚠️  수집된 매물 정보가 없습니다.")
            
    except Exception as e:
        print(f"❌ 크롤링 중 오류 발생: {str(e)}")
        return False
    
    return True

async def run_advanced_crawler(args):
    """고급 크롤러 실행"""
    print("🚀 고급 크롤러 시작 (비동기)...")
    print(f"📍 위치: 위도 {args.lat}, 경도 {args.lng}, 줌 {args.zoom}")
    print("-" * 50)
    
    try:
        crawler = AdvancedRealEstateCrawler()
        
        # 비동기 크롤링 실행
        await crawler.crawl_ziptoss_async(
            lat=args.lat,
            lng=args.lng,
            zoom=args.zoom
        )
        
        if crawler.data:
            print(f"✅ {len(crawler.data)}개의 매물 정보를 수집했습니다.")
            
            # 데이터 저장
            crawler.save_data(args.output)
            
        else:
            print("⚠️  수집된 매물 정보가 없습니다.")
            
    except Exception as e:
        print(f"❌ 크롤링 중 오류 발생: {str(e)}")
        return False
    
    return True

def main():
    """메인 함수"""
    print("🏠 부동산 크롤러 v1.0")
    print("=" * 50)
    
    # 인수 파싱
    args = parse_arguments()
    
    # 시작 시간 기록
    start_time = datetime.now()
    
    try:
        if args.advanced:
            # 고급 크롤러 실행
            success = asyncio.run(run_advanced_crawler(args))
        else:
            # 기본 크롤러 실행
            success = run_basic_crawler(args)
        
        # 종료 시간 및 소요 시간 계산
        end_time = datetime.now()
        duration = end_time - start_time
        
        print("\n" + "=" * 50)
        if success:
            print(f"✅ 크롤링 완료! 소요 시간: {duration}")
        else:
            print(f"❌ 크롤링 실패! 소요 시간: {duration}")
        
    except KeyboardInterrupt:
        print("\n⚠️  사용자에 의해 중단되었습니다.")
    except Exception as e:
        print(f"\n❌ 예상치 못한 오류: {str(e)}")

if __name__ == "__main__":
    main() 