import asyncio
import aiohttp
from bs4 import BeautifulSoup
import pandas as pd
import time
import json
import re
from datetime import datetime
from fake_useragent import UserAgent
import logging
from typing import List, Dict, Optional
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AdvancedRealEstateCrawler:
    def __init__(self):
        self.ua = UserAgent()
        self.session = None
        self.data = []
        
    async def create_session(self):
        """비동기 세션 생성"""
        headers = {
            'User-Agent': self.ua.random,
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'ko-KR,ko;q=0.8,en-US;q=0.5,en;q=0.3',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        }
        
        timeout = aiohttp.ClientTimeout(total=30)
        self.session = aiohttp.ClientSession(headers=headers, timeout=timeout)
    
    async def close_session(self):
        """세션 종료"""
        if self.session:
            await self.session.close()
    
    async def fetch_page(self, url: str) -> Optional[str]:
        """웹페이지 비동기 가져오기"""
        try:
            async with self.session.get(url) as response:
                if response.status == 200:
                    return await response.text()
                else:
                    logger.warning(f"HTTP {response.status}: {url}")
                    return None
        except Exception as e:
            logger.error(f"페이지 가져오기 오류: {url} - {str(e)}")
            return None
    
    def parse_ziptoss_data(self, html: str) -> List[Dict]:
        """Ziptoss HTML 파싱"""
        soup = BeautifulSoup(html, 'html.parser')
        properties = []
        
        # 다양한 선택자 시도
        selectors = [
            'div[class*="property"]',
            'div[class*="listing"]',
            'div[class*="item"]',
            'div[class*="card"]',
            'article',
            '.property-item',
            '.listing-item',
            '.item-card'
        ]
        
        for selector in selectors:
            elements = soup.select(selector)
            if elements:
                logger.info(f"선택자 '{selector}'로 {len(elements)}개 요소 발견")
                break
        
        if not elements:
            # 모든 div 요소 검사
            elements = soup.find_all('div', class_=True)
            logger.info(f"기본 div 요소 {len(elements)}개 검사")
        
        for element in elements:
            try:
                property_data = self.extract_property_info(element)
                if property_data:
                    properties.append(property_data)
            except Exception as e:
                logger.warning(f"매물 데이터 추출 오류: {str(e)}")
                continue
        
        return properties
    
    def extract_property_info(self, element) -> Optional[Dict]:
        """개별 매물 정보 추출"""
        # 텍스트 추출 함수들
        def extract_text(selectors):
            for selector in selectors:
                try:
                    found = element.select_one(selector)
                    if found:
                        text = found.get_text(strip=True)
                        if text:
                            return text
                except:
                    continue
            return ""
        
        def extract_price():
            price_selectors = [
                '.price', '.cost', '.amount', '.value',
                '[class*="price"]', '[class*="cost"]', '[class*="amount"]'
            ]
            price_text = extract_text(price_selectors)
            
            # 가격 정규화 (숫자만 추출)
            if price_text:
                price_match = re.search(r'[\d,]+', price_text.replace(' ', ''))
                if price_match:
                    return price_match.group()
            return price_text
        
        # 매물 정보 추출
        property_data = {
            'title': extract_text(['h1', 'h2', 'h3', '.title', '.name', '[class*="title"]']),
            'price': extract_price(),
            'address': extract_text(['.address', '.location', '.addr', '[class*="address"]']),
            'size': extract_text(['.size', '.area', '.square', '[class*="size"]']),
            'rooms': extract_text(['.rooms', '.bedrooms', '.room-count', '[class*="room"]']),
            'type': extract_text(['.type', '.category', '.property-type', '[class*="type"]']),
            'description': extract_text(['.description', '.desc', '.summary', '[class*="desc"]']),
            'crawled_at': datetime.now().isoformat()
        }
        
        # 빈 값 제거
        property_data = {k: v for k, v in property_data.items() if v}
        
        # 최소한 제목이나 가격 중 하나는 있어야 유효한 데이터로 간주
        if property_data.get('title') or property_data.get('price'):
            return property_data
        
        return None
    
    async def crawl_ziptoss_async(self, lat: float = 37.4801602, lng: float = 126.9521682, zoom: int = 17):
        """Ziptoss 비동기 크롤링"""
        url = f"https://ziptoss.com/?zoom={zoom}&lat={lat}&lng={lng}&sort=DEFAULT&order=DESC"
        
        await self.create_session()
        
        try:
            logger.info(f"Ziptoss 크롤링 시작: {url}")
            
            html = await self.fetch_page(url)
            if html:
                properties = self.parse_ziptoss_data(html)
                self.data.extend(properties)
                logger.info(f"Ziptoss에서 {len(properties)}개 매물 수집")
            else:
                logger.error("Ziptoss 페이지를 가져올 수 없습니다.")
                
        except Exception as e:
            logger.error(f"Ziptoss 크롤링 오류: {str(e)}")
        finally:
            await self.close_session()
    
    def save_data(self, filename_prefix: str = "real_estate"):
        """수집된 데이터 저장"""
        if not self.data:
            logger.warning("저장할 데이터가 없습니다.")
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # CSV 저장
        csv_filename = f"{filename_prefix}_{timestamp}.csv"
        df = pd.DataFrame(self.data)
        df.to_csv(csv_filename, index=False, encoding='utf-8-sig')
        logger.info(f"CSV 저장 완료: {csv_filename}")
        
        # JSON 저장
        json_filename = f"{filename_prefix}_{timestamp}.json"
        with open(json_filename, 'w', encoding='utf-8') as f:
            json.dump(self.data, f, ensure_ascii=False, indent=2)
        logger.info(f"JSON 저장 완료: {json_filename}")
        
        # 통계 출력
        self.print_statistics()
    
    def print_statistics(self):
        """수집된 데이터 통계 출력"""
        if not self.data:
            return
        
        print(f"\n=== 수집된 데이터 통계 ===")
        print(f"총 매물 수: {len(self.data)}")
        
        # 가격 통계
        prices = []
        for item in self.data:
            if item.get('price'):
                try:
                    price = int(item['price'].replace(',', ''))
                    prices.append(price)
                except:
                    continue
        
        if prices:
            print(f"평균 가격: {sum(prices) / len(prices):,.0f}")
            print(f"최고 가격: {max(prices):,.0f}")
            print(f"최저 가격: {min(prices):,.0f}")
        
        # 매물 타입별 통계
        types = {}
        for item in self.data:
            prop_type = item.get('type', '기타')
            types[prop_type] = types.get(prop_type, 0) + 1
        
        print(f"\n매물 타입별 분포:")
        for prop_type, count in sorted(types.items(), key=lambda x: x[1], reverse=True):
            print(f"  {prop_type}: {count}개")

async def main():
    """메인 실행 함수"""
    crawler = AdvancedRealEstateCrawler()
    
    # Ziptoss 크롤링
    await crawler.crawl_ziptoss_async()
    
    # 데이터 저장
    crawler.save_data()
    
    # 결과 미리보기
    if crawler.data:
        print(f"\n=== 처음 3개 매물 미리보기 ===")
        for i, prop in enumerate(crawler.data[:3], 1):
            print(f"\n매물 {i}:")
            for key, value in prop.items():
                if key != 'crawled_at':
                    print(f"  {key}: {value}")

if __name__ == "__main__":
    asyncio.run(main()) 