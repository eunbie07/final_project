import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import json
from fake_useragent import UserAgent
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RealEstateCrawler:
    def __init__(self):
        self.ua = UserAgent()
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': self.ua.random,
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'ko-KR,ko;q=0.8,en-US;q=0.5,en;q=0.3',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        })
        
    def setup_selenium_driver(self):
        """Selenium WebDriver 설정"""
        chrome_options = Options()
        chrome_options.add_argument('--headless')  # 헤드리스 모드
        chrome_options.add_argument('--no-sandbox')
        chrome_options.add_argument('--disable-dev-shm-usage')
        chrome_options.add_argument(f'--user-agent={self.ua.random}')
        
        service = Service(ChromeDriverManager().install())
        driver = webdriver.Chrome(service=service, options=chrome_options)
        return driver
    
    def crawl_ziptoss(self, lat=37.4801602, lng=126.9521682, zoom=17):
        """Ziptoss 부동산 데이터 크롤링"""
        url = f"https://ziptoss.com/?zoom={zoom}&lat={lat}&lng={lng}&sort=DEFAULT&order=DESC"
        
        try:
            logger.info(f"크롤링 시작: {url}")
            
            # Selenium을 사용하여 동적 콘텐츠 로드
            driver = self.setup_selenium_driver()
            driver.get(url)
            
            # 페이지 로딩 대기
            wait = WebDriverWait(driver, 10)
            
            # 부동산 매물 정보가 로드될 때까지 대기
            try:
                wait.until(EC.presence_of_element_located((By.CLASS_NAME, "property-item")))
            except:
                logger.warning("부동산 매물 요소를 찾을 수 없습니다. 기본 대기 시간을 사용합니다.")
                time.sleep(5)
            
            # 페이지 소스 가져오기
            page_source = driver.page_source
            soup = BeautifulSoup(page_source, 'html.parser')
            
            # 부동산 매물 정보 추출
            properties = self.extract_property_data(soup)
            
            driver.quit()
            
            return properties
            
        except Exception as e:
            logger.error(f"크롤링 중 오류 발생: {str(e)}")
            return []
    
    def extract_property_data(self, soup):
        """HTML에서 부동산 데이터 추출"""
        properties = []
        
        # 실제 CSS 선택자는 웹사이트 구조에 따라 조정 필요
        property_elements = soup.find_all('div', class_='property-item') or \
                           soup.find_all('div', class_='listing-item') or \
                           soup.find_all('div', {'data-testid': 'property-card'})
        
        if not property_elements:
            # 더 일반적인 선택자들 시도
            property_elements = soup.find_all('div', class_=lambda x: x and ('property' in x.lower() or 'listing' in x.lower()))
        
        logger.info(f"발견된 매물 수: {len(property_elements)}")
        
        for element in property_elements:
            try:
                property_data = {
                    'title': self.extract_text(element, ['h1', 'h2', 'h3', '.title', '.name']),
                    'price': self.extract_text(element, ['.price', '.cost', '.amount']),
                    'address': self.extract_text(element, ['.address', '.location', '.addr']),
                    'size': self.extract_text(element, ['.size', '.area', '.square']),
                    'rooms': self.extract_text(element, ['.rooms', '.bedrooms', '.room-count']),
                    'type': self.extract_text(element, ['.type', '.category', '.property-type']),
                    'raw_html': str(element)[:500]  # 디버깅용
                }
                
                # 빈 값 제거
                property_data = {k: v.strip() for k, v in property_data.items() if v and v.strip()}
                
                if property_data:
                    properties.append(property_data)
                    
            except Exception as e:
                logger.warning(f"매물 데이터 추출 중 오류: {str(e)}")
                continue
        
        return properties
    
    def extract_text(self, element, selectors):
        """여러 선택자를 시도하여 텍스트 추출"""
        for selector in selectors:
            try:
                found = element.select_one(selector)
                if found:
                    return found.get_text(strip=True)
            except:
                continue
        return ""
    
    def save_to_csv(self, properties, filename="real_estate_data.csv"):
        """데이터를 CSV 파일로 저장"""
        if properties:
            df = pd.DataFrame(properties)
            df.to_csv(filename, index=False, encoding='utf-8-sig')
            logger.info(f"데이터가 {filename}에 저장되었습니다.")
        else:
            logger.warning("저장할 데이터가 없습니다.")
    
    def save_to_json(self, properties, filename="real_estate_data.json"):
        """데이터를 JSON 파일로 저장"""
        if properties:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(properties, f, ensure_ascii=False, indent=2)
            logger.info(f"데이터가 {filename}에 저장되었습니다.")
        else:
            logger.warning("저장할 데이터가 없습니다.")

def main():
    """메인 실행 함수"""
    crawler = RealEstateCrawler()
    
    # Ziptoss 크롤링
    properties = crawler.crawl_ziptoss()
    
    if properties:
        logger.info(f"총 {len(properties)}개의 매물 정보를 수집했습니다.")
        
        # 데이터 저장
        crawler.save_to_csv(properties)
        crawler.save_to_json(properties)
        
        # 결과 출력
        for i, prop in enumerate(properties[:5], 1):  # 처음 5개만 출력
            print(f"\n매물 {i}:")
            for key, value in prop.items():
                if key != 'raw_html':
                    print(f"  {key}: {value}")
    else:
        logger.warning("수집된 매물 정보가 없습니다.")

if __name__ == "__main__":
    main() 