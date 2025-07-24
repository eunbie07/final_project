#!/usr/bin/env python3
"""
웹사이트 구조 디버깅 스크립트
실제 웹사이트의 HTML 구조를 분석하여 올바른 CSS 선택자를 찾습니다.
"""

import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
import time
import json

def setup_driver():
    """Selenium WebDriver 설정"""
    chrome_options = Options()
    # chrome_options.add_argument('--headless')  # 디버깅을 위해 헤드리스 모드 비활성화
    chrome_options.add_argument('--no-sandbox')
    chrome_options.add_argument('--disable-dev-shm-usage')
    chrome_options.add_argument('--disable-blink-features=AutomationControlled')
    chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
    chrome_options.add_experimental_option('useAutomationExtension', False)
    
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=chrome_options)
    
    # User-Agent 설정
    driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
    
    return driver

def analyze_page_structure(url):
    """페이지 구조 분석"""
    print(f"🔍 페이지 분석 시작: {url}")
    
    driver = setup_driver()
    
    try:
        # 페이지 로드
        driver.get(url)
        print("📄 페이지 로딩 완료")
        
        # 페이지가 완전히 로드될 때까지 대기
        time.sleep(10)
        
        # 페이지 소스 가져오기
        page_source = driver.page_source
        soup = BeautifulSoup(page_source, 'html.parser')
        
        print(f"📊 HTML 크기: {len(page_source)} 문자")
        
        # 모든 div 요소 찾기
        divs = soup.find_all('div')
        print(f"🔍 발견된 div 요소 수: {len(divs)}")
        
        # 클래스가 있는 div들 분석
        divs_with_class = [div for div in divs if div.get('class')]
        print(f"🏷️  클래스가 있는 div 수: {len(divs_with_class)}")
        
        # 클래스별로 그룹화
        class_counts = {}
        for div in divs_with_class:
            for class_name in div.get('class', []):
                class_counts[class_name] = class_counts.get(class_name, 0) + 1
        
        # 가장 많이 사용된 클래스들 출력
        print("\n📈 가장 많이 사용된 클래스들:")
        sorted_classes = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)
        for class_name, count in sorted_classes[:20]:
            print(f"  {class_name}: {count}개")
        
        # 부동산 관련 키워드가 포함된 클래스 찾기
        real_estate_keywords = ['property', 'listing', 'item', 'card', 'house', 'apartment', 'rent', 'sale', '매물', '부동산']
        
        print("\n🏠 부동산 관련 클래스들:")
        for class_name, count in sorted_classes:
            if any(keyword in class_name.lower() for keyword in real_estate_keywords):
                print(f"  {class_name}: {count}개")
        
        # 실제 매물 정보가 있을 것 같은 요소들 찾기
        print("\n🔍 잠재적 매물 요소들:")
        
        # 가격 관련 요소 찾기
        price_elements = soup.find_all(text=lambda text: text and any(keyword in text for keyword in ['원', '만원', '억', '가격', 'price', 'cost']))
        print(f"💰 가격 관련 텍스트: {len(price_elements)}개")
        for i, elem in enumerate(price_elements[:10]):
            print(f"  {i+1}. {elem.strip()[:100]}")
        
        # 주소 관련 요소 찾기
        address_elements = soup.find_all(text=lambda text: text and any(keyword in text for keyword in ['동', '구', '시', '로', '길', 'address']))
        print(f"📍 주소 관련 텍스트: {len(address_elements)}개")
        for i, elem in enumerate(address_elements[:10]):
            print(f"  {i+1}. {elem.strip()[:100]}")
        
        # HTML 구조를 파일로 저장 (디버깅용)
        with open('page_structure.html', 'w', encoding='utf-8') as f:
            f.write(page_source)
        print("\n💾 HTML 구조가 'page_structure.html' 파일에 저장되었습니다.")
        
        # 스크린샷 저장
        driver.save_screenshot('page_screenshot.png')
        print("📸 스크린샷이 'page_screenshot.png' 파일에 저장되었습니다.")
        
        return soup
        
    except Exception as e:
        print(f"❌ 오류 발생: {str(e)}")
        return None
    finally:
        driver.quit()

def test_selectors(soup):
    """다양한 선택자 테스트"""
    print("\n🧪 선택자 테스트:")
    
    # 테스트할 선택자들
    selectors = [
        'div[class*="property"]',
        'div[class*="listing"]',
        'div[class*="item"]',
        'div[class*="card"]',
        'div[class*="house"]',
        'div[class*="apartment"]',
        'div[class*="rent"]',
        'div[class*="sale"]',
        'article',
        '.property-item',
        '.listing-item',
        '.item-card',
        '[data-testid*="property"]',
        '[data-testid*="listing"]',
        '[data-testid*="item"]',
    ]
    
    for selector in selectors:
        elements = soup.select(selector)
        if elements:
            print(f"✅ {selector}: {len(elements)}개 요소 발견")
            # 첫 번째 요소의 내용 일부 출력
            if elements:
                first_elem = elements[0]
                text_content = first_elem.get_text(strip=True)[:100]
                print(f"   예시: {text_content}...")
        else:
            print(f"❌ {selector}: 요소 없음")

def main():
    """메인 함수"""
    url = "https://ziptoss.com/?zoom=17&lat=37.4801602&lng=126.9521682&sort=DEFAULT&order=DESC"
    
    print("🔍 Ziptoss 웹사이트 구조 분석")
    print("=" * 50)
    
    soup = analyze_page_structure(url)
    
    if soup:
        test_selectors(soup)
        
        print("\n" + "=" * 50)
        print("📋 다음 단계:")
        print("1. 'page_structure.html' 파일을 브라우저에서 열어서 구조 확인")
        print("2. 'page_screenshot.png' 파일로 실제 페이지 모습 확인")
        print("3. 발견된 클래스들을 바탕으로 main.py의 선택자 업데이트")
    else:
        print("❌ 페이지 분석 실패")

if __name__ == "__main__":
    main() 