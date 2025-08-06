# 🎨 스타일 선택 웹사이트 개발 작업 (React + Tailwind)

## 📋 작업 개요

**목표**: `image/` 폴더의 45장 참조 이미지를 활용하여 사용자가 원하는 인테리어 스타일을 선택할 수 있는 React 컴포넌트 개발

**기술 스택**: React + Tailwind CSS  
**예상 작업 시간**: 1-2시간  
**난이도**: ⭐⭐☆☆☆ (쉬움)  
**담당자**: 프론트엔드 개발자

## 🎯 기능 요구사항

### 1. 스타일 갤러리 페이지
- **5개 카테고리**를 탭이나 섹션으로 구분
  - Modern/Minimalist (9장)
  - Scandinavian (9장) 
  - Industrial (9장)
  - Bohemian/Natural (9장)
  - Cozy (9장)

### 2. 이미지 표시 방식
- **그리드 레이아웃** (3x3 또는 반응형)
- **썸네일 크기**: 200x200px 정도
- **호버 효과**: 마우스 올리면 살짝 확대
- **클릭 가능**: 선택된 이미지 강조 표시

### 3. 선택 기능
- **단일 선택**: 한 번에 하나의 스타일만 선택 가능
- **선택 표시**: 선택된 이미지에 테두리나 체크 표시
- **"생성하기" 버튼**: 선택 완료 후 AI 이미지 생성 요청

### 4. 반응형 디자인
- **데스크탑**: 3-4열 그리드
- **태블릿**: 2-3열 그리드  
- **모바일**: 1-2열 그리드

## 📁 파일 구조

```
ai-interior/
├── style-selector/
│   ├── src/
│   │   ├── components/
│   │   │   ├── StyleSelector.jsx     # 메인 컴포넌트
│   │   │   ├── StyleTab.jsx          # 탭 컴포넌트  
│   │   │   └── ImageGallery.jsx      # 갤러리 컴포넌트
│   │   ├── utils/
│   │   │   └── imageData.js          # 이미지 경로 데이터
│   │   ├── App.jsx
│   │   └── index.js
│   ├── package.json
│   └── README.md
└── image/                            # 기존 참조 이미지들 (45장)
```

## 🚀 프로젝트 설정

### 1. 리액트 프로젝트 생성
```bash
# ai-interior 폴더에서 실행
npx create-react-app style-selector
cd style-selector

# Tailwind CSS 설치
npm install -D tailwindcss postcss autoprefixer
npx tailwindcss init -p

# 추가 패키지 설치 (선택사항)
npm install lucide-react  # 아이콘용
```

### 2. Tailwind 설정 (tailwind.config.js)
```javascript
/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    "./src/**/*.{js,jsx,ts,tsx}",
  ],
  theme: {
    extend: {},
  },
  plugins: [],
}
```

### 3. CSS 설정 (src/index.css)
```css
@tailwind base;
@tailwind components;
@tailwind utilities;
```

## 🔧 React 컴포넌트 구현

### 1. 이미지 데이터 (src/utils/imageData.js)
```javascript
export const styleImages = {
  modern: Array.from({length: 9}, (_, i) => ({
    id: `modern-${i+1}`,
    src: `../image/Modern,Minimalist${i+1}.png`,
    alt: `Modern Minimalist ${i+1}`
  })),
  scandinavian: Array.from({length: 9}, (_, i) => ({
    id: `scandinavian-${i+1}`,
    src: `../image/Scandinavian${i+1}.png`, 
    alt: `Scandinavian ${i+1}`
  })),
  industrial: Array.from({length: 9}, (_, i) => ({
    id: `industrial-${i+1}`,
    src: `../image/Industrial${i+1}.png`,
    alt: `Industrial ${i+1}`
  })),
  bohemian: Array.from({length: 9}, (_, i) => ({
    id: `bohemian-${i+1}`,
    src: `../image/Bohemian,Natural${i+1}.png`,
    alt: `Bohemian Natural ${i+1}`
  })),
  cozy: Array.from({length: 9}, (_, i) => ({
    id: `cozy-${i+1}`,
    src: `../image/Cozy${i+1}.png`,
    alt: `Cozy ${i+1}`
  }))
};

export const styleLabels = {
  modern: 'Modern/Minimalist',
  scandinavian: 'Scandinavian', 
  industrial: 'Industrial',
  bohemian: 'Bohemian/Natural',
  cozy: 'Cozy'
};
```

### 2. 메인 컴포넌트 (src/components/StyleSelector.jsx)
```jsx
import React, { useState } from 'react';
import { styleImages, styleLabels } from '../utils/imageData';
import StyleTab from './StyleTab';
import ImageGallery from './ImageGallery';

const StyleSelector = () => {
  const [activeStyle, setActiveStyle] = useState('modern');
  const [selectedImage, setSelectedImage] = useState(null);

  const handleImageSelect = (image) => {
    setSelectedImage(image);
  };

  const handleGenerate = async () => {
    if (!selectedImage) return;
    
    console.log('Generating image with style:', activeStyle, selectedImage);
    // TODO: API 호출 구현
    alert(`${styleLabels[activeStyle]} 스타일로 AI 이미지 생성을 시작합니다!`);
  };

  return (
    <div className="max-w-7xl mx-auto p-6">
      {/* 헤더 */}
      <div className="text-center mb-8">
        <h1 className="text-3xl font-bold text-gray-800 mb-4">
          🏠 원하는 인테리어 스타일을 선택해주세요
        </h1>
        <p className="text-gray-600">
          45장의 참조 이미지에서 마음에 드는 스타일을 골라보세요
        </p>
      </div>

      {/* 스타일 탭 */}
      <StyleTab 
        styles={Object.keys(styleImages)}
        activeStyle={activeStyle}
        onStyleChange={setActiveStyle}
        styleLabels={styleLabels}
      />

      {/* 이미지 갤러리 */}
      <ImageGallery
        images={styleImages[activeStyle]}
        selectedImage={selectedImage}
        onImageSelect={handleImageSelect}
      />

      {/* 액션 버튼들 */}
      <div className="mt-8 text-center">
        {selectedImage && (
          <div className="mb-4 p-4 bg-blue-50 rounded-lg inline-block">
            <p className="text-blue-800 font-medium">
              ✅ 선택됨: {styleLabels[activeStyle]} 스타일
            </p>
            <p className="text-sm text-blue-600 mt-1">
              {selectedImage.alt}
            </p>
          </div>
        )}
        
        <button
          onClick={handleGenerate}
          disabled={!selectedImage}
          className={`px-8 py-3 rounded-lg font-semibold text-white transition-all ${
            selectedImage 
              ? 'bg-blue-600 hover:bg-blue-700 shadow-lg hover:shadow-xl' 
              : 'bg-gray-400 cursor-not-allowed'
          }`}
        >
          🎨 이 스타일로 생성하기
        </button>
      </div>
    </div>
  );
};

export default StyleSelector;
```

### 3. 탭 컴포넌트 (src/components/StyleTab.jsx)
```jsx
import React from 'react';

const StyleTab = ({ styles, activeStyle, onStyleChange, styleLabels }) => {
  return (
    <div className="flex flex-wrap justify-center gap-2 mb-8">
      {styles.map((style) => (
        <button
          key={style}
          onClick={() => onStyleChange(style)}
          className={`px-6 py-3 rounded-lg font-medium transition-all ${
            activeStyle === style
              ? 'bg-blue-600 text-white shadow-lg'
              : 'bg-white text-gray-700 border border-gray-300 hover:bg-gray-50 hover:border-gray-400'
          }`}
        >
          {styleLabels[style]}
        </button>
      ))}
    </div>
  );
};

export default StyleTab;
```

### 4. 갤러리 컴포넌트 (src/components/ImageGallery.jsx)
```jsx
import React from 'react';

const ImageGallery = ({ images, selectedImage, onImageSelect }) => {
  return (
    <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4 xl:grid-cols-5 gap-4">
      {images.map((image) => (
        <div
          key={image.id}
          onClick={() => onImageSelect(image)}
          className={`relative cursor-pointer rounded-lg overflow-hidden transition-all duration-200 ${
            selectedImage?.id === image.id
              ? 'ring-4 ring-blue-500 shadow-xl transform scale-105'
              : 'hover:shadow-lg hover:transform hover:scale-105'
          }`}
        >
          <img
            src={image.src}
            alt={image.alt}
            className="w-full h-48 object-cover"
            loading="lazy"
          />
          
          {/* 선택 표시 */}
          {selectedImage?.id === image.id && (
            <div className="absolute top-2 right-2 bg-blue-600 text-white rounded-full p-1">
              <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 20 20">
                <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
              </svg>
            </div>
          )}
          
          {/* 호버 오버레이 */}
          <div className="absolute inset-0 bg-black bg-opacity-0 hover:bg-opacity-20 transition-all duration-200 flex items-center justify-center">
            <span className="text-white font-medium opacity-0 hover:opacity-100 transition-opacity">
              선택하기
            </span>
          </div>
        </div>
      ))}
    </div>
  );
};

export default ImageGallery;
```

### 5. App.jsx
```jsx
import React from 'react';
import StyleSelector from './components/StyleSelector';

function App() {
  return (
    <div className="min-h-screen bg-gray-50">
      <StyleSelector />
    </div>
  );
}

export default App;
```

## 🚀 실행 방법

### 1. 개발 서버 실행
```bash
cd style-selector
npm start
```

### 2. 빌드 (배포용)
```bash
npm run build
```

## 🚀 API 연동 (선택사항)

기존 AI 이미지 생성 시스템과 연동하려면 `StyleSelector.jsx`의 `handleGenerate` 함수를 수정:

```jsx
const handleGenerate = async () => {
  if (!selectedImage) return;
  
  try {
    const response = await fetch('/api/generate-image', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        style: activeStyle,
        reference_image: selectedImage.src,
        // 방 데이터도 함께 전송 가능
      })
    });
    
    const result = await response.json();
    if (result.success) {
      // 생성된 이미지 표시
      window.open(result.image_url, '_blank');
    } else {
      alert('이미지 생성에 실패했습니다.');
    }
  } catch (error) {
    console.error('API 호출 오류:', error);
    alert('서버 연결에 실패했습니다.');
  }
};
```

## ✅ 완료 체크리스트

- [ ] React 프로젝트 생성 및 Tailwind CSS 설정
- [ ] 컴포넌트 구조 구현 (StyleSelector, StyleTab, ImageGallery)
- [ ] 이미지 데이터 설정 (45장 모든 이미지)
- [ ] 스타일 탭 전환 기능 구현
- [ ] 이미지 선택 및 하이라이트 기능
- [ ] 반응형 그리드 레이아웃 (Tailwind responsive classes)
- [ ] 모바일/태블릿/데스크탑 테스트
- [ ] 이미지 lazy loading 확인
- [ ] 생성 버튼 활성화/비활성화 로직
- [ ] (선택) API 연동 테스트

## 📁 제출물

1. **style-selector/ 폴더 전체** (React 프로젝트)
2. **스크린샷** (데스크탑/모바일/태블릿 화면)
3. **package.json** (의존성 확인용)
4. **README.md** (실행 방법 및 사용법)

## 💡 추가 개선 아이디어 (선택사항)

- **로딩 스피너**: 이미지 로딩 중 스피너 표시
- **이미지 모달**: 클릭하면 큰 이미지로 미리보기
- **애니메이션**: Framer Motion 추가로 부드러운 전환
- **검색 기능**: 스타일명으로 필터링
- **즐겨찾기**: localStorage로 선호 스타일 저장
- **다크모드**: Tailwind dark: 클래스 활용

## 🔧 문제 해결

### 이미지 경로 문제
```javascript
// public 폴더로 이미지 이동 후
src: `/image/Modern,Minimalist${i+1}.png`

// 또는 import 방식 사용
import modernImage1 from '../assets/image/Modern,Minimalist1.png';
```

### Tailwind 적용 안됨
```bash
# Tailwind 설정 재확인
npx tailwindcss -i ./src/index.css -o ./dist/output.css --watch
```

### 반응형 그리드 조정
```jsx
// 더 세밀한 반응형 설정
className="grid grid-cols-1 xs:grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 xl:grid-cols-6 gap-4"
```

---

**🎯 목표**: 사용자가 직관적으로 원하는 스타일을 선택할 수 있는 깔끔한 웹 인터페이스 완성!

**📞 질문이나 막히는 부분이 있으면 언제든 연락 주세요.**