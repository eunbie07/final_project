# Node.js 기반 프로덕션 서버
FROM node:18-alpine

WORKDIR /app

# package.json과 package-lock.json 복사
COPY package*.json ./

# 의존성 설치
RUN npm ci --only=production

# 소스 코드 복사
COPY . .

# 프로덕션 빌드
RUN npm run build

# 프로덕션용 서버 설치 (serve 패키지)
RUN npm install -g serve

# 헬스체크 추가
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:4000/health || exit 1

# 비루트 사용자 생성 및 전환
RUN addgroup -g 1001 -S nodejs
RUN adduser -S reactuser -u 1001
USER reactuser

# 포트 노출
EXPOSE 4000

# React 앱 실행 (serve 사용)
CMD ["serve", "-s", "dist", "-l", "4000"]