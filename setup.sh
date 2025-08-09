#!/bin/bash

# 색상 정의
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RESET='\033[0m'

echo "=========================================="
echo "    Final Project Environment Setup"
echo "=========================================="
echo

# 현재 디렉토리 확인
echo -e "${BLUE}Current directory: $(pwd)${RESET}"
echo

# 스크립트 디렉토리로 이동
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 1. imggen-realistic-uv7000-triple 설정
echo -e "${YELLOW}[1/4] Setting up imggen-realistic-uv7000-triple...${RESET}"

if [ -d "imggen-realistic-uv7000-triple/backend" ]; then
    cd imggen-realistic-uv7000-triple/backend
    
    # .env 파일 생성
    if [ ! -f ".env" ]; then
        echo "Creating .env file..."
        cat > .env << 'EOF'
# Stability AI API Key
# Get your key from: https://platform.stability.ai/account/keys
STABILITY_API_KEY=sk-your-stability-key-here

# Replicate API Token
# Get your token from: https://replicate.com/account/api-tokens
REPLICATE_API_TOKEN=r8_your-replicate-token-here

# Google Cloud Service Account (Optional)
GCP_SERVICE_ACCOUNT_JSON_PATH=./service-account.json
EOF
        echo -e "${GREEN}✓ Created imggen-realistic-uv7000-triple/backend/.env${RESET}"
    else
        echo -e "${YELLOW}! .env file already exists${RESET}"
    fi

    # Python 의존성 설치 시도
    echo "Installing Python dependencies..."
    if command -v uv &> /dev/null; then
        if uv pip install -e . &> /dev/null; then
            echo -e "${GREEN}✓ Python dependencies installed${RESET}"
        else
            echo -e "${RED}Warning: Failed to install Python dependencies${RESET}"
        fi
    else
        echo -e "${RED}Warning: 'uv' command not found. Install with: pip install uv${RESET}"
    fi
    
    cd "$SCRIPT_DIR"
else
    echo -e "${RED}Error: imggen-realistic-uv7000-triple/backend directory not found${RESET}"
    exit 1
fi

# 2. room-measure 설정
echo
echo -e "${YELLOW}[2/4] Setting up room-measure...${RESET}"

if [ -d "room-measure" ]; then
    cd room-measure
    
    # 메인 .env 파일 생성
    if [ ! -f ".env" ]; then
        echo "Creating main .env file..."
        cat > .env << 'EOF'
# PostgreSQL Database Settings
POSTGRES_USER=postgres
POSTGRES_PASSWORD=change_this_secure_password_123
POSTGRES_DB=room_measure
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
DATABASE_URL=postgresql://postgres:change_this_secure_password_123@localhost:5432/room_measure

# JWT Authentication (Change this in production)
JWT_SECRET=your-super-secret-jwt-key-minimum-32-characters-long-change-this

# Environment
NODE_ENV=development
EOF
        echo -e "${GREEN}✓ Created room-measure/.env${RESET}"
    else
        echo -e "${YELLOW}! Main .env file already exists${RESET}"
    fi

    # Backend cloud 설정
    if [ -d "eunbi/backend-cloud" ]; then
        cd eunbi/backend-cloud
        echo "Installing backend-cloud dependencies..."
        if command -v uv &> /dev/null && uv pip install -e . &> /dev/null; then
            echo -e "${GREEN}✓ Backend-cloud dependencies installed${RESET}"
        else
            echo -e "${RED}Warning: Failed to install backend-cloud dependencies${RESET}"
        fi
        cd "$SCRIPT_DIR/room-measure"
    fi

    # Backend local 설정
    if [ -d "eunbi/backend-local" ]; then
        cd eunbi/backend-local
        echo "Installing backend-local dependencies..."
        if command -v uv &> /dev/null && uv pip install -e . &> /dev/null; then
            echo -e "${GREEN}✓ Backend-local dependencies installed${RESET}"
        else
            echo -e "${RED}Warning: Failed to install backend-local dependencies${RESET}"
        fi
        cd "$SCRIPT_DIR/room-measure"
    fi
    
    cd "$SCRIPT_DIR"
else
    echo -e "${RED}Error: room-measure directory not found${RESET}"
    exit 1
fi

# 3. Frontend 의존성 설치
echo
echo -e "${YELLOW}[3/4] Installing Frontend dependencies...${RESET}"

# room-measure frontend
if [ -f "room-measure/eunbi/frontend/package.json" ]; then
    cd room-measure/eunbi/frontend
    echo "Installing room-measure frontend dependencies..."
    if npm install &> /dev/null; then
        echo -e "${GREEN}✓ Room-measure frontend dependencies installed${RESET}"
    else
        echo -e "${RED}Warning: Failed to install room-measure frontend dependencies${RESET}"
    fi
    cd "$SCRIPT_DIR"
fi

# main frontend
if [ -f "room-measure/frontend-main/package.json" ]; then
    cd room-measure/frontend-main
    echo "Installing main frontend dependencies..."
    if npm install &> /dev/null; then
        echo -e "${GREEN}✓ Main frontend dependencies installed${RESET}"
    else
        echo -e "${RED}Warning: Failed to install main frontend dependencies${RESET}"
    fi
    cd "$SCRIPT_DIR"
fi

# 4. 포트 확인
echo
echo -e "${YELLOW}[4/4] Checking port availability...${RESET}"

ports=(3000 3010 4000 4010 7000 8080)
for port in "${ports[@]}"; do
    if command -v lsof &> /dev/null; then
        if lsof -i:$port &> /dev/null; then
            echo -e "${YELLOW}! Port $port is in use${RESET}"
        else
            echo -e "${GREEN}✓ Port $port is available${RESET}"
        fi
    elif command -v netstat &> /dev/null; then
        if netstat -tuln | grep ":$port " &> /dev/null; then
            echo -e "${YELLOW}! Port $port is in use${RESET}"
        else
            echo -e "${GREEN}✓ Port $port is available${RESET}"
        fi
    else
        echo -e "${YELLOW}! Cannot check port $port (lsof/netstat not found)${RESET}"
    fi
done

# 완료 메시지
echo
echo -e "${GREEN}=========================================="
echo "    Setup Complete!"
echo "==========================================${RESET}"
echo

echo -e "${BLUE}Next Steps:${RESET}"
echo "1. Edit API keys in environment files:"
echo "   - $SCRIPT_DIR/imggen-realistic-uv7000-triple/backend/.env"
echo "   - $SCRIPT_DIR/room-measure/.env"
echo
echo "2. Install PostgreSQL if not already installed"
echo
echo "3. Get API keys from:"
echo "   - Stability AI: https://platform.stability.ai/account/keys"
echo "   - Replicate: https://replicate.com/account/api-tokens"
echo
echo "4. Start services:"
echo "   - Image Generation: cd imggen-realistic-uv7000-triple/backend && uv run uvicorn main:app --port 7000"
echo "   - Room Backend: cd room-measure/eunbi/backend-cloud && uv run uvicorn main:app --port 3000"
echo "   - Room Frontend: cd room-measure/eunbi/frontend && npm run dev"
echo
echo -e "${YELLOW}⚠️  Remember to update the default passwords and secrets!${RESET}"
echo