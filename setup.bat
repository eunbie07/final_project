@echo off
setlocal enabledelayedexpansion

echo ==========================================
echo    Final Project Environment Setup
echo ==========================================
echo.

:: 색상 설정
set "green=[92m"
set "red=[91m"
set "yellow=[93m"
set "blue=[94m"
set "reset=[0m"

:: 현재 디렉토리 확인
echo %blue%Current directory: %CD%%reset%
echo.

:: 1. imggen-realistic-uv7000-triple 설정
echo %yellow%[1/4] Setting up imggen-realistic-uv7000-triple...%reset%
cd /d "%~dp0\imggen-realistic-uv7000-triple\backend" 2>nul
if errorlevel 1 (
    echo %red%Error: imggen-realistic-uv7000-triple/backend directory not found%reset%
    pause
    exit /b 1
)

:: .env 파일 생성
if not exist ".env" (
    echo Creating .env file...
    echo # Stability AI API Key> .env
    echo # Get your key from: https://platform.stability.ai/account/keys>> .env
    echo STABILITY_API_KEY=sk-your-stability-key-here>> .env
    echo.>> .env
    echo # Replicate API Token>> .env  
    echo # Get your token from: https://replicate.com/account/api-tokens>> .env
    echo REPLICATE_API_TOKEN=r8_your-replicate-token-here>> .env
    echo.>> .env
    echo # Google Cloud Service Account ^(Optional^)>> .env
    echo GCP_SERVICE_ACCOUNT_JSON_PATH=./service-account.json>> .env
    echo %green%✓ Created imggen-realistic-uv7000-triple/backend/.env%reset%
) else (
    echo %yellow%! .env file already exists%reset%
)

:: Python 의존성 설치 시도
echo Installing Python dependencies...
uv pip install -e . >nul 2>&1
if errorlevel 1 (
    echo %red%Warning: Failed to install Python dependencies. Make sure 'uv' is installed.%reset%
    echo Run: pip install uv
) else (
    echo %green%✓ Python dependencies installed%reset%
)

:: 2. room-measure 설정
echo.
echo %yellow%[2/4] Setting up room-measure...%reset%
cd /d "%~dp0\room-measure" 2>nul
if errorlevel 1 (
    echo %red%Error: room-measure directory not found%reset%
    pause
    exit /b 1
)

:: 메인 .env 파일 생성
if not exist ".env" (
    echo Creating main .env file...
    echo # PostgreSQL Database Settings> .env
    echo POSTGRES_USER=postgres>> .env
    echo POSTGRES_PASSWORD=change_this_secure_password_123>> .env
    echo POSTGRES_DB=room_measure>> .env
    echo POSTGRES_HOST=localhost>> .env
    echo POSTGRES_PORT=5432>> .env
    echo DATABASE_URL=postgresql://postgres:change_this_secure_password_123@localhost:5432/room_measure>> .env
    echo.>> .env
    echo # JWT Authentication ^(Change this in production^)>> .env
    echo JWT_SECRET=your-super-secret-jwt-key-minimum-32-characters-long-change-this>> .env
    echo.>> .env
    echo # Environment>> .env
    echo NODE_ENV=development>> .env
    echo %green%✓ Created room-measure/.env%reset%
) else (
    echo %yellow%! Main .env file already exists%reset%
)

:: Backend cloud 설정
cd /d "%~dp0\room-measure\eunbi\backend-cloud" 2>nul
if exist "." (
    echo Installing backend-cloud dependencies...
    uv pip install -e . >nul 2>&1
    if errorlevel 1 (
        echo %red%Warning: Failed to install backend-cloud dependencies%reset%
    ) else (
        echo %green%✓ Backend-cloud dependencies installed%reset%
    )
)

:: Backend local 설정  
cd /d "%~dp0\room-measure\eunbi\backend-local" 2>nul
if exist "." (
    echo Installing backend-local dependencies...
    uv pip install -e . >nul 2>&1
    if errorlevel 1 (
        echo %red%Warning: Failed to install backend-local dependencies%reset%
    ) else (
        echo %green%✓ Backend-local dependencies installed%reset%
    )
)

:: 3. Frontend 의존성 설치
echo.
echo %yellow%[3/4] Installing Frontend dependencies...%reset%

:: room-measure frontend
cd /d "%~dp0\room-measure\eunbi\frontend" 2>nul
if exist "package.json" (
    echo Installing room-measure frontend dependencies...
    npm install >nul 2>&1
    if errorlevel 1 (
        echo %red%Warning: Failed to install room-measure frontend dependencies%reset%
    ) else (
        echo %green%✓ Room-measure frontend dependencies installed%reset%
    )
)

:: main frontend
cd /d "%~dp0\room-measure\frontend-main" 2>nul
if exist "package.json" (
    echo Installing main frontend dependencies...
    npm install >nul 2>&1
    if errorlevel 1 (
        echo %red%Warning: Failed to install main frontend dependencies%reset%
    ) else (
        echo %green%✓ Main frontend dependencies installed%reset%
    )
)

:: 4. 포트 확인
echo.
echo %yellow%[4/4] Checking port availability...%reset%

set "ports=3000 3010 4000 4010 7000 8080"
for %%p in (%ports%) do (
    netstat -ano | findstr :%%p >nul 2>&1
    if errorlevel 1 (
        echo %green%✓ Port %%p is available%reset%
    ) else (
        echo %yellow%! Port %%p is in use%reset%
    )
)

:: 완료 메시지
echo.
echo %green%==========================================
echo    Setup Complete!
echo ==========================================%reset%
echo.
echo %blue%Next Steps:%reset%
echo 1. Edit API keys in environment files:
echo    - %~dp0imggen-realistic-uv7000-triple\backend\.env
echo    - %~dp0room-measure\.env
echo.
echo 2. Install PostgreSQL if not already installed
echo.
echo 3. Get API keys from:
echo    - Stability AI: https://platform.stability.ai/account/keys
echo    - Replicate: https://replicate.com/account/api-tokens
echo.
echo 4. Start services:
echo    - Image Generation: cd imggen-realistic-uv7000-triple\backend ^&^& uv run uvicorn main:app --port 7000
echo    - Room Backend: cd room-measure\eunbi\backend-cloud ^&^& uv run uvicorn main:app --port 3000
echo    - Room Frontend: cd room-measure\eunbi\frontend ^&^& npm run dev
echo.
echo %yellow%⚠️  Remember to update the default passwords and secrets!%reset%
echo.

cd /d "%~dp0"
pause