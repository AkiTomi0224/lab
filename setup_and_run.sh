#!/bin/bash

# Advanced Technical Drawing Detection System - Setup & Launch Script

echo "======================================="
echo " 技術図面検出システム セットアップ"
echo " YOLOv8 & 最新技術実装版"
echo "======================================="

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check Python version
echo -e "${YELLOW}Pythonバージョンチェック...${NC}"
python_version=$(python3 --version 2>&1 | grep -Po '(?<=Python )[\d.]+')
required_version="3.8"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" = "$required_version" ]; then 
    echo -e "${GREEN}✓ Python $python_version${NC}"
else
    echo -e "${RED}✗ Python 3.8以上が必要です${NC}"
    exit 1
fi

# Create virtual environment if not exists
if [ ! -d "venv" ]; then
    echo -e "${YELLOW}仮想環境を作成中...${NC}"
    python3 -m venv venv
    echo -e "${GREEN}✓ 仮想環境作成完了${NC}"
fi

# Activate virtual environment
echo -e "${YELLOW}仮想環境を有効化...${NC}"
source venv/bin/activate

# Install/Update dependencies
echo -e "${YELLOW}依存パッケージをインストール中...${NC}"
pip install --upgrade pip > /dev/null 2>&1

# Install PyTorch with CUDA support if available
if command -v nvidia-smi &> /dev/null; then
    echo -e "${GREEN}✓ CUDA検出 - GPU版PyTorchをインストール${NC}"
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118 > /dev/null 2>&1
else
    echo -e "${YELLOW}CPU版PyTorchをインストール${NC}"
    pip install torch torchvision > /dev/null 2>&1
fi

# Install other requirements
echo -e "${YELLOW}その他のパッケージをインストール中...${NC}"
pip install -r requirements_v2.txt > /dev/null 2>&1

# System dependencies check
echo -e "${YELLOW}システム依存関係チェック...${NC}"

# Check for poppler-utils (for PDF processing)
if command -v pdftoppm &> /dev/null; then
    echo -e "${GREEN}✓ poppler-utils${NC}"
else
    echo -e "${YELLOW}poppler-utilsをインストールしてください:${NC}"
    echo "  Ubuntu/Debian: sudo apt-get install poppler-utils"
    echo "  macOS: brew install poppler"
fi

# Check for OpenCV dependencies
if pkg-config --exists opencv4 2>/dev/null; then
    echo -e "${GREEN}✓ OpenCV${NC}"
else
    echo -e "${YELLOW}OpenCVライブラリが見つかりません（Pythonパッケージは自動インストールされます）${NC}"
fi

# Download YOLOv8 model if not exists
echo -e "${YELLOW}YOLOv8モデルチェック...${NC}"
if [ ! -f "models/yolov8m.pt" ]; then
    echo -e "${YELLOW}YOLOv8モデルをダウンロード中...${NC}"
    mkdir -p models
    python3 -c "from ultralytics import YOLO; model = YOLO('yolov8m.pt'); model.save('models/yolov8m.pt')" > /dev/null 2>&1
    echo -e "${GREEN}✓ YOLOv8モデルダウンロード完了${NC}"
else
    echo -e "${GREEN}✓ YOLOv8モデル存在確認${NC}"
fi

# Create necessary directories
echo -e "${YELLOW}ディレクトリ構造を作成中...${NC}"
mkdir -p uploads static/images static/diagrams database models synthetic_data cache
echo -e "${GREEN}✓ ディレクトリ作成完了${NC}"

# Launch the application
echo -e "${GREEN}=======================================${NC}"
echo -e "${GREEN} システム起動準備完了！${NC}"
echo -e "${GREEN}=======================================${NC}"

# Check which backend to use
if [ -f "backend/app_v2.py" ]; then
    echo -e "${YELLOW}最新版バックエンド (app_v2.py) を起動中...${NC}"
    cd backend
    python app_v2.py &
    backend_pid=$!
else
    echo -e "${YELLOW}標準バックエンド (app.py) を起動中...${NC}"
    cd backend
    python app.py &
    backend_pid=$!
fi

cd ..

# Wait for backend to start
echo -e "${YELLOW}バックエンドの起動を待機中...${NC}"
sleep 5

# Check if backend is running
if ps -p $backend_pid > /dev/null; then
    echo -e "${GREEN}✓ バックエンド起動成功 (PID: $backend_pid)${NC}"
    
    # Open browser
    echo -e "${GREEN}=======================================${NC}"
    echo -e "${GREEN} システム稼働中！${NC}"
    echo -e "${GREEN} ブラウザで開く: http://localhost:8000${NC}"
    echo -e "${GREEN} 停止: Ctrl+C${NC}"
    echo -e "${GREEN}=======================================${NC}"
    
    # Try to open browser automatically
    if command -v xdg-open &> /dev/null; then
        xdg-open http://localhost:8000
    elif command -v open &> /dev/null; then
        open http://localhost:8000
    fi
    
    # Wait for interrupt
    trap "echo -e '\n${YELLOW}シャットダウン中...${NC}'; kill $backend_pid; exit" INT
    wait $backend_pid
    
else
    echo -e "${RED}✗ バックエンド起動失敗${NC}"
    echo -e "${RED}ログを確認してください: backend/server.log${NC}"
    exit 1
fi
