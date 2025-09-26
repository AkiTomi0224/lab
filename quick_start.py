#!/usr/bin/env python3
"""
YOLOv8技術図面検出システム - クイックスタートスクリプト
"""

import os
import sys
import subprocess
import platform
import time
from pathlib import Path

class SystemSetup:
    def __init__(self):
        self.base_dir = Path(__file__).parent
        self.venv_path = self.base_dir / 'venv'
        self.python_cmd = 'python3' if platform.system() != 'Windows' else 'python'
        
    def print_banner(self):
        """バナー表示"""
        print("=" * 60)
        print(" 🚢 3RDデッキ機関室機器配置可視化システム")
        print(" YOLOv8 AI搭載版 - セットアップ & 起動")
        print("=" * 60)
        
    def check_python(self):
        """Python バージョンチェック"""
        print("\n📌 Pythonバージョンチェック...")
        try:
            result = subprocess.run([self.python_cmd, '--version'], 
                                  capture_output=True, text=True)
            version = result.stdout.strip()
            print(f"  ✅ {version}")
            
            # バージョン番号を取得
            version_parts = version.split()[1].split('.')
            major = int(version_parts[0])
            minor = int(version_parts[1])
            
            if major < 3 or (major == 3 and minor < 8):
                print(f"  ⚠️  Python 3.8以上が必要です")
                return False
            return True
        except:
            print("  ❌ Pythonが見つかりません")
            return False
            
    def setup_venv(self):
        """仮想環境のセットアップ"""
        print("\n📌 仮想環境セットアップ...")
        
        if not self.venv_path.exists():
            print("  🔄 仮想環境を作成中...")
            subprocess.run([self.python_cmd, '-m', 'venv', str(self.venv_path)])
            print("  ✅ 仮想環境作成完了")
        else:
            print("  ✅ 仮想環境は既に存在します")
            
    def get_pip_cmd(self):
        """pip コマンドパスを取得"""
        if platform.system() == 'Windows':
            return str(self.venv_path / 'Scripts' / 'pip')
        else:
            return str(self.venv_path / 'bin' / 'pip')
            
    def get_python_venv_cmd(self):
        """仮想環境のPythonコマンドパスを取得"""
        if platform.system() == 'Windows':
            return str(self.venv_path / 'Scripts' / 'python')
        else:
            return str(self.venv_path / 'bin' / 'python')
            
    def install_dependencies(self):
        """依存関係のインストール"""
        print("\n📌 依存パッケージインストール...")
        
        pip_cmd = self.get_pip_cmd()
        python_venv = self.get_python_venv_cmd()
        
        # pipをアップグレード
        print("  🔄 pipをアップグレード中...")
        subprocess.run([pip_cmd, 'install', '--upgrade', 'pip'], 
                      stdout=subprocess.DEVNULL)
        
        # PyTorchのインストール（GPU対応チェック）
        print("  🔄 PyTorchをインストール中...")
        try:
            import torch
            if torch.cuda.is_available():
                print("  ✅ CUDA検出 - GPU版PyTorch")
        except:
            # CPU版をインストール
            subprocess.run([pip_cmd, 'install', 'torch', 'torchvision'],
                         stdout=subprocess.DEVNULL)
            
        # requirements_v2.txtが存在する場合はそれを使用
        req_file = self.base_dir / 'requirements_v2.txt'
        if not req_file.exists():
            req_file = self.base_dir / 'requirements.txt'
            
        if req_file.exists():
            print(f"  🔄 {req_file.name}から依存関係をインストール中...")
            subprocess.run([pip_cmd, 'install', '-r', str(req_file)],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            print("  ✅ 依存関係インストール完了")
        else:
            print("  ⚠️  requirements.txtが見つかりません")
            # 最小限のパッケージを直接インストール
            packages = [
                'flask', 'flask-cors', 'opencv-python', 
                'pillow', 'pdf2image', 'numpy', 'ultralytics'
            ]
            for pkg in packages:
                print(f"    🔄 {pkg}をインストール中...")
                subprocess.run([pip_cmd, 'install', pkg],
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                             
    def create_directories(self):
        """必要なディレクトリを作成"""
        print("\n📌 ディレクトリ構造を作成...")
        
        dirs = [
            'uploads', 'static/images', 'static/diagrams',
            'database', 'models', 'synthetic_data', 'cache'
        ]
        
        for dir_path in dirs:
            full_path = self.base_dir / dir_path
            full_path.mkdir(parents=True, exist_ok=True)
            
        print("  ✅ ディレクトリ作成完了")
        
    def download_yolo_model(self):
        """YOLOv8モデルのダウンロード"""
        print("\n📌 YOLOv8モデルチェック...")
        
        model_path = self.base_dir / 'models' / 'yolov8m.pt'
        if not model_path.exists():
            print("  🔄 YOLOv8モデルをダウンロード中...")
            python_venv = self.get_python_venv_cmd()
            
            download_script = """
from ultralytics import YOLO
import os
os.makedirs('models', exist_ok=True)
model = YOLO('yolov8m.pt')
print('モデルダウンロード完了')
"""
            subprocess.run([python_venv, '-c', download_script],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            print("  ✅ YOLOv8モデルダウンロード完了")
        else:
            print("  ✅ YOLOv8モデル確認済み")
            
    def start_backend(self):
        """バックエンドサーバーを起動"""
        print("\n📌 バックエンドサーバーを起動...")
        
        python_venv = self.get_python_venv_cmd()
        
        # 使用するバックエンドファイルを決定
        backend_v2 = self.base_dir / 'backend' / 'app_v2.py'
        backend_v1 = self.base_dir / 'backend' / 'app.py'
        
        if backend_v2.exists():
            backend_file = backend_v2
            print("  🚀 最新版バックエンド (app_v2.py) を起動中...")
        elif backend_v1.exists():
            backend_file = backend_v1
            print("  🚀 標準バックエンド (app.py) を起動中...")
        else:
            print("  ❌ バックエンドファイルが見つかりません")
            return None
            
        # バックエンドプロセスを起動
        process = subprocess.Popen([python_venv, str(backend_file)],
                                 stdout=subprocess.PIPE,
                                 stderr=subprocess.PIPE)
        
        # 起動を待つ
        time.sleep(3)
        
        # プロセスが実行中か確認
        if process.poll() is None:
            print("  ✅ バックエンドサーバー起動成功")
            return process
        else:
            print("  ❌ バックエンドサーバー起動失敗")
            return None
            
    def open_browser(self):
        """ブラウザを開く"""
        import webbrowser
        url = "http://localhost:8000"
        
        print(f"\n🌐 ブラウザで開いています: {url}")
        
        # 新しいHTMLファイルが存在する場合はそれを使用
        index_v2 = self.base_dir / 'frontend' / 'index_v2.html'
        if index_v2.exists():
            print("  📄 最新版UI (index_v2.html) を使用")
        
        try:
            webbrowser.open(url)
        except:
            print(f"  ⚠️  ブラウザを手動で開いてください: {url}")
            
    def run(self):
        """メイン実行"""
        self.print_banner()
        
        # チェックと準備
        if not self.check_python():
            print("\n❌ セットアップを中止します")
            sys.exit(1)
            
        self.setup_venv()
        self.install_dependencies()
        self.create_directories()
        self.download_yolo_model()
        
        # サーバー起動
        process = self.start_backend()
        
        if process:
            print("\n" + "=" * 60)
            print(" ✨ システム稼働中！")
            print(" 🌐 URL: http://localhost:8000")
            print(" 🛑 停止: Ctrl+C")
            print("=" * 60)
            
            # ブラウザを開く
            self.open_browser()
            
            # プロセスを待機
            try:
                process.wait()
            except KeyboardInterrupt:
                print("\n\n📌 シャットダウン中...")
                process.terminate()
                process.wait()
                print("  ✅ システム停止完了")
        else:
            print("\n❌ システム起動失敗")
            sys.exit(1)

if __name__ == "__main__":
    setup = SystemSetup()
    setup.run()
