# 図面のデジタル化システム

機関室図面（3RD DECK PLAN）をデジタル化し、機器の位置を特定・ハイライト表示するシステムです。

## 機能

1. **機器登録DB機能**: 機器名と複数の機器画像を登録
2. **図面情報取得機能**: PDF図面のアップロードと表示、座標取得
3. **機器選択機能**: 登録済み機器の選択とチェックボックス表示
4. **可視化機能**: 選択した機器を図面上でハイライト表示

## セットアップ

### 前提条件
- Python 3.8以上
- pip

### インストール

```bash
# プロジェクトディレクトリに移動
cd diagram-digitization-system

# Pythonパッケージをインストール
pip install -r requirements.txt

# 必要なシステムパッケージ（画像処理用）
# Ubuntu/Debian:
sudo apt-get install python3-opencv poppler-utils

# macOS:
brew install poppler
```

### 実行

```bash
# バックエンドサーバーを起動
cd backend
python app.py

# ブラウザでフロントエンドを開く
cd ../frontend
# ローカルサーバーで開く（例：Live Server拡張機能）
# または直接index.htmlをブラウザで開く
```

サーバーが起動したら、http://localhost:5000 でAPIにアクセスできます。

## 使用方法

1. **機器登録**: 機器名と画像をアップロードして機器をデータベースに登録
2. **図面アップロード**: PDF形式の機関室図面をアップロード
3. **可視化**: アップロードした図面を表示し、機器を選択してハイライト表示

## ディレクトリ構成

```
diagram-digitization-system/
├── backend/           # Flask APIサーバー
│   └── app.py
├── frontend/          # Webインターフェース
│   ├── index.html
│   ├── style.css
│   └── script.js
├── static/            # 静的ファイル
│   ├── images/        # 機器画像
│   └── diagrams/      # 図面画像
├── uploads/           # アップロードファイル
├── database/          # SQLiteデータベース
└── requirements.txt   # Python依存関係
```

## API エンドポイント

- `GET /api/equipment` - 登録済み機器一覧取得
- `POST /api/equipment` - 機器登録
- `GET /api/diagrams` - アップロード済み図面一覧取得
- `POST /api/diagrams` - 図面アップロード
- `GET /api/health` - ヘルスチェック