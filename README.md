# Machine Learning & Deep Learning From Scratch

本リポジトリは，機械学習およびディープラーニングの代表的なアルゴリズムを外部の機械学習ライブラリ（scikit-learnなど）を使わずに，NumPyとMatplotlibを中心にPythonでスクラッチ実装したプロジェクトです．

アルゴリズムの内部数式や更新ルールをコードレベルで直感的に理解できるように設計されています．

---

## 開発環境とセットアップ

本リポジトリは，再現可能な開発環境として **Nix**（`shell.nix`）をサポートしています．また，通常のPython環境でもインストールパッケージをセットアップすることで実行可能です．

### 動作環境
- **Python**: v3.12
- **主要ライブラリ**:
  - `numpy`: 数値計算・行列計算
  - `matplotlib`: データの可視化・グラフ描画
  - `jupyterlab`: インタラクティブな実行環境

---

### 環境構築方法

#### 1. Nix を使用する場合（推奨）
Nixがインストールされている環境であれば，リポジトリのルートディレクトリで以下のコマンドを実行するだけで，必要なツールとライブラリが完全にセットアップされたシェルが起動します．

```bash
nix-shell
```

起動すると，自動的にPythonおよびJupyter Labが利用可能な状態になります．

#### 2. 通常の Python 環境を使用する場合
一般的な仮想環境（`venv` または `conda`）を作成し，必要な依存ライブラリを手動でインストールしてください．

```bash
# 仮想環境の作成と有効化 (例: venv)
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate  # Windows

# 必要なライブラリのインストール
pip install numpy matplotlib jupyterlab
```

---

## 実行方法

各アルゴリズムは，個別ディレクトリ配下に完結した実行スクリプトとして配置されています．ルートディレクトリから各スクリプトを直接実行することで，モデルの学習が進み，結果がグラフ画像（`.png`）として保存されます．

```bash
# 実行例 (線形回帰の実行)
python 01_linear_regression/linear_regression.py
```

実行が完了すると，各ディレクトリ内に実行結果をプロットした可視化画像が生成されます．

---

## アルゴリズム一覧と詳細説明

各アルゴリズムの実装詳細，数式背景，使用した人工データセット，および得られた結果についての詳しい説明は，以下の個別ドキュメントを参照してください．

| ディレクトリ | アルゴリズム | 対象タスク | 詳細解説リンク |
| :--- | :--- | :--- | :--- |
| `01_linear_regression` | **線形回帰 (Linear Regression)** | 回帰（連続値の予測） | [詳細説明はこちら](./01_linear_regression/README.md) |
| `02_logistic_regression` | **ロジスティック回帰 (Logistic Regression)** | 二値分類 | [詳細説明はこちら](./02_logistic_regression/README.md) |
| `03_decision_tree` | **決定木 (Decision Tree)** | 非線形二値分類 | [詳細説明はこちら](./03_decision_tree/README.md) |
| `04_deep_learning` | **多層パーセプトロン (MLP)** | 非線形回帰 | [詳細説明はこちら](./04_deep_learning/README.md) |
| `05_kmeans` | **K-Meansクラスタリング** | クラスター分析（非監視学習） | [詳細説明はこちら](./05_kmeans/README.md) |
| `06_knn` | **k近傍法 (k-NN)** | 分類（インスタンス学習） | [詳細説明はこちら](./06_knn/README.md) |
| `07_naive_bayes` | **単純ベイズ分類器 (Naive Bayes)** | テキスト二値分類（スパムフィルタ） | [詳細説明はこちら](./07_naive_bayes/README.md) |
| `08_random_forest` | **ランダムフォレスト (Random Forest)** | アンサンブル非線形分類 | [詳細説明はこちら](./08_random_forest/README.md) |
