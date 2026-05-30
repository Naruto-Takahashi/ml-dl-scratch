import numpy as np
import matplotlib.pyplot as plt

class KNNScratch:
    def __init__(self, k=5):
        self.k = k
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        # KNNには「重みの更新」のような学習プロセスはありません！
        # 単に、後で予測するときのために訓練データをメモリに記憶（保持）しておくだけです。
        self.X_train = X
        self.y_train = y

    def _euclidean_distance(self, x1, x2):
        return np.sqrt(np.sum((x1 - x2) ** 2))

    def _predict_single(self, x):
        # 1. 新しいデータ点 x と、記憶しているすべての訓練データとの距離を計算
        distances = [self._euclidean_distance(x, x_train) for x_train in self.X_train]

        # 2. 距離が近い順にインデックスをソートし、最も近い K 個のラベルを取得
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]

        # 3. K個のラベルの中で多数決をとり、最も多いクラスを返す
        most_common = np.argmax(np.bincount(k_nearest_labels))
        return most_common

    def predict(self, X):
        # 複数の入力データに対して予測を実行
        return np.array([self._predict_single(x) for x in X])

def generate_movie_data():
    # 映画の「アクション度 (激しいシーンの割合)」と「ロマンス度 (恋愛シーンの割合)」
    # 0点から100点満点のスケールで、合計100作品分生成します
    np.random.seed(42)

    # アクション映画 (アクション度高く、ロマンス度低い) - クラス0
    action_movies = np.random.normal(loc=[75.0, 20.0], scale=[10.0, 8.0], size=(50, 2))
    y_action = np.zeros(50, dtype=int)

    # 恋愛映画 (アクション度低く、ロマンス度高い) - クラス1
    romance_movies = np.random.normal(loc=[25.0, 75.0], scale=[8.0, 10.0], size=(50, 2))
    y_romance = np.ones(50, dtype=int)

    X = np.vstack((action_movies, romance_movies))
    y = np.concatenate((y_action, y_romance))

    # 0〜100の範囲に収める
    X = np.clip(X, 0, 100)

    return X, y

if __name__ == "__main__":
    X_train, y_train = generate_movie_data()

    # KNNモデルの初期化 (K=5 に設定)
    model = KNNScratch(k=5)
    model.fit(X_train, y_train)

    # テスト対象となる新しい3つの「未知の新作映画」
    # 映画A: アクション度 60, ロマンス度 30 (ややアクション寄り)
    # 映画B: アクション度 40, ロマンス度 65 (ロマンス寄り)
    # 映画C: アクション度 45, ロマンス度 45 (中間・どっち？)
    test_movies = np.array([
        [60.0, 30.0],
        [40.0, 65.0],
        [45.0, 45.0]
    ])
    predictions = model.predict(test_movies)

    # 結果の可視化
    plt.figure(figsize=(10, 8))

    # 訓練データをプロット
    plt.scatter(X_train[y_train == 0][:, 0], X_train[y_train == 0][:, 1],
                color="red", label="Action Movies (0)", s=75, alpha=0.6)
    plt.scatter(X_train[y_train == 1][:, 0], X_train[y_train == 1][:, 1],
                color="blue", label="Romance Movies (1)", s=75, alpha=0.6)

    # 予測した新作映画をプロット
    genres = {0: "Action", 1: "Romance"}
    for i, test_movie in enumerate(test_movies):
        pred_label = predictions[i]
        plt.scatter(test_movie[0], test_movie[1],
                    color="green" if pred_label == 0 else "purple",
                    marker="X", s=250, edgecolor="black", linewidth=2,
                    label=f"New Movie {chr(65+i)}: Predicted {genres[pred_label]}")

    plt.title("Movie Genre Recommendation (KNN Classifier Scratch)")
    plt.xlabel("Action Scenes Score (0-100)")
    plt.ylabel("Romance Scenes Score (0-100)")
    plt.grid(True)
    plt.legend(loc="upper right")
    plt.savefig("06_knn/movie_recommendation.png")
    print("予測および可視化結果を '06_knn/movie_recommendation.png' に保存しました！")
