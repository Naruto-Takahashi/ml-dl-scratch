import numpy as np
import matplotlib.pyplot as plt

class KMeansScratch:
    def __init__(self, k=3, max_iters=100):
        self.k = k                  # クラスタ数 (グループ数)
        self.max_iters = max_iters  # 最大繰り返し回数
        self.centroids = None       # 各クラスタの重心の位置 (k, 特徴量数)
        self.labels = None          # 各データ点が所属するクラスタのインデックス

    def _euclidean_distance(self, x1, x2):
        # 2点間のユークリッド距離を計算
        return np.sqrt(np.sum((x1 - x2) ** 2))

    def fit(self, X):
        self.n_samples, self.n_features = X.shape

        # 1. 重心 (centroids) の初期化: データの中からランダムに K 個を選択
        np.random.seed(42)
        random_idxs = np.random.choice(self.n_samples, self.k, replace=False)
        self.centroids = X[random_idxs]

        for _ in range(self.max_iters):
            # 2. 各データ点を「最も近い重心」のクラスタに割り当て
            old_centroids = self.centroids.copy()
            self.labels = self._assign_clusters(X)

            # 3. 新しい重心を、各クラスタに属するデータの平均値として再計算
            self.centroids = self._update_centroids(X)

            # もし重心が1ミリも動かなくなったら、完全に収束したとみなしてループを終了
            if np.all(old_centroids == self.centroids):
                break

    def _assign_clusters(self, X):
        labels = np.zeros(self.n_samples, dtype=int)
        for i, x in enumerate(X):
            # 各重心との距離を計算し、最も距離が短い重心のインデックスを取得
            distances = [self._euclidean_distance(x, centroid) for centroid in self.centroids]
            labels[i] = np.argmin(distances)
        return labels

    def _update_centroids(self, X):
        centroids = np.zeros((self.k, self.n_features))
        for idx in range(self.k):
            # クラスタ idx に属するすべてのデータを抽出
            cluster_data = X[self.labels == idx]
            if len(cluster_data) > 0:
                # 平均値を新しい重心の位置にする
                centroids[idx] = np.mean(cluster_data, axis=0)
            else:
                # 万が一データが1つも属さなかったクラスタは、元の位置のままにする
                centroids[idx] = self.centroids[idx]
        return centroids

    def predict(self, X):
        # 新しいデータに対して、最も近い重心のクラスタを判定
        return self._assign_clusters(X)

def generate_player_data():
    # 3つの異なるタイプのゲームプレイヤーのデータを生成します (合計120人)
    np.random.seed(42)

    # グループA: 無課金ヘビーユーザー (プレイ時間長く、課金はほぼ0)
    # 平均プレイ時間: 6時間, 平均課金: 150円
    group_a = np.random.normal(loc=[6.0, 150.0], scale=[1.0, 50.0], size=(40, 2))

    # グループB: 重課金ライトユーザー (プレイ時間短く、課金は多め)
    # 平均プレイ時間: 1.5時間, 平均課金: 2500円
    group_b = np.random.normal(loc=[1.5, 2500.0], scale=[0.5, 300.0], size=(40, 2))

    # グループC: 廃課金トッププレイヤー (プレイ時間長く、課金も非常に多い)
    # 平均プレイ時間: 7.0時間, 平均課金: 8000円
    group_c = np.random.normal(loc=[7.0, 8000.0], scale=[1.2, 800.0], size=(40, 2))

    # すべてのデータを結合
    X = np.vstack((group_a, group_b, group_c))

    # 課金額がマイナスにならないようにガード
    X[:, 1] = np.clip(X[:, 1], 0, None)

    return X

if __name__ == "__main__":
    X_raw = generate_player_data()

    # スケールが「時間 (0〜10)」と「円 (0〜10000)」で全く異なるため、
    # 距離計算が課金額に支配されないよう、ここでも「標準化」を適用します！
    X_mean, X_std = X_raw.mean(axis=0), X_raw.std(axis=0)
    X = (X_raw - X_mean) / X_std

    # K-Meansモデルの作成 (プレイヤーのクラスタ数を 3 に設定)
    model = KMeansScratch(k=3)
    model.fit(X)

    # 結果のプロット
    plt.figure(figsize=(10, 7))

    # 各クラスタを色分けして散布図に描画 (元のスケールに戻してプロットするので非常に見やすい！)
    colors = ["blue", "red", "green"]
    cluster_names = {
        0: "Casual Spenders",
        1: "Hardcore Gamers / Non-spenders",
        2: "Whales (High Spenders)"
    }

    for idx in range(model.k):
        cluster_data = X_raw[model.labels == idx]
        plt.scatter(cluster_data[:, 0], cluster_data[:, 1],
                    color=colors[idx], label=cluster_names[idx], s=70, alpha=0.7)

    # 重心 (Centroids) の位置も元のスケールに逆変換して星マークでプロット！
    centroids_raw = model.centroids * X_std + X_mean
    plt.scatter(centroids_raw[:, 0], centroids_raw[:, 1],
                color="black", marker="*", s=300, edgecolor="white", linewidth=2, label="Centroids (Center)")

    plt.title("Game Player Segmentation (K-Means Clustering Scratch)")
    plt.xlabel("Daily Play Time (Hours)")
    plt.ylabel("Monthly Spending (Yen)")
    plt.grid(True)
    plt.legend()
    plt.savefig("05_kmeans/player_segmentation.png")
    print("クラスタリング結果を '05_kmeans/player_segmentation.png' に保存しました！")
