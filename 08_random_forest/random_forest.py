import numpy as np
import matplotlib.pyplot as plt

# --- 以前作成した自作の決定木ノード・クラスをここに記述 ---
class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value
    def is_leaf(self):
        return self.value is not None
class DecisionTreeScratch:
    def __init__(self, max_depth=3, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None
    def _entropy(self, y):
        proportions = np.bincount(y) / len(y)
        return -np.sum([p * np.log2(p) for p in proportions if p > 0])
    def _information_gain(self, X, y, threshold):
        parent_entropy = self._entropy(y)
        left_idxs = np.where(X <= threshold)[0]
        right_idxs = np.where(X > threshold)[0]
        if len(left_idxs) == 0 or len(right_idxs) == 0:
            return 0
        n = len(y)
        n_l, n_r = len(left_idxs), len(right_idxs)
        e_l, e_r = self._entropy(y[left_idxs]), self._entropy(y[right_idxs])
        child_entropy = (n_l / n) * e_l + (n_r / n) * e_r
        return parent_entropy - child_entropy
    def _best_split(self, X, y):
        best_gain = -1
        split_thresh = None
        for thresh in X:
            gain = self._information_gain(X, y, thresh)
            if gain > best_gain:
                best_gain = gain
                split_thresh = thresh
        return split_thresh, best_gain
    def _build_tree(self, X, y, depth=0):
        n_samples = len(X)
        n_labels = len(np.unique(y))
        if (depth >= self.max_depth or n_samples < self.min_samples_split or n_labels == 1):
            leaf_value = np.argmax(np.bincount(y))
            return Node(value=leaf_value)
        threshold, gain = self._best_split(X, y)
        if gain <= 0:
            leaf_value = np.argmax(np.bincount(y))
            return Node(value=leaf_value)
        left_idxs = np.where(X <= threshold)[0]
        right_idxs = np.where(X > threshold)[0]
        left_child = self._build_tree(X[left_idxs], y[left_idxs], depth + 1)
        right_child = self._build_tree(X[right_idxs], y[right_idxs], depth + 1)
        return Node(feature=0, threshold=threshold, left=left_child, right=right_child)
    def fit(self, X, y):
        self.root = self._build_tree(X, y)

    def _predict_row(self, node, x):
        if node.is_leaf():
            return node.value
        if x <= node.threshold:
            return self._predict_row(node.left, x)
        return self._predict_row(node.right, x)
    def predict(self, X):
        return np.array([self._predict_row(self.root, x) for x in X])
# --- ランダムフォレスト (森) のスクラッチ実装 ---
class RandomForestScratch:
    def __init__(self, n_trees=10, max_depth=3, min_samples_split=2):
        self.n_trees = n_trees                      # 森を構成する木の数 (並列度)
        self.max_depth = max_depth                  # 各木の最大深さ
        self.min_samples_split = min_samples_split  # 各木の分割最小サンプル数
        self.trees = []                             # 木を格納するリスト

    def fit(self, X, y):
        self.trees = []
        n_samples = len(X)

        for _ in range(self.n_trees):
            # 1. ブートストラップサンプリング (重複を許して元のデータ数分だけランダムに抽出)
            bootstrap_idxs = np.random.choice(n_samples, n_samples, replace=True)
            X_bootstrap = X[bootstrap_idxs]
            y_bootstrap = y[bootstrap_idxs]

            # 2. 決定木を個別に初期化して学習させる
            tree = DecisionTreeScratch(max_depth=self.max_depth, min_samples_split=self.min_samples_split)
            tree.fit(X_bootstrap, y_bootstrap)
            self.trees.append(tree)

    def predict(self, X):
        # 3. すべての木に予測を行わせる
        # 各行が1本の木の予測結果になる行列を生成 (n_trees, N)
        tree_preds = np.array([tree.predict(X) for tree in self.trees])

        # 4. データ点ごとに縦軸で「多数決」をとる
        # 転置して (N, n_trees) にし、各行で最頻値を求める
        tree_preds_T = tree_preds.T
        final_preds = []
        for row in tree_preds_T:
            most_common = np.argmax(np.bincount(row))
            final_preds.append(most_common)

        return np.array(final_preds)

def generate_complex_data():
    # 決定木1本では表現できない「中心にドーナツ（円）があり、外側に別のクラスがある」複雑な分布データ
    np.random.seed(42)
    X = np.random.uniform(-4, 4, 120)

    # 中心からの距離 (円の半径) に基づいてラベル付け
    # 半径が 1.2 以上かつ 2.8 以下の範囲をクラス 1 (合格)、それ以外をクラス 0 (不合格) とする
    y = np.where((np.abs(X) >= 1.2) & (np.abs(X) <= 2.8), 1, 0)

    # リアルなノイズ (10%の確率で例外を混ぜる)
    noise = np.random.rand(120) < 0.1
    y = np.where(noise, 1 - y, y)

    return X, y

if __name__ == "__main__":
    X_raw, y = generate_complex_data()

    # 1. 比較用：自作の決定木1本だけ (max_depth=3)
    single_tree = DecisionTreeScratch(max_depth=3)
    single_tree.fit(X_raw, y)

    # 2. 本命：自作のランダムフォレスト (決定木を「15本」束ねた森)
    forest = RandomForestScratch(n_trees=15, max_depth=3)
    forest.fit(X_raw, y)

    # 結果のプロット
    plt.figure(figsize=(12, 5))
    X_test = np.linspace(-4, 4, 500)

    # グラフ1: 決定木1本の境界線 (過学習したり、表現しきれなかったりする)
    plt.subplot(1, 2, 1)
    plt.scatter(X_raw[y == 0], y[y == 0], color="blue", label="Class 0 (Border Out)", alpha=0.6)
    plt.scatter(X_raw[y == 1], y[y == 1], color="red", label="Class 1 (Inside Ring)", alpha=0.6)
    plt.step(X_test, single_tree.predict(X_test), color="orange", where="mid", linewidth=3, label="Single Tree Line")
    plt.title("Single Decision Tree (Weak Classifier)")
    plt.xlabel("Feature x")
    plt.ylabel("Prediction")
    plt.grid(True)
    plt.legend()

    # グラフ2: ランダムフォレスト（15本の森の多数決）の境界線 (非常に堅牢で、ドーナツを完璧に捉える！)
    plt.subplot(1, 2, 2)
    plt.scatter(X_raw[y == 0], y[y == 0], color="blue", label="Class 0 (Border Out)", alpha=0.6)
    plt.scatter(X_raw[y == 1], y[y == 1], color="red", label="Class 1 (Inside Ring)", alpha=0.6)
    plt.step(X_test, forest.predict(X_test), color="green", where="mid", linewidth=3, label="Random Forest Line")
    plt.title("Random Forest Scratch (15 Trees Assembly)")
    plt.xlabel("Feature x")
    plt.ylabel("Prediction")
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.savefig("08_random_forest/forest_result.png")
    print("森の可視化結果を '08_random_forest/forest_result.png' に保存しました！")
