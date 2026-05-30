import numpy as np
import matplotlib.pyplot as plt
class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        # 中間ノード (条件分岐) のパラメータ
        self.feature = feature      # 分割に使用する特徴量のインデックス (今回は常に0)
        self.threshold = threshold  # 分割の閾値 (境界値)
        self.left = left            # 左の子ノード (閾値以下のデータ)
        self.right = right          # 右の子ノード (閾値より大きいデータ)
        # 葉ノード (末端) のパラメータ
        self.value = value          # 最終的な予測クラス (0 or 1)
    def is_leaf(self):
        return self.value is not None

class DecisionTreeScratch:
    def __init__(self, max_depth=3, min_samples_split=2):
        self.max_depth = max_depth                  # 木の最大深さ (これ以上深くしない)
        self.min_samples_split = min_samples_split  # 分割を行うのに必要な最小データ数
        self.root = None                            # 木の根ノード
    def _entropy(self, y):
        # エントロピー (雑乱さ・不純度) の計算
        proportions = np.bincount(y) / len(y)
        # p log2(p) の計算 (p=0 のときは無視する)
        return -np.sum([p * np.log2(p) for p in proportions if p > 0])

    def _information_gain(self, X, y, threshold):
        # 親ノードのエントロピー
        parent_entropy = self._entropy(y)
        # データを閾値で左右にスプリット (分割)
        left_idxs = np.where(X <= threshold)[0]
        right_idxs = np.where(X > threshold)[0]

        if len(left_idxs) == 0 or len(right_idxs) == 0:
            return 0
        # 左右の子ノードのエントロピーを計算し、データ数で重み付け平均をとる
        n = len(y)
        n_l, n_r = len(left_idxs), len(right_idxs)
        e_l, e_r = self._entropy(y[left_idxs]), self._entropy(y[right_idxs])
        child_entropy = (n_l / n) * e_l + (n_r / n) * e_r
        # 情報利得 (Information Gain) = 分割によって減少したエントロピー量
        return parent_entropy - child_entropy

    def _best_split(self, X, y):
        best_gain = -1
        split_thresh = None
        # すべてのデータ点そのものを分割の閾値候補として、全探索で一番「綺麗に分かれる点」を探す
        for thresh in X:
            gain = self._information_gain(X, y, thresh)
            if gain > best_gain:
                best_gain = gain
                split_thresh = thresh

        return split_thresh, best_gain

    def _build_tree(self, X, y, depth=0):
        n_samples = len(X)
        n_labels = len(np.unique(y))

        # 【停止基準】 最大深さに達した、データ数が少なすぎる、または完全に一色に染まった(不純度0)場合
        if (depth >= self.max_depth or
            n_samples < self.min_samples_split or
            n_labels == 1):
            # 多数決で葉ノードの値を決定する
            leaf_value = np.argmax(np.bincount(y))
            return Node(value=leaf_value)

        # 最もエントロピーが下がる「最高の閾値」を探索
        threshold, gain = self._best_split(X, y)

        # これ以上分割しても綺麗にならない場合は、探索を打ち切って葉ノードにする
        if gain <= 0:
            leaf_value = np.argmax(np.bincount(y))
            return Node(value=leaf_value)

        # データを左右に分割して、再帰的に木を下に伸ばす (depth + 1)
        left_idxs = np.where(X <= threshold)[0]
        right_idxs = np.where(X > threshold)[0]

        left_child = self._build_tree(X[left_idxs], y[left_idxs], depth + 1)
        right_child = self._build_tree(X[right_idxs], y[right_idxs], depth + 1)

        return Node(feature=0, threshold=threshold, left=left_child, right=right_child)

    def fit(self, X, y):
        self.root = self._build_tree(X, y)

    def _predict_row(self, node, x):
        # 1つのデータ点に対して、条件分岐の木を上から辿る再帰関数
        if node.is_leaf():
            return node.value

        if x <= node.threshold:
            return self._predict_row(node.left, x)
        return self._predict_row(node.right, x)

    def predict(self, X):
        return np.array([self._predict_row(self.root, x) for x in X])

def generate_tree_data():
    np.random.seed(42)
    # 0点から100点までのテスト点数を100人分生成
    scores = np.random.uniform(20, 95, 100)

    # 決定木の特徴が最もよく出る「階段状の境界」を設定します。
    # 「45点未満」または「80点より大きい」人は追試(1)になり、その間(45点〜80点)の人は一発合格(0)とする
    y = np.where((scores < 45) | (scores > 80), 1, 0)

    # リアルさを出すため、5%の確率でノイズ(例外)を混ぜる
    noise = np.random.rand(100) < 0.05
    y = np.where(noise, 1 - y, y)

    return scores, y

if __name__ == "__main__":
    X, y = generate_tree_data()

    # 決定木モデルの作成と学習 (最大深さ 3)
    model = DecisionTreeScratch(max_depth=3)
    model.fit(X, y)

    # 結果のプロット
    plt.figure(figsize=(10, 6))
    plt.scatter(X[y == 0], y[y == 0], color="blue", label="Normal Pass (0)", s=60, alpha=0.7)
    plt.scatter(X[y == 1], y[y == 1], color="red", label="Makeup Exam (1)", s=60, alpha=0.7)

    # テストデータで全領域を予測して「階段状の境界」を描画
    X_test = np.linspace(20, 100, 500)
    y_pred = model.predict(X_test)

    # plt.step を用いて、木構造特有の「パキッとした条件分岐」の境界線を描きます
    plt.step(X_test, y_pred, color="green", where="mid", linewidth=3, label="Tree Decision Border")

    plt.title("Decision Tree Scratch (Exam Status Prediction)")
    plt.xlabel("Test Score")
    plt.ylabel("Exam Status (0: Pass, 1: Makeup)")
    plt.grid(True)
    plt.legend()
    plt.savefig("03_decision_tree/tree_result.png")
    print("決定木の結果グラフを '03_decision_tree/tree_result.png' に保存しました！")
