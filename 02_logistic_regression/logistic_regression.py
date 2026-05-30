import numpy as np
import matplotlib.pyplot as plt

class LogisticRegressionScratch:
    def __init__(self, lr=0.1, epochs=200):
        self.lr = lr
        self.epochs = epochs
        self.w = 0.0
        self.b = 0.0
        self.loss_history = []

    def _sigmoid(self, z):
        # 数値的に安定したシグモイド関数の実装 (オーバーフロー対策)
        return 1.0 / (1.0 + np.exp(-np.clip(z, -250, 250)))

    def fit(self, X, y):
        N = len(X)
        for epoch in range(self.epochs):
            # 1. 順伝播 (z の計算からシグモイド関数を通す)
            z = self.w * X + self.b
            p = self._sigmoid(z)

            # 2. バイナリ交差エントロピー損失の計算 (0除算を防ぐため微小な値 1e-15 を加算)
            loss = -np.mean(y * np.log(p + 1e-15) + (1.0 - y) * np.log(1.0 - p + 1e-15))
            self.loss_history.append(loss)

            # 3. 勾配の手動計算
            dw = np.mean((p - y) * X)
            db = np.mean(p - y)

            # 4. パラメータの更新
            self.w -= self.lr * dw
            self.b -= self.lr * db

            if (epoch + 1) % 20 == 0:
                print(f"Epoch {epoch+1:3d} | Loss: {loss:.4f} | w: {self.w:.4f} | b: {self.b:.4f}")

    def predict_proba(self, X):
        z = self.w * X + self.b
        return self._sigmoid(z)

    def predict(self, X):
        # 確率が 0.5 以上ならクラス 1, そうでなければクラス 0 と判定
        return (self.predict_proba(X) >= 0.5).astype(int)

def generate_classification_data():
    np.random.seed(42)
    # 0点から100点までのテスト点数を100人分生成
    scores = np.random.uniform(20, 95, 100)

    # 55点を合格(1)/不合格(0)の境界とする
    # 55点より上なら合格しやすく、下なら不合格になりやすい確率分布を作る
    z = 0.2 * (scores - 55) # 傾き0.2
    probs = 1.0 / (1.0 + np.exp(-z))

    # 確率に基づいて合否(0:不合格, 1:合格)をランダム決定
    pass_fail = (np.random.rand(100) < probs).astype(int)

    return scores, pass_fail

if __name__ == "__main__":
    # データ生成
    X_raw, y = generate_classification_data()

    # データの標準化 (学習率の調整をしやすくし、学習を圧倒的に安定させるため)
    X_mean, X_std = X_raw.mean(), X_raw.std()
    X = (X_raw - X_mean) / X_std

    # モデルの学習 (標準化されたため、学習率 lr=0.5 程度で超高速に収束します)
    print("--- テスト点数から合否を予測するモデルの学習開始 ---")
    model = LogisticRegressionScratch(lr=0.5, epochs=300)
    model.fit(X, y)
    print("--- 学習終了 ---")

    # 結果のプロット
    plt.figure(figsize=(12, 5))

    # グラフ1: テスト点数と合否予測確率
    plt.subplot(1, 2, 1)
    plt.scatter(X_raw[y == 0], y[y == 0], color="blue", label="Fail (0)", alpha=0.7, s=50)
    plt.scatter(X_raw[y == 1], y[y == 1], color="red", label="Pass (1)", alpha=0.7, s=50)

    # 滑らかな予測確率曲線を描画 (描画の際は、元のスケールから予測します)
    X_test_raw = np.linspace(20, 100, 300)
    X_test_scaled = (X_test_raw - X_mean) / X_std
    plt.plot(X_test_raw, model.predict_proba(X_test_scaled), color="green", linewidth=3,
label="Pass Probability")

    # 決定境界 (合格確率がちょうど50%になるテストの素点)
    # w * X_scaled + b = 0 => X_scaled = -b/w
    # 元のスケールに戻す: X_raw = X_scaled * X_std + X_mean
    decision_boundary_scaled = -model.b / model.w
    decision_boundary = decision_boundary_scaled * X_std + X_mean

    plt.axvline(x=decision_boundary, color="black", linestyle="--", linewidth=2,
                label=f"Pass/Fail Border: {decision_boundary:.1f} scores")

    plt.title("Exam Pass/Fail Prediction (Logistic Regression)")
    plt.xlabel("Test Score")
    plt.ylabel("Probability of Passing")
    plt.grid(True)
    plt.legend()

    # グラフ2: 学習の進捗に伴う交差エントロピー損失の低下
    plt.subplot(1, 2, 2)
    plt.plot(model.loss_history, color="purple", linewidth=2)
    plt.title("Learning Curve (BCE Loss)")
    plt.xlabel("Epoch")
    plt.ylabel("BCE Loss")
    plt.grid(True)

    plt.tight_layout()
    plt.savefig("02_logistic_regression/classification_exam.png")
    print("結果グラフを '02_logistic_regression/classification_exam.png' に保存しました！")
