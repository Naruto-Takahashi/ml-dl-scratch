import numpy as np
import matplotlib.pyplot as plt

class LinearRegressionScratch:
    def __init__(self, lr=0.05, epochs=100):
        self.lr = lr          # 学習率 (Learning Rate)
        self.epochs = epochs  # エポック数 (繰り返し回数)
        self.w = 0.0          # 重み (weight)
        self.b = 0.0          # バイアス (bias)
        self.loss_history = []# 損失の履歴を保存するリスト

    def fit(self, X, y):
        N = len(X)
        for epoch in range(self.epochs):
            # 1. 順伝播 (予測値の計算)
            y_pred = self.w * X + self.b

            # 2. 損失 (MSE) の計算
            loss = np.mean((y - y_pred) ** 2) / 2.0
            self.loss_history.append(loss)

            # 3. 勾配 (Gradient) の手動計算 (偏微分)
            dw = (-2 / N) * np.sum(X * (y - y_pred)) / 2.0
            db = (-2 / N) * np.sum(y - y_pred) / 2.0

            # 4. パラメータの更新
            self.w -= self.lr * dw
            self.b -= self.lr * db

            # 10エポックごとにログを出力
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1:3d} | Loss: {loss:.4f} | w: {self.w:.4f} | b: {self.b:.4f}")

    def predict(self, X):
        return self.w * X + self.b

def generate_data():
    np.random.seed(42)
    # 150cm から 185cm までの身長データを100人分生成
    heights = np.random.uniform(150, 185, 100)
    # 体重は適度なノイズ(個人の体格差)を乗せて計算
    noise = np.random.normal(0, 5, 100) # 平均0, 標準偏差5kgのバラつき
    weights = (heights - 100) * 0.9 + noise

    return heights, weights
if __name__ == "__main__":
    # 1. データの準備
    X_raw, y = generate_data()
    # 2. データの標準化 (平均0, 標準偏差1にする)
    X_mean, X_std = X_raw.mean(), X_raw.std()
    X = (X_raw - X_mean) / X_std # ここで X を安全なスケールに変換！
    # 3. モデルの初期化と学習
    # スケーリングされたため、学習率 lr=0.05 で超高速かつ超安定して収束します！
    print("--- 身長から体重を予測するモデルの学習開始 ---")
    model = LinearRegressionScratch(lr=0.05, epochs=100)
    model.fit(X, y)
    print("--- 学習終了 ---")
    # 4. 最終的な予測結果をプロット (描画の際は、標準化されたXを元の身長スケールに戻します)
    plt.figure(figsize=(12, 5))
    # グラフ1: データと予測線のフィッティング結果
    plt.subplot(1, 2, 1)
    plt.scatter(X_raw, y, color="blue", label="Actual Students", alpha=0.7)
    # 予測用の滑らかな線を描画するために、元の身長スケールから予測
    X_test_raw = np.linspace(X_raw.min(), X_raw.max(), 100)
    X_test_scaled = (X_test_raw - X_mean) / X_std
    y_test_pred = model.predict(X_test_scaled)
    # プロットする際は「元の身長(X_test_raw)」と「予測された体重(y_test_pred)」を使うので直感的！
    plt.plot(X_test_raw, y_test_pred, color="red", linewidth=3,
             label="Fitted Prediction Line")
    plt.title("Height vs Weight Regression (Standardized)")
    plt.xlabel("Height (cm)")
    plt.ylabel("Weight (kg)")
    plt.grid(True)
    plt.legend()
    # グラフ2: 学習の進捗に伴う損失関数の低下
    plt.subplot(1, 2, 2)
    plt.plot(model.loss_history, color="purple", linewidth=2)
    plt.title("Learning Curve (MSE Loss)")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("01_linear_regression/regression_height_weight.png")
    print("結果グラフを '01_linear_regression/regression_height_weight.png' に保存しました！")
