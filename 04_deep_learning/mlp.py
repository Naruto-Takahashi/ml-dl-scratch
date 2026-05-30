import numpy as np
import matplotlib.pyplot as plt

class MLPScratch:
    def __init__(self, input_dim=1, hidden_dim=8, output_dim=1, lr=0.1):
        self.lr = lr

        # パラメータの初期化 (重みはランダムな小数値、バイアスは0で初期化)
        np.random.seed(42)
        self.W1 = np.random.randn(hidden_dim, input_dim) * 0.1
        self.b1 = np.zeros((hidden_dim, 1))
        self.W2 = np.random.randn(output_dim, hidden_dim) * 0.1
        self.b2 = np.zeros((output_dim, 1))

        self.loss_history = []

    def _sigmoid(self, z):
        return 1.0 / (1.0 + np.exp(-np.clip(z, -250, 250)))

    def _sigmoid_derivative(self, a):
        # シグモイド関数の微分: f'(z) = a * (1 - a)  (※aは活性化後の値)
        return a * (1.0 - a)

    def forward(self, X):
        # X: (input_dim, N)
        # 1. 隠れ層の計算
        self.z1 = np.dot(self.W1, X) + self.b1
        self.a1 = self._sigmoid(self.z1) # 隠れ層の出力 (hidden_dim, N)

        # 2. 出力層の計算
        self.z2 = np.dot(self.W2, self.a1) + self.b2
        # 今回は回帰問題 (体重予測や連続関数の近似)を扱うため、出力層は活性化関数なし(恒等写像)とします
        self.a2 = self.z2 # 予測値 (output_dim, N)

        return self.a2

    def backward(self, X, y):
        # X: (input_dim, N), y: (output_dim, N)
        N = X.shape[1]

        # --- 誤差逆伝播 (Backpropagation) の手動行列計算 ---

        # 1. 出力層の誤差 (予測値 - 正解)
        # MSE Loss = 1/(2N) * sum((a2 - y)^2) の微分
        dz2 = (self.a2 - y) / N # (output_dim, N)

        # 2. 出力層のパラメータの勾配
        dW2 = np.dot(dz2, self.a1.T) # (output_dim, hidden_dim)
        db2 = np.sum(dz2, axis=1, keepdims=True) # (output_dim, 1)

        # 3. 隠れ層への逆伝播
        # 連鎖律: W2の転置を行いて隠れ層に誤差を逆伝播し、さらに活性化関数(Sigmoid)の微分を掛ける
        da1 = np.dot(self.W2.T, dz2) # (hidden_dim, N)
        dz1 = da1 * self._sigmoid_derivative(self.a1) # (hidden_dim, N)

        # 4. 隠れ層のパラメータの勾配
        dW1 = np.dot(dz1, X.T) # (hidden_dim, input_dim)
        db1 = np.sum(dz1, axis=1, keepdims=True) # (hidden_dim, 1)

        # 5. パラメータの更新 (最急降下法)
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1

    def fit(self, X, y, epochs=5000):
        # X: (N,), y: (N,) の入力から行列形状 (1, N) にリシェイプします
        X_input = X.reshape(1, -1)
        y_input = y.reshape(1, -1)

        for epoch in range(epochs):
            # 順伝播
            self.forward(X_input)

            # 損失計算 (MSE)
            loss = np.mean((self.a2 - y_input) ** 2) / 2.0
            self.loss_history.append(loss)

            # 逆伝播・パラメータ更新
            self.backward(X_input, y_input)

            if (epoch + 1) % 500 == 0:
                print(f"Epoch {epoch+1:4d} | MSE Loss: {loss:.6f}")

    def predict(self, X):
        X_input = X.reshape(1, -1)
        return self.forward(X_input).flatten()

def generate_non_linear_data():
    np.random.seed(42)
    # 春から真夏までのフェスティバルの日の「気温 (℃)」を 100日分生成 (12℃ 〜 38℃)
    temperatures = np.random.uniform(12, 38, 100)

    # ビールの売上本数を予測する (山型のカーブを作る)
    # 気温 26℃付近がピーク (約 150本) になるような2次関数的な関係式
    sales = 150.0 - 0.7 * (temperatures - 26) ** 2

    # 個々の日のノイズ (雨やプロモーションなどによる変動) を乗せる
    noise = np.random.normal(0, 8, 100)
    sales = sales + noise

    # 売上本数がマイナスにならないようにガード
    sales = np.clip(sales, 10, None)

    return temperatures, sales

if __name__ == "__main__":
    # 1. データの準備
    X_raw, y = generate_non_linear_data()

    # 2. 気温データのスケーリング (標準化) - これによりMLPの学習が圧倒的に安定します
    X_mean, X_std = X_raw.mean(), X_raw.std()
    X = (X_raw - X_mean) / X_std

    # 売上データも同様にスケーリング (MLPの出力層のスケールに合わせるため)
    y_mean, y_std = y.mean(), y.std()
    y_norm = (y - y_mean) / y_std

    # 3. MLPモデルの初期化と学習
    # 入力1次元(気温)、隠れ層8次元、出力1次元(売上本数)
    model = MLPScratch(input_dim=1, hidden_dim=8, output_dim=1, lr=0.1)

    print("--- 気温からビールの売上を予測するMLPモデルの学習開始 ---")
    model.fit(X, y_norm, epochs=4000)
    print("--- 学習終了 ---")

    # 4. 最終的な予測結果をプロット (描画の際は、元のスケールに逆変換して直感的にします)
    plt.figure(figsize=(12, 5))

    # グラフ1: 気温とビールの売上データの山型カーブのフィッティング
    plt.subplot(1, 2, 1)
    plt.scatter(X_raw, y, color="blue", label="Actual Sales Per Day", alpha=0.7)

    # 滑らかな予測曲線の描画
    X_test_raw = np.linspace(12, 38, 200)
    X_test_scaled = (X_test_raw - X_mean) / X_std

    # MLPの正規化された予測値を取得し、元の「本数」スケールへ逆変換
    y_test_pred_norm = model.predict(X_test_scaled)
    y_test_pred = y_test_pred_norm * y_std + y_mean

    plt.plot(X_test_raw, y_test_pred, color="red", linewidth=3,
             label="MLP Learned Demand Curve")

    plt.title("Beer Sales Prediction based on Temp (Non-linear MLP)")
    plt.xlabel("Temperature (C)")
    plt.ylabel("Beer Sales (Units)")
    plt.grid(True)
    plt.legend()

    # グラフ2: 学習の進捗に伴うMSE損失の低下
    plt.subplot(1, 2, 2)
    plt.plot(model.loss_history, color="purple", linewidth=2)
    plt.title("Learning Curve (MSE Loss)")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.grid(True)

    plt.tight_layout()
    plt.savefig("04_deep_learning/mlp_result.png")
    print("結果グラフを '04_deep_learning/mlp_result.png' に保存しました！")
