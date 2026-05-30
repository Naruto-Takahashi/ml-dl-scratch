import numpy as np
import matplotlib.pyplot as plt
class NaiveBayesSpamFilter:
    def __init__(self, alpha=1.0):
        self.alpha = alpha  # ラプラススムージングのパラメータ (+1)
        self.vocab = set()  # 全単語の辞書 (ボキャブラリ)

        # クラスごとの単語出現確率の保持用
        self.word_probs_spam = {}
        self.word_probs_ham = {}
        # スパムである事前確率 P(Spam)
        self.p_spam = 0.0
    def _tokenize(self, text):
        # 簡単なトークナイザー (小文字化して空白で分割)
        return text.lower().split()
    def fit(self, X_texts, y):
        n_samples = len(X_texts)
        n_spam = np.sum(y)
        n_ham = n_samples - n_spam

        # 1. 事前確率 P(Spam) の計算
        self.p_spam = n_spam / n_samples

        # 2. ボキャブラリ (辞書) の構築と単語カウントの初期化
        spam_word_counts = {}
        ham_word_counts = {}

        for text, label in zip(X_texts, y):
            words = self._tokenize(text)
            for word in words:
                self.vocab.add(word)
                if label == 1: # Spam
                    spam_word_counts[word] = spam_word_counts.get(word, 0) + 1
                else:          # Ham (通常メール)
                    ham_word_counts[word] = ham_word_counts.get(word, 0) + 1

        # 3. 各クラスにおける全単語の総カウント数
        total_spam_words = sum(spam_word_counts.values())
        total_ham_words = sum(ham_word_counts.values())
        vocab_size = len(self.vocab)

        # 4. 単語ごとの出現確率 P(word | Class) の計算 (ラプラススムージング適用)
        # P(word | Spam) = (Spam内のその単語のカウント + alpha) / (Spam内の全単語数 + alpha * 語彙数)
        for word in self.vocab:
            count_spam = spam_word_counts.get(word, 0)
            count_ham = ham_word_counts.get(word, 0)

            self.word_probs_spam[word] = (count_spam + self.alpha) / (total_spam_words + self.alpha * vocab_size)
            self.word_probs_ham[word] = (count_ham + self.alpha) / (total_ham_words + self.alpha * vocab_size)

    def predict_proba(self, text):
        words = self._tokenize(text)

        # 確率が極めて小さくなって0になる(アンダーフロー)のを防ぐため、
        # 掛け算を「Log(対数)の足し算」に変換して計算します！
        log_prob_spam = np.log(self.p_spam)
        log_prob_ham = np.log(1.0 - self.p_spam)

        vocab_size = len(self.vocab)
        # 訓練時に見つからなかった未知の単語用のアウトオブボキャブラリ確率 (デフォルト)
        default_prob_spam = self.alpha / (sum(self.word_probs_spam.values()) + self.alpha * vocab_size)
        default_prob_ham = self.alpha / (sum(self.word_probs_ham.values()) + self.alpha * vocab_size)

        for word in words:
            # 辞書にあればその確率を、なければデフォルトの平滑化確率を対数で加算します
            log_prob_spam += np.log(self.word_probs_spam.get(word, default_prob_spam))
            log_prob_ham += np.log(self.word_probs_ham.get(word, default_prob_ham))

        # Logスケールから確率 [0, 1] に復元する (Softmax的なアプローチ)
        # P(Spam) = e^(log_p_spam) / (e^(log_p_spam) + e^(log_p_ham))
        # オーバーフローを防ぐため、最大値を引いて調整します
        max_log = max(log_prob_spam, log_prob_ham)
        prob_spam = np.exp(log_prob_spam - max_log) / (np.exp(log_prob_spam - max_log) + np.exp(log_prob_ham - max_log))

        return prob_spam

    def predict(self, X_texts):
        # 確率が 0.5 以上ならスパム (1) と判定します
        return np.array([1 if self.predict_proba(text) >= 0.5 else 0 for text in X_texts])

def generate_spam_dataset():
    # 訓練用メールテキストデータ (0: 通常メール「Ham」, 1: 迷惑メール「Spam」)
    emails = [
        # --- 通常メール (Ham) ---
        "明日の プロジェクト 会議 は 午後 2時 から 開始 します",
        "資料 の 確認 と 修正 を よろしく お願い します",
        "来週 の 進捗 報告 会議 の アジェンダ を 送付 します",
        "本日の 面談 の 件、 スケジュール を 調整 しました",
        "確認 したい 技術 的な 仕様 が ありますので メール します",

        # --- 迷惑メール (Spam) ---
        "【重要】 無料 で 豪華 プレゼント が 当選 しました ！",
        "至急 ご確認 ください ！ 100万円 当選 の チャンス 実施 中 ！",
        "今すぐ 無料 登録 して 限定 特典 を 獲得 しましょう",
        "無料 カタログ 請求 で 豪華 ギフト カード プレゼント 中",
        "当選 者 限定 の 特別な プレゼント の ご案内 です"
    ]
    labels = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    return emails, labels

if __name__ == "__main__":
    emails, labels = generate_spam_dataset()

    # モデルの学習
    model = NaiveBayesSpamFilter(alpha=1.0)
    model.fit(emails, labels)

    # 未知の新しい受信メール 4 通に対する判定テスト
    test_emails = [
        "明日の 会議 の 資料 を 確認 して 送ります",                  # テストA (通常ぽい)
        "無料 の プレゼント キャンペーン が 当選 しました ！",          # テストB (スパムぽい)
        "プロジェクト の スケジュール を 確認 してください",             # テストC (通常ぽい)
        "至急 豪華 特典 を 無料 で 獲得 する チャンス です"             # テストD (スパムぽい)
    ]

    print("--- 迷惑メールフィルタ (単純ベイズ) 判定開始 ---")

    # グラフのプロット用データ保持
    test_names = ["Email A (Ham-like)", "Email B (Spam-like)", "Email C (Ham-like)", "Email D (Spam-like)"]
    spam_probabilities = []

    for i, test_email in enumerate(test_emails):
        prob = model.predict_proba(test_email)
        spam_probabilities.append(prob)
        prediction = 1 if prob >= 0.5 else 0
        status = "SPAM (1)" if prediction == 1 else "NORMAL (0)"
        print(f"[{test_names[i]}]")
        print(f"  本文: \"{test_email}\"")
        print(f"  スパム確率: {prob*100:.2f}%  -->  判定: {status}\n")

    # 結果のプロット (棒グラフでスパム確率の可視化)
    plt.figure(figsize=(10, 6))
    bars = plt.bar(test_names, spam_probabilities, color=["blue", "red", "blue", "red"], alpha=0.7, edgecolor="black")

    # 決定境界 (閾値 50%) の赤線を引く
    plt.axhline(y=0.5, color="black", linestyle="--", linewidth=2, label="Spam Threshold (50%)")

    # 棒の上にパーセンテージをテキスト表示
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.02, f"{yval*100:.1f}%", ha="center", va="bottom", fontweight="bold")

    plt.title("Email Spam Probability Prediction (Naive Bayes Scratch)")
    plt.ylabel("Calculated Spam Probability")
    plt.ylim(0, 1.1)
    plt.grid(axis="y", linestyle=":")
    plt.legend()
    plt.savefig("07_naive_bayes/spam_prediction.png")
    print("予測結果の確率グラフを '07_naive_bayes/spam_prediction.png' に保存しました！")
