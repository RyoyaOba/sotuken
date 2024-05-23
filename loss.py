
import matplotlib.pyplot as plt
import pandas as pd

# CSVファイルからデータを読み込む
file_path = '/app/Oba_卒業研究A/gen_losses_model_c23.csv'
data = pd.read_csv(file_path)

# 1列目と2列目のデータを取得
gen_g =data.iloc[:,1]
gen_f = data.iloc[:,2]
disc_x = data.iloc[:,3]
disc_y = data.iloc[:,4]

# gen_loss_data = data.iloc[:, 1]

# disc_loss_data = data.iloc[:, 3]

# gen_lossとdisc_lossのグラフをプロット
plt.figure(figsize=(10, 6))
# x軸の数字が見切れないようにする

#, frameon=True, loc='upper left')

plt.plot(gen_f, label='Generator fake to real',color = 'red')
plt.plot(gen_g, label='Generator real to fake', color = 'blue' )

plt.plot(disc_x, label='Discriminator target gait', color = 'coral' )
plt.plot(disc_y, label='Discriminator non target gait', color = 'dodgerblue')
plt.legend(fontsize=16, )


plt.xlabel('Epoch', fontsize = 28)
plt.ylabel('Loss', fontsize=28)

plt.autoscale(axis='x', tight=True)

plt.xticks(fontsize=20)#,rotation = 45)
plt.yticks(fontsize=20)

plt.grid(True, color='black', linestyle='--', linewidth=0.5)
plt.gca().spines['top'].set_color('black')
plt.gca().spines['bottom'].set_color('black')
plt.gca().spines['left'].set_color('black')
plt.gca().spines['right'].set_color('black')

plt.xlim(0, 1000)
plt.ylim(0,2)
#plt.title('Generator_G and Discriminator_G Loss')
#plt.autoscale(axis='x', tight=True)
plt.grid(True)

# 画像として保存

plt.tight_layout()
plt.show()

plt.savefig('SENet_Gf_Dy.png')
#--------------
# Oba_卒業研究A
#--------------

# import numpy as np
# import matplotlib.pyplot as plt

# # クロスエントロピー関数
# def cross_entropy(p, q):
#     return -np.sum(p * np.log(q))

# # バイナリクロスエントロピー関数
# def binary_cross_entropy(p, q):
#     return - (p * np.log(q) + (1 - p) * np.log(1 - q))

# # MSE損失関数
# def mean_squared_error(y_true, y_pred):
#     return np.mean((y_true - y_pred) ** 2)

# # 真の値と予測された値を定義
# y_true = 5.0  # 真の値
# y_values = np.linspace(0, 10, 100)  # 予測された値の範囲を生成


# # 真の確率分布と予測された確率分布を定義
# p_true = 0.7  # 真の確率（バイナリクラス分類の場合は1つの確率）
# q_values = np.linspace(0.01, 0.99, 100)  # 予測された確率の範囲を生成

# # 損失関数を計算
# cross_entropy_values = [cross_entropy(np.array([p_true]), np.array([q])) for q in q_values]
# binary_cross_entropy_values = [binary_cross_entropy(p_true, q) for q in q_values]
# # 損失関数を計算
# mse_values = [mean_squared_error(y_true, y_pred) for y_pred in y_values]

# # グラフ化
# plt.figure(figsize=(8, 6))
# plt.plot(q_values, cross_entropy_values, label='Cross Entropy', color='blue')
# plt.plot(q_values, binary_cross_entropy_values, label='Binary Cross Entropy', color='red')
# plt.plot(y_values, mse_values, label='Mean Squared Error', color='green')
# plt.xlabel('Predicted Probability')
# plt.ylabel('Loss')
# plt.title('Comparison of Cross Entropy and Binary Cross Entropy')
# plt.legend()
# plt.grid(True)

# # 画像として保存
# plt.savefig('test.png')
# plt.show()


# import numpy as np
# import matplotlib.pyplot as plt

# # 仮想的な損失の推移を生成（ランダムな値を使用）
# num_epochs = 100
# generator_loss = np.random.rand(num_epochs) * 5  # Generatorの損失（ランダムな値を生成）
# discriminator_loss = np.random.rand(num_epochs) * 3  # Discriminatorの損失（ランダムな値を生成）

# # グラフ化
# plt.figure(figsize=(8, 6))
# epochs = np.arange(1, num_epochs + 1)
# plt.plot(epochs, generator_loss, label='Generator Loss', color='blue')
# plt.plot(epochs, discriminator_loss, label='Discriminator Loss', color='red')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.title('MSE Loss Transition in GAN')
# plt.legend()
# plt.grid(True)
# plt.savefig('test2.png')
# plt.show()

