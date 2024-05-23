
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# CSVファイルの読み込み
file_path = '/app/Oba_卒業研究A/loss_history2.csv'
df = pd.read_csv(file_path)

# グラフの描画
plt.figure(figsize=(10, 6))
plt.plot(df['epoch'], 
         df['train loss'], 
         label='Train Loss', 
         linewidth = 1.2, 
         color = 'blue')

plt.plot(df['epoch'],
          df['val loss'], 
          label='Validation Loss', 
          linewidth = 1.2, 
          color = 'red', alpha = 0.7)



# グラフにラベルやタイトルを追加
plt.xlabel('Epoch', fontsize = 28)
plt.ylabel('Loss', fontsize=28)
#plt.xticks(np.linspace(0,100,6))     
#plt.yticks(np.linspace(0,1,6)) 

# x軸の数字が見切れないようにする
plt.autoscale(axis='x', tight=True)

plt.xticks(fontsize=18) 
plt.yticks(fontsize=18)
plt.grid(True, color='black', linestyle='--', linewidth=0.5)
plt.gca().spines['top'].set_color('black')
plt.gca().spines['bottom'].set_color('black')
plt.gca().spines['left'].set_color('black')
plt.gca().spines['right'].set_color('black')
plt.tight_layout()
plt.xlim(0, df['epoch'].max())

plt.ylim(0, 0.9)
#plt.title('Generator_G and Discriminator_G Loss')
plt.legend(fontsize=20, frameon=False)
plt.grid(True)

# グラフを保存
save_path = '/app/Oba_卒業研究A/for_emotional_engineering/loss2_plot.png'
plt.savefig(save_path)

# グラフを表示
plt.show()
