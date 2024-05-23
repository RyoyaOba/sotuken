
import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
print(matplotlib.matplotlib_fname)
from matplotlib import rcParams
import japanize_matplotlib
import tensorflow as tf
from PIL import Image

tf.random.set_seed(1122)
tf.keras.backend
#最も悪い歩容データ
#number = 7276#
#number= 2214
#number = 5991
number =6656
# 中間の悪い歩容データ# number = 6496
#---------------------------
# 任意の入力データセットを作成
#---------------------------
# Set the font to Times New Roman
rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['Times New Roman']

class ClassificationDataset:
    def __init__(self, folder_path: str):
        self.folder_path = folder_path
        self.data = []

    def __call__(self):
        thumb_1_path = os.path.join(self.folder_path, \
                                    '/app/hinann/sotukenB_clustering')

        if os.path.exists(thumb_1_path):
            csv_file_path = os.path.join(thumb_1_path, f'SotukenA_data_{number}.csv')
            if os.path.exists(csv_file_path):
                # SotukenA_data_5.csvの読み込みと処理
                df = pd.read_csv(csv_file_path, encoding='latin-1')
                # 21列目と22列目を除外（列は0から始まる）
                excluded_columns = [20, 21]
                selected_columns = [col for col in range(len(df.columns)) if col not in excluded_columns]
                df = df.iloc[:, selected_columns]
                # label_data = df.iloc[0].values
                data = df.iloc[1:].values
                self.data.append(data)
                
                return self.data
            else:
                print(f"{number}.csv_NG。")
        #else:
            
        return None  # データの読み込みに問題がある場合はNoneを返す

#---------確認------------#
ds = ClassificationDataset('/app/hinann/sotukenB_clustering')
data = ds()

print(data)
generator = tf.keras.models.load_model("/app/Oba_卒業研究A/generator_f/model_c9_epoch100")
#generator= tf.keras.models.load_model('./generator_f/model_c9_epoch10')
output_path = "/app/Oba_卒業研究A/cc/"
# データを (1, 399, 20) の形状に変更する
data_reshaped = np.expand_dims(data[0], axis=0)

gen_data = generator(data_reshaped)

print(gen_data)


#-------選んだデータを生成器に入力-----------#
for i in range(len(data)):
    # Generate data for each sample in the loop
    gen_sample = generator(np.expand_dims(data[i], axis=0))
    gen_tensor = gen_sample[0].numpy()
    reshaped_tensor = gen_tensor.reshape(399, 20)

    # Process non-ideal data
    no_ideal_data = np.expand_dims(data[i], axis=0)
    reshaped_non_ideal_tensor = no_ideal_data.reshape(399, 20)

    # Concatenate generated and non-ideal data
    combined_output = f"{output_path}combined_data_{number}.csv"
    combined_data = np.concatenate((reshaped_tensor, reshaped_non_ideal_tensor), axis=1)
    np.savetxt(combined_output, combined_data, delimiter=",")


selected_combined_data = pd.read_csv(f"/app/Oba_卒業研究A/cc/combined_data_{number}.csv", header=None)

# タイトルのリスト
column_labels_japanize = ['体幹角度_矢状面','体幹角度_前額面',\
                '右股関節角度_矢状面','右股関節角度_前額面', \
                '左股関節角度_矢状面', '左股関節角度_前額面',\
                '右膝関節角度_矢状面', '右膝関節角度_前額面', \
                '左膝関節角度_矢状面','左膝関節角度_前額面',\
                '右足関節角度_矢状面','右足関節角度_前額面',\
                '左足関節角度_矢状面', '左足関節角度_前額面',\
                'L_FP1-Fx(N/kg)', 'L_FP1-Fy(N/kg)',' L_FP1-Fz(N/kg)',\
                'R_FP2-Fx(N/kg)', 'R_FP2-Fy(N/kg)','R_FP2-Fz(N/kg)']
# グラフのタイトル
column_labels = [
    "Normalized change\ntrunk angle yz [-]", \
    "Normalized change\ntrunk angle zx [-]", \
    "Normalized change\nR hip angle yz [-]",\
    "Normalized change\nR hip angle zx [-]",\
    "Normalized change\nL hip angle yz [-]",\
    "Normalized change\nL hip angle zx [-]", \
    "Normalized change\nR knee angle yz [-]", \
    "Normalized change\nR knee angle zx [-]", \
    "Normalized change\nL knee angle yz [-]", \
    "Normalized change\nL knee angle zx [-]",\
    "Normalized change\nR ankle angle yz [-]",\
    "Normalized change\nR ankle angle zx [-]",\
    "Normalized change\nL ankle angle yz [-]",\
    "Normalized change\nL ankle angle zx [-]", \
    "Normalized\nL ground reaction force x [-]",\
    "Normalized\nL ground reaction force y [-]", \
    "Normalized\nL ground reaction force z [-]", \
    "Normalized\nR ground reaction force x [-]", \
    "Normalized\nR ground reaction force y [-]", \
    "Normalized\nR ground reaction force z [-]",\
    "Normalized change of right\nthumb-to-grond distance [-]",
    "Normlaized change of left\nthumb-to-ground distance [-]"
]

column_labels_2 = ["change in\ntrunk angle YZ [-]",
                   "change in\ntrunk angle XZ [-]",
                   "change in\nhip angle YZ\nof the target leg [-]", "change in\nhip angle XZ\nof the target leg [-]", 
                "change in\nhip angle YZ\nof the non-target leg [-]", "change in\nhip angle XZ\nof the non-target leg [-]", "change in\nknee angle YZ\nof the target leg [-]", "change in\nknee angle XZ\nof the target leg [-]", 
                "change in\nknee angle YZ\nof the non-target leg [-]", "change in\nknee angle XZ\nof the non-target leg [-]", "change in\nankle angle YZ\nof the target leg [-]", "change in\nankle angle XZ\nof the target leg [-]", 
                "change in\nankle angle YZ\nof the non-target leg [-]", "change in\nankle angle XZ\nof the non-target leg [-]", 
                "\nground reaction force X\nof the non-target leg [-]", "\nground reaction force Y\nof the non-target leg [-]", "\nground reaction force Z\nof the non-target leg [-]",
                "\nground reaction force X\nof the target leg [-]", "\nground reaction force Y\nof the target leg [-]", 
                "\nground reaction force Z\nof the target leg [-]", "Normalized change of right\nthumb-to-grond distance [-]",
                "Normlaized change of left\nthumb-to-ground distance [-]"]
# 特殊文字への対応
def clean_filename(name):
    return re.sub(r'[^\w\-_\.]', '', name)
#---------------------------------
# データ1つだけを可視化（横線時間軸）
#---------------------------------
for i in range(20):
    plt.figure(figsize=(12, 6))

    
    plt.plot(
        selected_combined_data.index, selected_combined_data[i + 20],
        label='non_ideal_gait',
        color='blue'
    )

    plt.plot(
        selected_combined_data.index, selected_combined_data[i],
        label='ideal_gait',
        #linestyle='--',
        color='red'
    )    
    plt.legend(fontsize=18, frameon=False, fontname = 'Times New Roman')
    plt.xlabel('Walking cycle', fontsize=24, fontname = 'Times New Roman')
    plt.ylabel(f'{column_labels[i]}', fontsize=24, fontname = 'Times New Roman')
    plt.xticks(fontsize=20, fontname = 'Times New Roman')
    plt.yticks(fontsize=20, fontname = 'Times New Roman')
    plt.grid(True, color='black', linestyle='--', linewidth=0.5)
    plt.gca().spines['top'].set_color('black')
    plt.gca().spines['bottom'].set_color('black')
    plt.gca().spines['left'].set_color('black')
    plt.gca().spines['right'].set_color('black')
    plt.tight_layout()
    plt.xlim(0, 400)
    plt.ylim(0, 1)
    #plt.title(f'{column_labels[i]}')
    cleaned_label = clean_filename(column_labels[i])
 
    plt.savefig(f"{output_path}_{number}Time_series_{cleaned_label}_{i+1}.png", dpi=300)
    plt.close()


#-----------------------------
#データ1だけを可視化（歩行周期）
#-----------------------------
for i in range(20):
    plt.figure(figsize=(12, 6))
    total_steps = len(selected_combined_data) - 1
    percentage_steps = [i * 100 / total_steps for i in range(len(selected_combined_data))]
    
    plt.plot(
        percentage_steps, selected_combined_data[i + 20],
        label='non_ideal_gait',
        color='blue'
    )
    plt.plot(
        percentage_steps, selected_combined_data[i],
        label='ideal_gait',
        #linestyle='--',
        color='red'
    )
    plt.legend(fontsize=18, frameon=False)
    plt.xlabel('Walking Cycle [%]', fontsize=24)
    plt.ylabel(f'{column_labels[i]}', fontsize=24)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.grid(True, color='black', linestyle='--', linewidth=0.5)
    plt.gca().spines['top'].set_color('black')
    plt.gca().spines['bottom'].set_color('black')
    plt.gca().spines['left'].set_color('black')
    plt.gca().spines['right'].set_color('black')
    plt.tight_layout()
    plt.xlim(0, 100)
    plt.ylim(0, 1)
    plt.tight_layout()
    #plt.title(f'{column_labels[i]}')
    cleaned_label = clean_filename(column_labels[i])
 
    plt.savefig(f"{output_path}walking_cycle_{i+1}.png")
    plt.clf()
