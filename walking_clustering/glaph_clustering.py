import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib

def plot_columns_together(file_paths, column_number, std_line_color='royalblue'):
    plt.figure(figsize=(7, 5))

    all_data = []

    for file_path in file_paths:
        df = pd.read_csv(file_path, encoding='shift_jis')
        num_rows = df.shape[0]

        y = df.iloc[:, column_number]

        x = [i * 100 / (num_rows - 1) for i in range(num_rows)]

        plt.plot(x, y, color='gray', linewidth=0.5)
        all_data.append(y)

    mean_data = np.mean(all_data, axis=0)
    std_data = np.std(all_data, axis=0)

    for i in range(num_rows):
        plt.plot([x[i], x[i]], [mean_data[i] - std_data[i], mean_data[i] + std_data[i]], color=std_line_color, linewidth=1)

    plt.plot(x, mean_data, color='red', linewidth=2, label='Mean')
    plt.fill_between(x, mean_data - std_data, mean_data + std_data, color=std_line_color, alpha=0.3, label='Standard Deviation')
    plt.xlabel('Walking cycle [%]', fontsize=11)
    plt.ylabel('Normalized data', fontsize=11)

    # タイトルを入力ファイルの各列の先頭の行に設定
    column_label = df.columns[column_number]
    plt.title(f'{column_label}', fontsize=14)

    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.grid(True)
    plt.xlim(0, 100)
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.legend(loc='upper right', fontsize=11)

# フォルダのパス
input_clustering_1 = r'C:\Users\human\Oba_卒業研究A\2023年度歩容測定実験\venus3d_data\normalization_data\cluster_1'
input_clustering_2 = r'C:\Users\human\Oba_卒業研究A\2023年度歩容測定実験\venus3d_data\normalization_data\cluster_2'

# フォルダ内のファイルを取得
file_list_r = os.listdir(input_clustering_1)
file_list_l = os.listdir(input_clustering_2)

# フォルダ内の全ファイルを選択
selected_files_r = [os.path.join(input_clustering_1, file_name) for file_name in file_list_r]
selected_files_l = [os.path.join(input_clustering_2, file_name) for file_name in file_list_l]

# 1列目から20列目をグラフ化
#for column_number in range(0, 22):
#    # 各ファイルの先頭行の名前を取得
 #   df = pd.read_csv(selected_files_r[0], encoding='shift_jis')
  #  column_label = df.columns[column_number]
   ##
    # グラフを画像として保存
    # 特殊文字を削除してファイル名を生成
   # file_name_cleaned = ''.join(c for c in column_label if c.isalnum() or c in ['-', '_'])
    #plt.savefig(f'graph_column_{file_name_cleaned}.png', dpi=300)

    # ...
column_names = [
    "体幹角度_矢状面", "体幹角度_前額面", 
    "右股関節角度_矢状面", "右股関節角度_前額面",
    "左股関節角度_矢状面", "左股関節角度_前額面",
    "右膝関節角度_矢状面", "右膝関節角度_前額面",
    "左膝関節角度_矢状面", "左膝関節角度_前額面",
    "右足関節角度_矢状面", "右足関節角度_前額面",
    "左足関節角度_矢状面", "左足関節角度_前額面",
    "L_FP1-Fx(N/kg)", "L_FP1-Fy(N/kg)", "L_FP1-Fz(N/kg)",
    "R_FP2-Fx(N/kg)", "R_FP2-Fy(N/kg)", "R_FP2-Fz(N/kg)",
    "R拇趾床間距離", "L拇趾床間距離"
]

for column_label in column_names:
    plt.figure(figsize=(7, 5))
    
    # Find the corresponding index of the column_label
    column_number = column_names.index(column_label)
    
    plot_columns_together(selected_files_r, column_number, std_line_color='royalblue')
    plt.title(column_label, fontsize=14)  # Set the title directly here
    file_name_cleaned = ''.join(c for c in column_label if c.isalnum() or c in ['-', '_'])
    plt.savefig(f'graph_column_cluster1_{file_name_cleaned}.png', dpi=300)
    plt.clf()

for column_label in column_names:
    plt.figure(figsize=(7, 5))
    
    # Find the corresponding index of the column_label
    column_number = column_names.index(column_label)
    
    plot_columns_together(selected_files_l, column_number, std_line_color='royalblue')
    plt.title(column_label, fontsize=14)  # Set the title directly here
    file_name_cleaned = ''.join(c for c in column_label if c.isalnum() or c in ['-', '_'])
    plt.savefig(f'graph_column_cluster2_{file_name_cleaned}.png', dpi=300)
    plt.clf()
