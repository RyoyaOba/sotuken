import os
import numpy as np

# 入力フォルダのリスト
input_folders = [r'C:\Users\human\Oba_卒業研究A\2023年度歩容測定実験\venus3d_data\stride_data',
                 r'C:\Users\human\Oba_卒業研究A\2023年度歩容測定実験\venus3d_data\stride_L_data']

# 出力フォルダの共通パス
output_base_folder = r'C:\Users\human\Oba_卒業研究A\2023年度歩容測定実験\venus3d_data\normalization_data'

# 出力フォルダが存在しない場合は作成する
if not os.path.exists(output_base_folder):
    os.makedirs(output_base_folder)

# データを正規化して出力
for input_folder in input_folders:
    output_folder = os.path.join(output_base_folder, os.path.basename(input_folder))
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.endswith('.csv'):
            input_filepath = os.path.join(input_folder, filename)
            output_filename = filename.replace('filtered_linking_', '')  # "filtered_linking_"を取り除く
            output_filepath = os.path.join(output_folder, output_filename)
            data = np.genfromtxt(input_filepath, delimiter=',', skip_header=1)  # 1行目をスキップしてデータを読み込み
            header_line = open(input_filepath).readline().strip()

            # 列ごとに最小値と最大値を求めて正規化
            min_values = data.min(axis=0)
            max_values = data.max(axis=0)
            normalized_data = (data - min_values) / (max_values - min_values)

            # 0 < x < 1の範囲にクリップする
            normalized_data = np.clip(normalized_data, 0, 1)

            # 指数表記を小数形式で表示する設定
            np.set_printoptions(suppress=True)

            # 正規化後のデータをcsvファイルに保存
            np.savetxt(output_filepath, normalized_data, delimiter=',', header=header_line, comments='')
