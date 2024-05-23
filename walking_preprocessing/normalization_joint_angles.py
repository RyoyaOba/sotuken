
import os
import pandas as pd

def normalize_data(data):
    # データを0-1の範囲に正規化する関数
    min_val = data.min()
    max_val = data.max()
    return (data - min_val) / (max_val - min_val)

# 入力フォルダのパス
input_folder_path = "C:/Users/human/Oba_卒業研究A/2023年度歩容測定実験/venus3d_data/joint_angles/"

# 出力フォルダのパス
output_folder_path = "C:/Users/human/Oba_卒業研究A/2023年度歩容測定実験/venus3d_data/normalization_joint_angles/"

# joint_anglesフォルダ内の全てのCSVファイルを取得
file_list = os.listdir(input_folder_path)

for file_name in file_list:
    if file_name.startswith("angle_diff"):
        # CSVファイルを読み込む
        file_path = os.path.join(input_folder_path, file_name)
        data = pd.read_csv(file_path, header=0, encoding='shift_jis')  # utf-8で読み込む

        # 各列のデータを正規化する
        normalized_data = data.apply(normalize_data, axis=0)

        # 出力ファイル名を指定
        output_file_name = "normalization_" + file_name

        # フルパスを結合してCSVファイルに出力
        output_file_path = os.path.join(output_folder_path, output_file_name)
        normalized_data.to_csv(output_file_path, index = False, encoding = 'shift_jis')