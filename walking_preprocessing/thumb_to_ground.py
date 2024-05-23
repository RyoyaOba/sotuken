
import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# 入力ファイルのパスを指定
input_folder = r'C:\Users\human\Oba_卒業研究A\2023年度歩容測定実験\venus3d_data\spline_interpolation_data'
output_folder = r'C:\Users\human\Oba_卒業研究A\2023年度歩容測定実験\venus3d_data\拇趾床間距離'

# 出力フォルダが存在しない場合は作成
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# フォルダ内の全てのCSVファイルに対して処理
for file_name in os.listdir(input_folder):
    if file_name.endswith('.csv'):
        input_file_path = os.path.join(input_folder, file_name)
        output_file_name = file_name.replace('spline', 'tc').replace('.csv', '_拇趾床間距離.csv')  # 出力ファイルの名前を設定

        # CSVファイルを読み込みます
        df = pd.read_csv(input_file_path)

        # 19(Z)と20(Z)の列のデータを抽出します
        toe_distance_data = df[['19(Z)', '20(Z)']]

        toe_average_data = toe_distance_data[:2500]

        diff_toe_distance_data = toe_distance_data - toe_average_data.mean()

        # データを0から1の範囲に正規化します
        scaler = MinMaxScaler()
        normalized_diff_toe_distance_data = scaler.fit_transform(diff_toe_distance_data)

        # 正規化されたデータをDataFrameに変換します
        normalized_diff_toe_distance_df = pd.DataFrame(normalized_diff_toe_distance_data, columns=diff_toe_distance_data.columns)

        # 出力フォルダに新しいCSVファイルを保存します
        output_file_path = os.path.join(output_folder, output_file_name)
        normalized_diff_toe_distance_df.to_csv(output_file_path, index=False)

print("処理が完了しました。")

