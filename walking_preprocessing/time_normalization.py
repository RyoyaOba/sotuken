import os
import numpy as np

# 時系列データの長さ
Time_Series_len = 400

def resize(dt_ori):
    # dt_oriのshapeをリサイズ後のshapeに合わせる
    original_length = dt_ori.shape[0]
    new_length = Time_Series_len

    # リサイズ後の時間軸を生成
    new_time_axis = np.linspace(0, original_length - 1, new_length)

    # リサイズ後のデータを補間して生成
    dt_resize = np.zeros((new_length, dt_ori.shape[1]))
    for i in range(dt_ori.shape[1]):
        dt_resize[:, i] = np.interp(new_time_axis, np.arange(original_length), dt_ori[:, i])

    return dt_resize

# 入力フォルダのパスリストを作成
input_folders = [
    r'C:\Users\human\Oba_卒業研究A\2023年度歩容測定実験\venus3d_data\normalization_data\stride_data',
    r'C:\Users\human\Oba_卒業研究A\2023年度歩容測定実験\venus3d_data\normalization_data\stride_L_data'
]

# 出力先フォルダのパスリストを作成
output_folders = [
    r'C:\Users\human\Oba_卒業研究A\2023年度歩容測定実験\venus3d_data\normalization_data\テストR', 
    r'C:\Users\human\Oba_卒業研究A\2023年度歩容測定実験\venus3d_data\normalization_data\テストL'
]

# 入力フォルダからファイルを読み込み、リサイズしてCSVファイルに保存
for input_folder, output_folder in zip(input_folders, output_folders):
    file_list = os.listdir(input_folder)
    for file_name in file_list:
        file_path = os.path.join(input_folder, file_name)
        # ファイルを読み込んでリサイズ
        dt_ori = np.genfromtxt(file_path, delimiter=',', skip_header=1)  # 1行目をスキップしてデータを読み込み
        dt_resize = resize(dt_ori)

        # リサイズ後のデータを0から1の範囲に正規化
        min_value = dt_resize.min()
        max_value = dt_resize.max()
        dt_normalized = (dt_resize - min_value) / (max_value - min_value)

        # 出力フォルダに保存
        output_file_path = os.path.join(output_folder, file_name)
        output_file_path = os.path.splitext(output_file_path)[0] + '.csv'  # 拡張子を.csvに変更
        
        # リサイズした配列を指定した形式で保存
        np.savetxt(output_file_path, dt_normalized, delimiter=',', fmt='%.6f')  # fmt='%.6f'で小数点以下6桁までの表示形式を指定
