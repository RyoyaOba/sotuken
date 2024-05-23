import os
import numpy as np

# 時系列データの長さ
Time_Series_len = 400

def normalize(dt_ori, all_min_values, all_max_values):
    # データを0から1の範囲に正規化
    dt_normalized = (dt_ori - all_min_values) / (all_max_values - all_min_values)
    return dt_normalized

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
    r'C:\Users\human\Oba_卒業研究A\2023年度歩容測定実験\venus3d_data\stride_data',
    r'C:\Users\human\Oba_卒業研究A\2023年度歩容測定実験\venus3d_data\stride_L_data'
]

# 出力先フォルダのパスリストを作成
output_folders = [
    r'C:\Users\human\Oba_卒業研究A\2023年度歩容測定実験\venus3d_data\normalization_data\テストR', 
    r'C:\Users\human\Oba_卒業研究A\2023年度歩容測定実験\venus3d_data\normalization_data\テストL'
]

# ヘッダー
header = '体幹角度_矢状面,体幹角度_前額面,右股関節角度_矢状面,右股関節角度_前額面, 左股関節角度_矢状面, 左股関節角度_前額面,右膝関節角度_矢状面, 右膝関節角度_前額面, 左膝関節角度_矢状面,左膝関節角度_前額面, 右足関節角度_矢状面, 右足関節角度_前額面, 左足関節角度_矢状面, 左足関節角度_前額面, L_FP1-Fx(N/kg), L_FP1-Fy(N/kg), L_FP1-Fz(N/kg), R_FP2-Fx(N/kg), R_FP2-Fy(N/kg),R_FP2-Fz(N/kg), R拇趾床間距離, L拇趾床間距離' 
# 全CSVファイルから最大値と最小値を求める
all_min_values = np.inf
all_max_values = -np.inf
for input_folder in input_folders:
    file_list = os.listdir(input_folder)
    for file_name in file_list:
        file_path = os.path.join(input_folder, file_name)
        dt_ori = np.genfromtxt(file_path, delimiter=',', skip_header=1)  # 1行目をスキップせずにデータを読み込み
        min_values = dt_ori.min(axis=0)
        max_values = dt_ori.max(axis=0)
        all_min_values = np.minimum(all_min_values, min_values)
        all_max_values = np.maximum(all_max_values, max_values)

# 入力フォルダからファイルを読み込み、正規化してリサイズし、CSVファイルに保存
for input_folder, output_folder in zip(input_folders, output_folders):
    file_list = os.listdir(input_folder)
    for file_name in file_list:
        file_path = os.path.join(input_folder, file_name)
        # ファイルを読み込んで正規化
        dt_ori = np.genfromtxt(file_path, delimiter=',', skip_header=1)  # 1行目をスキップせずにデータを読み込み

        # 正規化
        dt_normalized = normalize(dt_ori, all_min_values, all_max_values)

        # リサイズ
        dt_resize = resize(dt_normalized)

        # 出力フォルダに保存
        output_file_path = os.path.join(output_folder, file_name)
        output_file_path = os.path.splitext(output_file_path)[0] + '.csv'  # 拡張子を.csvに変更

        # 指数表記で保存
        np.savetxt(output_file_path, dt_resize, delimiter=',', header=header, comments='', fmt='%.18e')  # Save with custom header

print("正規化とリサイズ、保存が完了しました。")
