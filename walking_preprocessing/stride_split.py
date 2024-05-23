
import os
import pandas as pd

# 時間範囲
start_time_threshold = int(0.8 * 250)  # 0.8秒をサンプル数に変換
end_time_threshold = int(1.7 * 250)   # 1.7秒をサンプル数に変換
# 体重の対応表


# 体重の対応表
weight_dict = {
    'DLVI': 48.9,
    'SESV': 61.9,
    'IFCN': 65.3,
    'OSRZ': 55.5,
    'RVGK': 66.4,
    'MXNI': 59.1,
    'ZUIO': 66.3,
    'NVIX': 71.9,
    'NJUA': 84.8,
    'UTWG': 54.4,
    'BHVL': 54.3,
    # 8/23以降追加データ
    'SZZY': 63.7,
    'CUJT': 52.8,
    'BUAL': 53.9,
    'HAWC': 58.9,
    'KDDO': 68.0,
    'HKGP': 46.0,
    'DIPJ': 57.7,
    'JAQJ': 42.6,
    'WWOT': 54.3,
    'DOMG': 49.0,
    'YXRA': 57.8,
    'FTND': 50.5,
    'CXIP': 56.7,
    'XDIO': 87.0,
    'KILB': 45.2,
    'BAEO': 66.2,
    'AZDJ': 49.8,
    'KSKV': 79.8,
    'GUIK': 89.3,
    'IOOP': 58.5,
}

# 出力フォルダのパス（右足のストライド）
input_folder = r'C:\Users\human\Oba_卒業研究A\2023年度歩容測定実験\venus3d_data\condition_split_data'
# ... (省略) ...

# 出力フォルダのパス（右足のストライド）
output_folder_R = r'C:\Users\human\Oba_卒業研究A\2023年度歩容測定実験\venus3d_data\stride_data'

#  出力フォルダのパス（左足のストライド）
output_folder_L = r'C:\Users\human\Oba_卒業研究A\2023年度歩容測定実験\venus3d_data\stride_L_data'
# ダのパス（右足のストライド、ファイルサイズが140 kBのものを再出力）
output_folder_R_resized = r'C:\Users\human\Oba_卒業研究A\2023年度歩容測定実験\venus3d_data\stride_data'

# 出力フォルダのパス（左足のストライド、ファイルサイズが140 kBのものを再出力）
output_folder_L_resized = r'C:\Users\human\Oba_卒業研究A\2023年度歩容測定実験\venus3d_data\stride_L_data'
# ファイルサイズの上限（KB）
max_file_size_kb = 140

def file_size_check(data, max_file_size_kb):
    temp_file = os.path.join(output_folder_R, "temp.csv")
    data.to_csv(temp_file, index=False, encoding='shift_jis')
    file_size_kb = os.path.getsize(temp_file) / 1024
    os.remove(temp_file)
    return file_size_kb <= max_file_size_kb


def preprocess_left_leg_data(data):
    data = data.copy()
    data['体幹角度_前額面'] = -data['体幹角度_前額面']
    data['左股関節角度_前額面'], data['右股関節角度_前額面'] = -data['右股関節角度_前額面'], -data['左股関節角度_前額面']
    data['左膝関節角度_前額面'], data['右膝関節角度_前額面'] = -data['右膝関節角度_前額面'], -data['左膝関節角度_前額面']
    data['左足関節角度_前額面'], data['右足関節角度_前額面'] = -data['右足関節角度_前額面'], -data['左足関節角度_前額面']
    data['FP1-Fx(N/kg)'], data['FP2-Fx(N/kg)'] = -data['FP2-Fx(N/kg)'], -data['FP1-Fx(N/kg)']
    data['FP1-Fy(N/kg)'], data['FP2-Fy(N/kg)'] = data['FP2-Fy(N/kg)'], data['FP1-Fy(N/kg)']
    data['FP1-Fz(N/kg)'], data['FP2-Fz(N/kg)'] = data['FP2-Fz(N/kg)'], data['FP1-Fz(N/kg)']
    data['左股関節角度_矢状面'], data['右股関節角度_矢状面'] = data['右股関節角度_矢状面'], data['左股関節角度_矢状面']
    data['左足関節角度_矢状面'], data['右足関節角度_矢状面'] = data['右足関節角度_矢状面'], data['左足関節角度_矢状面']
    data['左膝関節角度_矢状面'], data['右膝関節角度_矢状面'] = data['右膝関節角度_矢状面'], data['左膝関節角度_矢状面']
    data['19(Z)'], data['20(Z)'] = data['20(Z)'], data['19(Z)']
    return data

# 入力フォルダ内の全てのCSVファイルを取得
csv_files = [f for f in os.listdir(input_folder) if f.endswith('.csv')]

# 右脚と左脚のストライドデータを一度に保存するリスト
all_stride_data_R = []
all_stride_data_L = []

# すべてのCSVファイルに対して処理を行う
for csv_file in csv_files:
    input_path = os.path.join(input_folder, csv_file)
    try:
        df = pd.read_csv(input_path, encoding='shift_jis')
    except UnicodeDecodeError:
        df = pd.read_csv(input_path, encoding='別のエンコーディング')  # 別のエンコーディングを指定してください

    person_id = csv_file.split('_')[1]
    threshold = weight_dict.get(person_id, 0.0) * 0.03

    # 右脚のストライド検出
    ground_contact_mask_R = df['FP2-Fz(N/kg)'] >= threshold
    stride_data_list_R = []
    stride_start = 0
    for i in range(1, len(df)):
        if (
            ground_contact_mask_R[i]
            and not all(ground_contact_mask_R[i - 5 : i])
            and all(ground_contact_mask_R[i : i + 10])
        ):
            stride_data = df.iloc[stride_start : i + 10]
            stride_data_list_R.append(stride_data)
            stride_start = i

    # 左脚のストライド検出
    ground_contact_mask_L = df['FP1-Fz(N/kg)'] >= threshold
    stride_data_list_L = []
    stride_start = 0
    for i in range(1, len(df)):
        if (
            ground_contact_mask_L[i]
            and not all(ground_contact_mask_L[i - 5 : i])
            and all(ground_contact_mask_L[i : i + 10])
        ):
            stride_data = df.iloc[stride_start : i + 10]
            stride_data_list_L.append(stride_data)
            stride_start = i

    # 右脚のストライドデータを追加
    all_stride_data_R.extend(stride_data_list_R)

    # 左脚のストライドデータを追加
    all_stride_data_L.extend(stride_data_list_L)

# 右脚のストライドデータを一括保存
for idx, stride_data in enumerate(all_stride_data_R):
    num_samples = len(stride_data)
    time_condition = start_time_threshold <= num_samples <= end_time_threshold
    row_condition = 200 <= num_samples <= 400
    if time_condition and row_condition and file_size_check(stride_data, max_file_size_kb):
        output_filename = f"stride_R_{idx+1}_{csv_file}"
        output_path = os.path.join(output_folder_R, output_filename)
        stride_data.to_csv(output_path, index=False, encoding='shift_jis')

# 左脚のストライドデータを一括保存
for idx, stride_data in enumerate(all_stride_data_L):
    num_samples = len(stride_data)
    time_condition = start_time_threshold <= num_samples <= end_time_threshold
    row_condition = 200 <= num_samples <= 400
    if time_condition and row_condition and file_size_check(stride_data, max_file_size_kb):
        stride_data = preprocess_left_leg_data(stride_data)
        output_filename = f"stride_L_{idx+1}_{csv_file}"
        output_path = os.path.join(output_folder_L, output_filename)
        stride_data.to_csv(output_path, index=False, encoding='shift_jis')

# ファイルの再出力（ファイルサイズが140 kBのもの）
for folder, output_folder_resized in [(output_folder_R, output_folder_R_resized), (output_folder_L, output_folder_L_resized)]:
    stride_files = [f for f in os.listdir(folder) if f.startswith('stride')]
    for stride_file in stride_files:
        stride_data = pd.read_csv(os.path.join(folder, stride_file), encoding='shift_jis')
        if file_size_check(stride_data, max_file_size_kb):
            output_path = os.path.join(output_folder_resized, stride_file)
            stride_data.to_csv(output_path, index=False, encoding='shift_jis')
