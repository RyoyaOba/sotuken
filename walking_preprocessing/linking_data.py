
import os
import pandas as pd

# ファイルのパスとフォルダの設定
input_folder_angles = 'C:/Users/human/Oba_卒業研究A/2023年度歩容測定実験/venus3d_data/joint_angles/'
input_folder_distances = r'C:\Users\human\Oba_卒業研究A\2023年度歩容測定実験\venus3d_data\拇趾床間距離'
input_folder_forces = r'C:\Users\human\Oba_卒業研究A\2023年度歩容測定実験\venus3d_data\normalization_floor_reaction_force'
output_folder = r'C:\Users\human\Oba_卒業研究A\2023年度歩容測定実験\venus3d_data\linked_data'

# normalization_joint_anglesフォルダ内のファイルリストを取得
angle_files = os.listdir(input_folder_angles)

# 拇趾床間距離フォルダ内のファイルリストを取得
distance_files = os.listdir(input_folder_distances)

# normalization_floor_reaction_forceフォルダ内のファイルリストを取得
force_files = os.listdir(input_folder_forces)

# ファイルを1つずつ処理
for angle_file in angle_files:
    try:
        # normalization_angle_diff_以降の4文字のアルファベットを取得
        angle_suffix = angle_file.split("angle_diff_", 1)[1][:4]
        
        # 対応する拇趾床間距離ファイルを検索
        matching_distance_file = [distance_file for distance_file in distance_files if distance_file.startswith("tc_" + angle_suffix)]

        # 対応する床反力ファイルを検索
        matching_force_file = [force_file for force_file in force_files if force_file.startswith("normalization_force_" + angle_suffix)]

        if len(matching_distance_file) == 1 and len(matching_force_file) == 1:
            # 関節角度データを読み込み（エンコーディングを指定してUTF-8以外の文字をサポート）
            angle_data = pd.read_csv(os.path.join(input_folder_angles, angle_file), encoding='shift_jis')
            
            # 拇趾床間距離データを読み込み（エンコーディングを指定してUTF-8以外の文字をサポート）
            distance_data = pd.read_csv(os.path.join(input_folder_distances, matching_distance_file[0]), encoding='shift_jis')
            
            # 床反力データを読み込み（エンコーディングを指定してUTF-8以外の文字をサポート）
            force_data = pd.read_csv(os.path.join(input_folder_forces, matching_force_file[0]), encoding='shift_jis')
            
            # 3つのデータを横一列に連結
            merged_data = pd.concat([angle_data, force_data, distance_data], axis=1)
            
            # 新しいファイル名を作成（例：normalization_angle_diff_1.csv → linking_1.csv）
            linked_file_name = angle_file.replace("angle_diff", "linking")
            
            # 連結したデータを新しいフォルダに保存
            merged_data.to_csv(os.path.join(output_folder, linked_file_name), index=False, encoding='shift_jis')
        else:
            print(f"関節角度ファイル '{angle_file}' に対応する拇趾床間距離ファイルまたは床反力ファイルが見つかりませんでした。")
    except Exception as e:
        print(f"ファイルの連結中にエラーが発生しました。ファイル名: {angle_file}")
        print(f"エラー詳細: {str(e)}")

print("ファイルの連結が完了しました。")
