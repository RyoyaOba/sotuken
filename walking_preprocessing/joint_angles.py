
import pandas as pd
import numpy as np
import os

# 関節角度を計算するための関数を作成
def calculate_angles(v1, v2, plane):
    v1_df = pd.DataFrame(v1, columns = ['X', 'Y', 'Z'])
    v2_df = pd.DataFrame(v2, columns = ['X', 'Y', 'Z'])

    # ベクトルの内積を計算
    dot_product = v1_df['X'] * v2_df['X'] + v1_df['Y'] * v2_df['Y'] + v1_df['Z'] * v2_df['Z']

    # ベクトルの大きさを計算
    norm_v1 = np.linalg.norm(v1, axis = 1)
    norm_v2 = np.linalg.norm(v2, axis = 1)

    # 内積から角度を計算（ラジアンを度に変換）
    angle_rad = np.arccos(dot_product / (norm_v1 * norm_v2))
    angle_deg = np.degrees(angle_rad)

    if plane == 'sagittal':
        #矢状面の関節角度を計算（Y-Z平面上の角度）
        angles = np.arctan2(v1_df['Y'] - v2_df['Y'], v1_df['Z'] - v2_df['Z'])
    elif plane == 'frontal':
        # 前額面の関節角度を計算（X-Z平面上の角度）
        angles = np.arctan2(v1_df['X'] - v2_df['X'], v1_df['Z'] - v2_df['Z'])
    else:
        raise ValueError("Invalid plane type. Use 'sagittal' or 'frontal'.")
    
    return np.degrees(angles)

# 入力フォルダのパスを指定
input_folder_path = 'C:/Users/human/Oba_卒業研究A/2023年度歩容測定実験/venus3d_data/spline_interpolation_data/'

# 出力フォルダのパスを指定
output_folder_path = 'C:/Users/human/Oba_卒業研究A/2023年度歩容測定実験/venus3d_data/joint_angles/'

# 基準角度を保存する変数
reference_trunk_angle = None
reference_left_hip_joint_angle = None
reference_right_hip_joint_angle = None
reference_left_knee_joint_angle = None
reference_right_knee_joint_angle = None
reference_left_ankle_joint_angle = None
reference_right_ankle_joint_angle = None

for filename in os.listdir(input_folder_path):
    if filename.endswith('csv'):
        csv_file_path = os.path.join(input_folder_path, filename)
        spline_data = pd.read_csv(csv_file_path)

        # ベクトル座標を取得
        head_top = spline_data[['1(X)', '1(Y)', '1(Z)']].values
        sacrum = spline_data[['10(X)', '10(Y)', '10(Z)']].values
        shoulder_mid = (spline_data[['4(X)', '4(Y)', '4(Z)']].values + spline_data[['5(X)', '5(Y)', '5(Z)']].values) / 2
        left_ilac_crest = spline_data[['11(X)', '11(Y)', '11(Z)']].values
        right_ilac_crest = spline_data[['12(X)', '12(Y)', '12(Z)']].values

        left_greater_trochanter = spline_data[['13(X)', '13(Y)','13(Z)' ]].values
        right_greater_trochanter = spline_data[['14(X)', '14(Y)','14(Z)' ]].values
        

        left_lateral_knee = spline_data[['15(X)', '15(Y)', '15(Z)']].values
        right_lateral_knee = spline_data[['16(X)', '16(Y)', '16(Z)']].values
        left_lateral_malleolus = spline_data[['17(X)', '17(Y)', '17(Z)']].values
        right_lateral_malleolus = spline_data[['18(X)', '18(Y)', '18(Z)']].values
        left_first_matatarsal = spline_data[['19(X)', '19(Y)', '19(Z)']].values
        right_first_matatarsal = spline_data[['20(X)', '20(Y)', '20(Z)']].values

        # 体幹角度を計算
        v1 = head_top - shoulder_mid 
        v2 = sacrum - shoulder_mid 

       # 矢状面の体幹角度を計算（X-Z平面上の角度）
        trunk_angle_sagittal = calculate_angles(v1, v2, plane = 'sagittal')
        # 前額面の体幹角度を計算（Y-Z平面上の角度）
        trunk_angle_frontal = calculate_angles(v1, v2, plane='frontal')

        # 左股関節角度を計算
        v3 = left_ilac_crest - left_greater_trochanter
        v4 = left_lateral_knee - left_greater_trochanter

        left_hip_joint_angle_sagittal = calculate_angles(v3, v4, plane = 'sagittal') 
        left_hip_joint_angle_frontal = calculate_angles(v3, v4, plane = 'frontal') 

        # 右股関節角度を計算
        v5 = right_ilac_crest - right_greater_trochanter
        v6 = right_lateral_knee - right_greater_trochanter

        right_hip_joint_angle_sagittal = calculate_angles(v5, v6, plane = 'sagittal')
        right_hip_joint_angle_frontal = calculate_angles(v5, v6, plane = 'frontal')
        
        # 左膝関節角度を計算
        v7 = left_lateral_knee - left_greater_trochanter
        v8 = left_lateral_malleolus - left_lateral_knee
        
        left_knee_joint_angle_sagittal = calculate_angles(v7, v8, plane = 'sagittal')
        left_knee_joint_angle_frontal = calculate_angles(v7, v8, plane = 'frontal')
        
        # 右膝関節角度を計算
        v9 = right_lateral_knee - right_greater_trochanter
        v10 = right_lateral_malleolus - right_lateral_knee

        right_knee_joint_angle_sagittal = calculate_angles(v9, v10, plane = 'sagittal')
        right_knee_joint_angle_frontal = calculate_angles(v9, v10, plane = 'frontal')

        # 左足関節角度を計算
        v11 = left_lateral_knee - left_lateral_malleolus
        v12 = left_first_matatarsal - left_lateral_malleolus

        left_ankle_joint_angle_sagittal = calculate_angles(v11, v12, plane = 'sagittal')
        left_ankle_joint_angle_frontal = calculate_angles(v11, v12, plane = 'frontal')
        
        # 右足関節角度を計算
        v13 = right_lateral_knee - right_lateral_malleolus
        v14 = right_first_matatarsal - right_lateral_malleolus

        right_ankle_joint_angle_sagittal = calculate_angles(v13, v14, plane = 'sagittal')
        right_ankle_joint_angle_frontal = calculate_angles(v13, v14, plane = 'frontal')
       
        # 体幹角度を計算
        '''trunk_angle = calculate_angles(v1, v2)
        # 左股関節角度を計算
        left_hip_joint_angle = calculate_angles(v3, v4)
        # 右股関節角度を計算
        right_hip_joint_angle = calculate_angles(v5, v6)
        # 左膝関節角度を計算
        left_knee_joint_angle = calculate_angles(v7, v8)
        # 右膝関節角度を計算
        right_knee_joint_angle = calculate_angles(v9, v10)
        # 左足関節角度を計算
        left_ankle_joint_angle = calculate_angles(v11, v12)
        # 右足関節角度を計算
        right_ankle_joint_angle = calculate_angles(v13, v14)'''

    

        # 基準角度を最初の10秒間のデータから算出（最初の2500データポイント）
        if reference_trunk_angle is None:
            reference_trunk_angle_sagittal = trunk_angle_sagittal[:2500]
            reference_trunk_angle_frontal = trunk_angle_frontal[:2500]
            
            reference_left_hip_joint_angle_sagittal = left_hip_joint_angle_sagittal[:2500]
            reference_left_hip_joint_angle_frontal = left_hip_joint_angle_frontal[:2500]

            reference_right_hip_joint_angle_sagittal = right_hip_joint_angle_sagittal[:2500]
            reference_right_hip_joint_angle_frontal = right_hip_joint_angle_frontal[:2500]
            
            reference_left_knee_joint_angle_sagittal = left_knee_joint_angle_sagittal[:2500]
            reference_left_knee_joint_angle_frontal = left_knee_joint_angle_frontal[:2500]

            reference_right_knee_joint_angle_sagittal = right_knee_joint_angle_sagittal[:2500]
            reference_right_knee_joint_angle_frontal = right_knee_joint_angle_frontal[:2500]
            
            reference_left_ankle_joint_angle_sagittal = left_ankle_joint_angle_sagittal[:2500]
            reference_left_ankle_joint_angle_frontal = left_ankle_joint_angle_frontal[:2500]

            reference_right_ankle_joint_angle_sagittal = right_ankle_joint_angle_sagittal[:2500]
            reference_right_ankle_joint_angle_frontal =  right_ankle_joint_angle_frontal[:2500] 


        
        # 差分角度を計算（基準角度は最初の2500データポイントの平均値を用いる）
        diff_trunk_angle_sagittal = trunk_angle_sagittal - reference_trunk_angle_sagittal.mean()
        diff_trunk_angle_frontal = trunk_angle_frontal - reference_trunk_angle_frontal.mean()
        diff_left_hip_joint_angle_sagittal = left_hip_joint_angle_sagittal - reference_left_hip_joint_angle_sagittal.mean()
        diff_left_hip_joint_angle_frontal = left_hip_joint_angle_frontal - reference_left_hip_joint_angle_frontal.mean()
        diff_right_hip_joint_angle_sagittal = right_hip_joint_angle_sagittal - reference_right_hip_joint_angle_sagittal.mean()
        diff_right_hip_joint_angle_frontal = right_hip_joint_angle_frontal - reference_right_hip_joint_angle_frontal.mean()
        diff_left_knee_joint_angle_sagittal = left_knee_joint_angle_sagittal - reference_left_knee_joint_angle_sagittal.mean()
        diff_left_knee_joint_angle_frontal = left_knee_joint_angle_frontal - reference_left_knee_joint_angle_frontal.mean()
        diff_right_knee_joint_angle_sagittal = right_knee_joint_angle_sagittal - reference_right_knee_joint_angle_sagittal.mean()
        diff_right_knee_joint_angle_frontal = right_knee_joint_angle_frontal - reference_right_knee_joint_angle_frontal.mean()
        diff_left_ankle_joint_angle_sagittal = left_ankle_joint_angle_sagittal - reference_left_ankle_joint_angle_sagittal.mean()
        diff_left_ankle_joint_angle_frontal = left_ankle_joint_angle_frontal - reference_left_ankle_joint_angle_frontal.mean()
        diff_right_ankle_joint_angle_sagittal = right_ankle_joint_angle_sagittal - reference_right_ankle_joint_angle_sagittal.mean()
        diff_right_ankle_joint_angle_frontal = right_ankle_joint_angle_frontal - reference_right_ankle_joint_angle_frontal.mean()



        # 新しいデータフレームの作成
        output_data = pd.DataFrame({'体幹角度_矢状面': diff_trunk_angle_sagittal})
        output_data['体幹角度_前額面']= diff_trunk_angle_frontal
        output_data['左股関節角度_矢状面'] = diff_left_hip_joint_angle_sagittal
        output_data['左股関節角度_前額面'] = diff_left_hip_joint_angle_frontal
        output_data['右股関節角度_矢状面'] = diff_right_hip_joint_angle_sagittal
        output_data['右股関節角度_前額面'] = diff_right_hip_joint_angle_frontal
        
        output_data['左膝関節角度_矢状面'] = diff_left_knee_joint_angle_sagittal
        output_data['左膝関節角度_前額面'] = diff_left_knee_joint_angle_frontal
        output_data['右膝関節角度_矢状面'] = diff_right_knee_joint_angle_sagittal
        output_data['右膝関節角度_前額面'] = diff_right_knee_joint_angle_frontal
        output_data['左足関節角度_矢状面'] = diff_left_ankle_joint_angle_sagittal
        output_data['左足関節角度_前額面'] = diff_left_ankle_joint_angle_frontal
        output_data['右足関節角度_矢状面'] = diff_right_ankle_joint_angle_sagittal
        output_data['右足関節角度_前額面'] = diff_right_ankle_joint_angle_frontal

        # 出力ファイル名を作成
        output_file_name = "angle_diff" + filename.split("spline", 1)[-1]
        output_csv_path = os.path.join(output_folder_path, output_file_name)

        # CSVファイルに出力
        output_data.to_csv(output_csv_path, index = False, encoding = 'shift_jis')