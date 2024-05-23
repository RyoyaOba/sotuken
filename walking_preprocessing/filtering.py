
import os
import pandas as pd
import numpy as np
from scipy.signal import butter, filtfilt
#バターワースフィルタ（Butterworth）掛ける

def butter_lowpass_filter(data, cutoff_freq, sampling_freq, order = 4):
    nyquist_freq = 0.5 * sampling_freq
    normal_cutoff = cutoff_freq / nyquist_freq
    b, a = butter(order, normal_cutoff, btype = 'low', analog = False)
    filtered_data = filtfilt(b, a, data)
    return filtered_data

def apply_butter_lowpass_filter(input_folder, output_folder, cutoff_freq = 30):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    file_list = os.listdir(input_folder)

    for file in file_list:
        try:
            input_path = os.path.join(input_folder, file)
            output_path = os.path.join(output_folder, "filtered_" + file)

            if file.endswith(".csv"):
                df = pd.read_csv(input_path, encoding = 'shift_jis')

                     # カラム名に対応するカットオフ周波数を設定します
                column_cutoff_freq = {
                    'FP1-Fx(N/kg)': cutoff_freq,
                    'FP1-Fy(N/kg)': cutoff_freq,
                    'FP1-Fz(N/kg)': cutoff_freq,
                    'FP2-Fx(N/kg)': cutoff_freq,
                    'FP2-Fy(N/kg)': cutoff_freq,
                    'FP2-Fz(N/kg)': cutoff_freq
                }

                for col in df.columns:
                    if col != 'time' and col in column_cutoff_freq:# 'time'列はフィルタリング対象外にする場合
                        data_to_filter = df[col].values
                        filtered_data = butter_lowpass_filter(data_to_filter, cutoff_freq, sampling_freq = 1/0.004)#サンプリング周波数250[Hz]
                        df[col] = filtered_data

                df.to_csv(output_path, index = False, encoding = 'shift_jis')
        except Exception as e:
            print(f"ファイルの処理中にエラーが発生しました。ファイル名: {file}")
            print(f"エラー詳細: {str(e)}")

    print("バターワースローパスフィルタの適用が完了しました。")

# 入力フォルダと出力フォルダのパス
input_folder = r'C:\Users\human\Oba_卒業研究A\2023年度歩容測定実験\venus3d_data\linked_data'
output_folder = r'C:\Users\human\Oba_卒業研究A\2023年度歩容測定実験\venus3d_data\filtered_data'
apply_butter_lowpass_filter(input_folder, output_folder, column_cutoff_freq = 56)
