
import os
import pandas as pd

def split_data_by_time(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    file_list = os.listdir(input_folder)

    for file in file_list:
        try:
            input_path = os.path.join(input_folder, file)

            if file.endswith('.csv'):
                df = pd.read_csv(input_path, encoding='shift_jis')

                # 条件1: 7501行から22501行までのデータ
                data_condition1 = df.iloc[7500:22500, :]
                output_file_condition1 = 'condition1_' + file
                output_path_condition1 = os.path.join(output_folder, output_file_condition1)
                data_condition1.to_csv(output_path_condition1, index=False, encoding='shift_jis')

                # 条件2: 30001行から45001行までのデータ
                data_condition2 = df.iloc[30000:45000, :]
                output_file_condition2 = 'condition2_' + file
                output_path_condition2 = os.path.join(output_folder, output_file_condition2)
                data_condition2.to_csv(output_path_condition2, index=False, encoding='shift_jis')

                # 条件3: 52501行から67501行までのデータ
                data_condition3 = df.iloc[52500:67500, :]
                output_file_condition3 = 'condition3_' + file
                output_path_condition3 = os.path.join(output_folder, output_file_condition3)
                data_condition3.to_csv(output_path_condition3, index=False, encoding='shift_jis')

        except Exception as e:
            print(f"ファイルの処理中にエラーが発生しました。ファイル名: {file}")
            print(f"エラー詳細: {str(e)}")

    print("ファイルの分割が完了しました。")

# 入力フォルダと出力フォルダのパス
input_folder = r'C:\Users\human\Oba_卒業研究A\2023年度歩容測定実験\venus3d_data\filtered_data'
output_folder = r'C:\Users\human\Oba_卒業研究A\2023年度歩容測定実験\venus3d_data\condition_split_data'

split_data_by_time(input_folder, output_folder)
