
import os
import glob
import shutil

# フォルダとファイルのパスを設定
input_folder = r'C:\Users\human\Oba_卒業研究A\2023年度歩容測定実験\venus3d_data\normalization_data\テスト'
output_folder = r'C:\Users\human\Oba_卒業研究A\2023年度歩容測定実験\venus3d_data\normalization_data\クラスタリング用'

# 出力フォルダが存在しない場合は作成
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# フォルダ内のCSVファイルを取得
csv_files = glob.glob(os.path.join(input_folder, '*.csv'))

# CSVファイルを読み込んで出力
for idx, csv_file in enumerate(csv_files):
    # 新しいファイル名を生成
    new_filename = os.path.join(output_folder, f'data_{idx + 1}.csv')
    
    # ファイルをコピーして出力
    shutil.copy2(csv_file, new_filename)
    print(f'コピー完了: {csv_file} -> {new_filename}')

print('処理が完了しました。')
