
import pandas as pd
import numpy as np

from scipy.interpolate import interp1d

# CSVファイルのパス
sort_csv_path = 'C:/Users/human/Oba_卒業研究A/2023年度歩容測定実験/venus3d_data/sorted_data/sort_SZZY_P_20230801_134432_01.csv'


# DataFrameのカラム名を表示
# CSVファイルを読み込み（low_memory=Falseオプションを設定）
df = pd.read_csv(sort_csv_path, skiprows = 16)
print(df.columns)

# 「*」を欠損値として扱うためにNaNに置換
df.replace('*', np.nan, inplace = True)

# 元データに含まれるNaNの数を数え上げてprint
num_missing_values = df.isna().sum().sum()
print(f"元データに含まれる欠損値の数 : {num_missing_values}")

# 時間列をfloat型に変換
df['Time'] = df['Time'].astype(float)

# スプライン補間関数を定義
def interpolate_spline_with_missing(x, y):
    # 欠損値のインデックスを取得
    missing_indices = np.where(np.isnan(y))[0]
    
    # スプライン補間関数を作成
    f = interp1d(x[~np.isnan(y)], y[~np.isnan(y)], kind = 'cubic', fill_value = 'extrapolate')
    
    # 欠損値を補間
    y[missing_indices] = f(x[missing_indices])
    return y

'''
# 各列に対して欠損値をスプライン補間
for col in df.columns:
    if col != 'Time':  # 'Time'列以外はスプライン補間を実行
        df[col] = df[col].replace('*', np.nan).astype(float)# '*'をNaNに置換してから数値に変換
        df[col] = interpolate_spline_with_missing(df['Time'], df[col])
'''
for col in df.columns:
    if col != 'Time':
        df[col] = df[col].replace('*', np.nan).astype(float)
        
        # 有効なデータが存在するかチェック
        if not df[col].isnull().all():
            df[col] = interpolate_spline_with_missing(df['Time'].values, df[col].values)


# スプライン補間したデータを新しいCSVファイルに出力
output_csv_path = 'C:/Users/human/Oba_卒業研究A/2023年度歩容測定実験/venus3d_data/spline_interpolation_data/spline_SZZY_P_20230801_134432_01.csv'
df.to_csv(output_csv_path, index = False)

'''
import os
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d

# 入力フォルダと出力フォルダのパス
input_folder = 'C:/Users/human/Oba_卒業研究A/2023年度歩容測定実験/venus3d_data/sorted_data'
output_folder = 'C:/Users/human/Oba_卒業研究A/2023年度歩容測定実験/venus3d_data/spline_interpolation_data'

# 入力フォルダ内のCSVファイルを処理
for filename in os.listdir(input_folder):
    if filename.startswith('sort') and filename.endswith('.csv') and not filename.endswith('FP.csv'):
        input_csv_path = os.path.join(input_folder, filename)
        output_csv_filename = 'spline_' + filename
        output_csv_path = os.path.join(output_folder, output_csv_filename)
        
        # CSVファイルの処理
        df = pd.read_csv(input_csv_path, skiprows=16)
        df.replace('*', np.nan, inplace=True)
        df['Time'] = df['Time'].astype(float)
        
        def interpolate_spline_with_missing(x, y):
            missing_indices = np.where(np.isnan(y))[0]
            f = interp1d(x[~np.isnan(y)], y[~np.isnan(y)], kind='cubic', fill_value='extrapolate')
            y[missing_indices] = f(x[missing_indices])
            return y
        
        # 各列に対して欠損値をスプライン補間
        for col in df.columns:
            if col != 'Time':  # 'Time'列以外はスプライン補間を実行
                df_copy = df.copy()  # DataFrameのコピーを作成
                df_copy[col] = df_copy[col].replace('*', np.nan).astype(float)  # '*'をNaNに置換してから数値に変換
                
                sorted_df = df_copy.sort_values(by='Time')  # 時間でソート
                sorted_df.reset_index(drop=True, inplace=True)  # インデックスをリセット
                
                sorted_df[col] = interpolate_spline_with_missing(sorted_df['Time'], sorted_df[col])
                
                # 処理が完了したら、ソートを元に戻す
                df[col] = sorted_df.sort_index()['Time']
        df.to_csv(output_csv_path, index=False)
        print(f"処理済み: {filename} -> {output_csv_filename}")
'''