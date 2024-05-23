
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from typing import Any

from tqdm import tqdm
import time

class labelDataset:
 
    def __init__(self, folder_path: str, seed: int = 123, single_y: bool = False) -> None:
        self.seed = seed
        tf.random.set_seed(self.seed)
        self.folder_path = folder_path
        self.datasets = []
        self.data_x = []
        self.data_y = []


        self.label_0_count = 0
        self.label_1_count = 0

        #count = 0

        self.data01 = []
        self.data10 = []

        for label_folder in tqdm(os.listdir(self.folder_path)[::-1]):#スライス（逆順から指定する）
            label = None
            if label_folder == 'sotukenB_thumb_2':
                label = 0
            elif label_folder == 'sotukenB_thumb_1':
                label = 1
            if label is not None:  # labelがNoneでない場合にのみ処理を行うように修正
                label_folder_path = os.path.join(self.folder_path, label_folder)

                # フォルダかどうかを判別
                if not os.path.isdir(label_folder_path):
                    continue  # フォルダでない場合はスキップ

                for csv_file in os.listdir(label_folder_path)[:3000]:
                    if csv_file.endswith('.csv'):
                        csv_path = os.path.join(label_folder_path, csv_file)
                        self.datasets.append({
                            "label": label,
                            "csv_path": csv_path,
                            "csv_name" : csv_file
                        })

                        # CSVファイルからデータを読み込み
                        df = pd.read_csv(csv_path, encoding='latin-1')


                        # Exclude the 21st and 22nd columns (columns are zero-based)
                        excluded_columns = [20, 21]
                        selected_columns = [col for col in range(len(df.columns)) if col not in excluded_columns]

                        # Select only the desired columns
                        df = df.iloc[:, selected_columns]
                        # 2行目以降をデータとして使用
                        # data = df.iloc[1:].values
                        # 最初の行をラベルとして使用
                        label_data = df.iloc[0].values
                        # ラベルごとにカウントを増やす
                        if label == 0:
                            self.label_0_count += 1
                        elif label == 1:
                            self.label_1_count += 1

                        # 2行目以降をデータとして使用
                        data = df.iloc[1:].values
                        
                        #---------------


                        # データの形状を調整
                        #max_sequence_length = 400  # データの最大シーケンス長を指定
                        #if data.shape[0] < max_sequence_length:
                            # データが最大シーケンス長未満の場合、ゼロでパディング
                        #    pad_width = max_sequence_length - data.shape[0]
                        #    data = np.pad(data, [(0, pad_width), (0, 0)], mode='constant')

                        # データを統合
                        # self.data_x.append(data)


                        #-------------------------
                        # 12/29

                        # # CSVファイルの名前とデータをタプルとして追加
                        # if label == 0:
                        #     self.data10.append((csv_file, data))
                        # elif label == 1:
                        #     self.data01.append((csv_file, data))

                        # #-------------------------
                        if label == 0:
                            self.data10.append(data)
                            #self.data10.append((csv_file, data))
                        elif label == 1:
                            self.data01.append(data)
                            #self.data01.append((csv_file, data))
                        # # ラベルデータ追加
                        # self.data_y.append(label)

                        '''count += 1
                        if count >= 2000:
                            break  # 100点読み込んだらループを終了
            if count >= 2000:
                break

'''       
        # data_length = max([len(i) for i in self.data_x])
        # 次元（独立なデータ数）
        # data_width = len(self.data_x)

        # パディングで長さを揃える（意味はなかった）
        # x_pad: np.ndarray = tf.keras.preprocessing.sequence.pad_sequences(
        #     self.data_x, padding="post", dtype=np.float32
        # )

        # ラベルを適切な形式に変換
        if single_y:
            y_pad = np.array(self.data_y, dtype=np.float32)

        else:
            num_labels = 2
            y_pad = np.zeros((len(self.data_y), num_labels), dtype=np.float32)
            for i, label_data in enumerate(self.data_y):
                label = 0 if np.any(label_data == 0) else 1
                # クラス0なら[1, 0]、クラス1なら[0, 1]に直す
                one_hot_label = [1.0, 0.0] if label == 0 else [0.0, 1.0]
                y_pad[i] = one_hot_label


        # データセットの生成
        # self.dataset = tf.data.Dataset.from_tensor_slices(
        #     # (x_pad, y_pad)
        #     x_pad
        # )

        # ラベル0とラベル1のデータ数を表示
        print("Label 0 count:", self.label_0_count)
        print("Label 1 count:", self.label_1_count)

    def __call__(self) -> Any:
        # dataset_10 = self.dataset.filter(lambda x, y: tf.reduce_all(tf.equal(y, [1.0, 0.0])))
        # dataset_01 = self.dataset.filter(lambda x, y: tf.reduce_all(tf.equal(y, [0.0, 1.0])))

        # return dataset_10, dataset_01

        return self.data10, self.data01
        ds10 = tf.data.Dataset.from_tensor_slices(self.data10)#.batch(1).prefetch(tf.data.AUTOTUNE)
        ds01 = tf.data.Dataset.from_tensor_slices(self.data01)#.batch(1).prefetch(tf.data.AUTOTUNE)

        return ds10, ds01

    # def __len__(self):
    #     return len(self.dataset)
if __name__ == '__main__':
    ds = labelDataset('/app/Walking_Clustering/sotukenB_clustering')
    data10, data01 = ds()
    print(data10)

    # data10のCSVファイル名を表示
    print("CSV files in data10:")
    # for data in data10:
    #     if isinstance(data, tuple) and len(data) >= 1:
    #         csv_file = data[0]
    #         print(csv_file)

    # # data01のCSVファイル名を表示
    # print("\nCSV files in data01:")
    # for data in data01:
    #     if isinstance(data, tuple) and len(data) >= 1:
    #         csv_file = data[0]
    #         print(csv_file)