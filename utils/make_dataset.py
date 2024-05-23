import os
import numpy as np
import pandas as pd
import tensorflow as tf
from typing import Any
#-------------------
# GPU使いたくないとき
#-------------------
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from tqdm import tqdm
import time
from sklearn.model_selection import train_test_split

from sklearn.utils import shuffle

class CustomDataset:
    def __init__(self, folder_path: str, seed: int = 35, single_y: bool = False) -> None:
        self.seed = seed
        tf.random.set_seed(self.seed)
        self.folder_path = folder_path
        self.datasets = []
        self.data_x = []
        self.data_y = []

        self.label_0_count = 0
        self.label_1_count = 0

        # データセットの情報を収集する
        dataset_info = []
        for label_folder in os.listdir(self.folder_path)[::-1]:
            label = None
            if label_folder == 'sotukenB_thumb_1':
                label = 1
            elif label_folder == 'sotukenB_thumb_2':
                label = 0
            if label is not None:
                label_folder_path = os.path.join(self.folder_path, label_folder)
                if not os.path.isdir(label_folder_path):
                    continue

                for csv_file in os.listdir(label_folder_path)[:2315]:
                    if csv_file.endswith('.csv'):
                        csv_path = os.path.join(label_folder_path, csv_file)
                        dataset_info.append({
                            "label": label,
                            "csv_path": csv_path,
                            "csv_name": csv_file
                        })

        # データセットの情報をシャッフルする
        self.datasets = shuffle(dataset_info, random_state=self.seed)

        # データの処理を行う
        for data_info in self.datasets:
            label = data_info["label"]
            csv_path = data_info["csv_path"]
            df = pd.read_csv(csv_path, encoding='latin-1')
            excluded_columns = [20, 21]
            selected_columns = [col for col in range(len(df.columns)) if col not in excluded_columns]
            df = df.iloc[:, selected_columns]
            label_data = df.iloc[0].values
            if label == 0:
                self.label_0_count += 1
            elif label == 1:
                self.label_1_count += 1
            data = df.iloc[1:].values
            self.data_x.append(data)
            self.data_y.append(label)
     
        # data_length = max([len(i) for i in self.data_x])
        # # 次元（独立なデータ数）
        # data_width = len(self.data_x)

        # パディングで長さを揃える（意味はなかった）
        x_pad: np.ndarray = tf.keras.preprocessing.sequence.pad_sequences(
            self.data_x, padding="post", dtype=np.float32
        )

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
        self.dataset = tf.data.Dataset.from_tensor_slices(
            (x_pad, y_pad)
        )

        # ラベル0とラベル1のデータ数を表示
        print("Label 0 count:", self.label_0_count)
        print("Label 1 count:", self.label_1_count)

    def __call__(self, size: int, batch_size: int=1, train_split: float=0.8, val_split: float=0.1, test_split: float=0.1) -> Any:
        '''
        データセットをtrain, val, testに分割
        Parameters:
            size(int) : データセットの長さ
            batch_size(int) : バッチサイズ（未実装）
            train_spplit(float) : 訓練データの割合 defalut is 0.8
            val_split(float) : 評価データの割合 default is 0.1
            test_split(float) : 検証データの割合 default is 0.1

        Returns:
            train_ds, test_ds, val_ds (tensorflow._Dataset)
        
        Samples:
           >>>ds = CustomDataset("/mnt/sotukenA_Nozawa/stride_data.csv")
           >>>train, val, test = ds(ds.__len__(), train_split = 0.8, val_split=0.1, test_split=0.1)       
        '''
        assert (train_split + test_split + val_split) == 1
        #シャッフル
        ds = self.dataset.shuffle(10000, seed = self.seed)

        ds = ds.batch(batch_size)
        ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        train_size = int(train_split * size)
        val_size = int(val_split * size)
        #test_size = int(test_split * size)

        train_ds = ds.take(train_size)
        val_ds = ds.skip(train_size).take(val_size)
        test_ds = ds.skip(train_size).skip(val_size)

        return train_ds, val_ds, test_ds


    def __len__(self):
        return len(self.dataset)

if __name__ == '__main__':
    ds = CustomDataset('/app/Walking_Clustering/sotukenB_clustering')
    size = ds.__len__()
    print(size)
    print("Dataset size:", size)

    train, val, test = ds(size, batch_size=1)
    print("Train dataset size:", len(train))
    print("Validation dataset size:", len(val))
    print("Test dataset size:", len(test))


      # Get counts of label 0 and label 1 CSV files
    label_0_count = ds.label_0_count
    label_1_count = ds.label_1_count
    print("Label 0 CSV count:", label_0_count)
    print("Label 1 CSV count:", label_1_count)

    x, y = next(iter(train))

    for batch in train.take(1):  # 1つのバッチを取得してみるい
        x, y = batch  # 入力データとターゲットデータ
        print("Input data (x):", x)
        print("Target data (y):", y)
    
    # for x, y in train:
    #     print(y)
        print(size)
        train, val, test = ds(size, batch_size=1)

        print(train)
        print(len(train), len(val), len(test))

        # ... (previous code remains the same)

# ... (previous code remains the same)

        # Split the dataset into train, val, and test
        #train, val, test = ds(size, batch_size=1)

        # Initialize counters for label 0 and label 1 in the train and val sets
        label_0_train_count = 0
        label_1_train_count = 0
        label_0_val_count = 0
        label_1_val_count = 0

        label_0_test_count = 0
        label_1_test_count = 0

        # Count label occurrences in the train set
        for _, labels in train:
            # Sum the occurrences of label 0 and label 1
            label_0_train_count += np.sum(labels[:, 0] == 1)  # Assuming label 0 is encoded as [1, 0]
            label_1_train_count += np.sum(labels[:, 1] == 1)  # Assuming label 1 is encoded as [0, 1]

        # Count label occurrences in the validation set
        for _, labels in val:
            # Sum the occurrences of label 0 and label 1
            label_0_val_count += np.sum(labels[:, 0] == 1)  # Assuming label 0 is encoded as [1, 0]
            label_1_val_count += np.sum(labels[:, 1] == 1)  # Assuming label 1 is encoded as [0, 1]


               # Count label occurrences in the validation set
        for _, labels in test:
            # Sum the occurrences of label 0 and label 1
            label_0_test_count += np.sum(labels[:, 0] == 1)  # Assuming label 0 is encoded as [1, 0]
            label_1_test_count  += np.sum(labels[:, 1] == 1)  # Assuming label 1 is encoded as [0, 1]

        # Print the counts of label 0 and label 1 in train and val datasets
        print("Label 0 count in train dataset:", label_0_train_count)
        print("Label 1 count in train dataset:", label_1_train_count)
        print("Label 0 count in validation dataset:", label_0_val_count)
        print("Label 1 count in validation dataset:", label_1_val_count)
        print("Label 0 count in test dataset:", label_0_test_count)
        print("Label 1 count in test dataset:", label_1_test_count)

        label_0_csv_names = []
        label_1_csv_names = []

        label_0_csv_count = 0
        label_1_csv_count = 0
        thumb_1_folder = '/app/Walking_Clustering/sotukenB_clustering/sotukenB_thumb_1'

        thumb_1_csv_files = os.listdir(thumb_1_folder)

        for i, (data, labels) in enumerate(val):
            for j in range(len(labels)):
                csv_name = ds.datasets[i * len(labels) + j]["csv_name"]
                if labels[j][0] == 1:  # Check if label 0 is encoded as [1, 0]
                    label_0_csv_names.append(csv_name)  # Append CSV names for label 0
                    label_0_csv_count += 1
                elif labels[j][1] == 1:  # Check if label 1 is encoded as [0, 1]
                    label_1_csv_names.append(csv_name)  # Append CSV names for label 1
                    label_1_csv_count += 1
        
        print("Matching CSV names with sotukenB_thumb_1 folder:")

        matching_csv_names = [csv_name for csv_name \
                              in label_0_csv_names + label_1_csv_names if csv_name in thumb_1_csv_files]

        for csv_name in matching_csv_names:
            print(csv_name)

        # Count the matching CSV names
        print("Total count of matching CSV names:", len(matching_csv_names))