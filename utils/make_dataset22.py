
import os

import numpy as np
import pandas as pd
import tensorflow as tf

from typing import Any

class CustomDataset:
    def __init__(self, filename:str, path:str=None, seed:int=1, single_y:bool=False) -> None:
        self.seed = seed

        #乱数シードの固定
        tf.random.set_seed(self.seed)

        self.path = filename.replace('SotukenA_data.csv', '') if path is None else path

        ds = pd.read_csv(filename, index_col = 0)

        datasets =[]
        data_x, data_y = [], []
        for _, raw in ds.iterrows():
            date = raw["date"]
            cdt = raw["condition"]
            str_n = raw["stride"]


            for i in range(str_n):
                emg_file = f"{self.path}/stride_data/{date}/{cdt}/stride{i}_sEMG.npz"
                mocap_file = f"{self.path}/stride_data/{date}/{cdt}/stride{i+1}_mocap.npz"

                datasets.append({
                    "sEMG" : emg_file,
                    "angle": mocap_file
                })

                data_x.append(self._load_file(emg_file)[:,:-1])
                if single_y:
                    data_y.append(self._load_file(mocap_file)[:,0])
                else:
                    data_y.append(self._load_file(mocap_file))


    #データセットの形状をパディングで揃える
        
        #長さ（時間方向）
        data_length = max([len(i) for i in data_x])
        #次元（独立なデータ数）
        data_width = len(data_x)

        #パディングで長さを揃える
        x_pad : np.ndarray = tf.keras.preprocessing.sequence.pad_sequences(
            data_x, padding = "post", dtype = np.float32
        )
        if single_y:
            y_pad : np.array = tf.keras.preprocessing.sequence.pad_sequences(
                data_y, padding="post", dtype=np.float32
            )

        else:
            y_pad = np.zeros_like(x_pad)
            print(y_pad.shape)
            print(data_y)
            for i, take in enumerate(data_y):
                for j, row in enumerate(take):
                    for k, value in enumerate(row):
                        y_pad[i][j][k] = value
        
        #データセットの生成
        self.dataset = tf.data.Dataset.from_tensor_slices(
            (x_pad, y_pad)
        )
    
        print(data_x.shape, data_y.reshape((-1, 405, 1)).shape)


    def __call__(self, size: int, batch_size: int=10, train_split: float=0.8, val_split: float=0.1, test_split: float=0.1) -> Any:
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

        train_ds = ds.take(train_size)
        val_ds = ds.skip(train_size).take(val_size)
        test_ds = ds.skip(train_size).skip(val_size)

        return train_ds, val_ds, test_ds
    
    def _load_file(self, filename:str, label:str="data"):
        '''
        '.npz' ファイルの読み込み
        
        Parameters :
            filename(str) : ファイル名
            label(str) : optional, defalut is 'data'
            
        Returns:
            data(numpy.ndarray)
        
        '''

        with np.load(filename, allow_pickle=True)as d:
            data = d[label]

        return data
    
    def __len__(self):
        return len(self.dataset)
    

if __name__ == '__main__':
    ds = CustomDataset('/mnt/usb/Oba_卒業研究A/2023年度歩容測定実験/venus3d_data/normalization_data/')
    size = ds.__len__()
    print(size)
    train, val, test = ds(size, batch_size=1)

    print(train)
    print(len(train), len(val), len(test))

    x, y = next(iter(train))

