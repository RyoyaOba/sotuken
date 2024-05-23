import sys
sys.path.append("../utils")

import numpy as np
from label_dataset import labelDataset
from sklearn.model_selection import train_test_split

class GanDatasets:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.dataset_10_train = None
        self.dataset_10_val = None
        self.dataset_01_train = None
        self.dataset_01_val = None

    def load_ds(self, sample_size=2622):
        ds = labelDataset(self.data_dir)
        ds_10, ds_01 = ds()

        if sample_size > len(ds_01):
            sample_size = len(ds_01)

        #         # CSVファイル名とデータをsamp_sizeで切り取る
        # data10_samples = ds_10[:sample_size]
        # data01_samples = ds_01[:sample_size]

        dataset_10_samples = ds_10[:sample_size]
        dataset_01_samples = ds_01[:sample_size]

        dataset_10_labels = [[1, 0] for _ in range(sample_size)]
        dataset_01_labels = [[0, 1] for _ in range(sample_size)]

        return np.array(dataset_10_samples), np.array(dataset_01_samples), \
              np.array(dataset_10_labels), np.array(dataset_01_labels)

    def split_data(self, dataset_10_samples, dataset_01_samples):
        train_split_ratio = 0.8
        self.dataset_10_train, self.dataset_10_val = train_test_split(dataset_10_samples, train_size=train_split_ratio, random_state=1)
        self.dataset_01_train, self.dataset_01_val = train_test_split(dataset_01_samples, train_size=train_split_ratio, random_state=1)

        return self.dataset_10_train, self.dataset_10_val, self.dataset_01_train, self.dataset_01_val

if __name__ == "__main__":
    ds = GanDatasets('/app/Oba_卒業研究A/2023年度歩容測定実験/venus3d_data/normalization_data/sotukenB_clustering')
    ds10_samples, ds01_samples, _, _ = ds.load_ds(sample_size=10560)
    train_10, val_10, train_01, val_01 = ds.split_data(ds10_samples, ds01_samples)

    print(val_01)
    print(len(val_01))

    print(f"Dataset 10 Train size: {len(train_10)}")
    print(f"Dataset 10 Validation size: {len(val_10)}")
    # print("つまずきにくい Train files:")
    # for file_name, _ in train_10:
    #     print(file_name)
    # print("つまずきにくい Validation files:")
    # for file_name, _ in val_10:
    #     print(file_name)

    # print(f"Dataset 01 Train size: {len(train_01)}")
    # print(f"Dataset 01 Validation size: {len(val_01)}")
    # print("つまずきやすい Train files:")
    # for file_name, _ in train_01:
    #     print(file_name)
    # print("つまずきやすい Validation files:")
    # for file_name, _ in val_01:
    #     print(file_name)
