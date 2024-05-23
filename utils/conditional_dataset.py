import os
import tensorflow as tf

class ConditionalDataset:
    def __init__(self, folder_path: str) -> None:
        self.folder_path = folder_path

    def __call__(self) -> tuple:
        dataset_0 = self._load_dataset('tc_cluster_1')
        dataset_1 = self._load_dataset('tc_cluster_2')
        return dataset_0, dataset_1

    def _load_dataset(self, label_folder: str) -> tf.data.Dataset:
        label_folder_path = os.path.join(self.folder_path, label_folder)

        if not os.path.isdir(label_folder_path):
            raise ValueError(f"Path {label_folder_path} is not a directory.")

        file_paths = [os.path.join(label_folder_path, csv_file) for csv_file in os.listdir(label_folder_path) if csv_file.endswith('.csv')]

        # Create a dataset from file paths
        file_paths_dataset = tf.data.Dataset.from_tensor_slices(file_paths)

        # Load and preprocess each CSV file
        dataset = file_paths_dataset.flat_map(lambda file_path: tf.data.TextLineDataset(file_path).skip(1).map(lambda line: self._parse_csv(line, label_folder)))

        return dataset

    def _parse_csv(self, line, label_folder: str) -> tuple:
        # Assuming 2 columns in your CSV, adjust accordingly
        record_defaults = [tf.constant([], dtype=tf.float32)] * 2
        values = tf.io.decode_csv(line, record_defaults, field_delim=',')

        # Pad the sequences to a maximum length
        max_sequence_length = 399
        data = tf.pad(tf.reshape(values[0], [-1, 1]), paddings=[[0, max_sequence_length - tf.shape(values[0])[0]], [0, 0]])

        # Add a constant label column based on the label folder
        label = 0 if label_folder == 'tc_cluster_1' else 1

        return data, label

# フォルダのパスを指定してデータセットを取得
folder_path = '/app/Oba_卒業研究A/2023年度歩容測定実験/venus3d_data/normalization_data/'
ds = ConditionalDataset(folder_path)
dataset_0, dataset_1 = ds()
