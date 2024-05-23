import tensorflow as tf
import os
import pandas as pd

class ClassificationDataset:
    def __init__(self, folder_path: str):
        self.folder_path = folder_path
        self.data = []

    def __call__(self):
        thumb_1_path = os.path.join(self.folder_path, '/app/Oba_卒業研究A/2023年度歩容測定実験/venus3d_data/normalization_data/sotukenB_clustering/sotukenB_thumb_1')

        if os.path.exists(thumb_1_path):
            csv_file_path = os.path.join(thumb_1_path, 'SotukenA_data_7650.csv')
            if os.path.exists(csv_file_path):
                # Read and process SotukenA_data_5.csv
                df = pd.read_csv(csv_file_path, encoding='latin-1')
                # Exclude the 21st and 22nd columns (columns are zero-based)
                excluded_columns = [20, 21]
                selected_columns = [col for col in range(len(df.columns)) if col not in excluded_columns]
                df = df.iloc[:, selected_columns]
                label_data = df.iloc[0].values
                data = df.iloc[1:].values
                self.data.append(data)
                return self.data
            else:
                print("SotukenA_data_5.csv not found in sotukenB_thumb_1 folder.")
        else:
            print("sotukenB_thumb_1 folder not found.")


# ds = ClassificationDataset('/app/Oba_卒業研究A/2023年度歩容測定実験/venus3d_data/normalization_data')
# data = ds()

# generator = tf.keras.models.load_model("./generator_f/model_c6")

# gen_data = generator(data)

# print(gen_data)

if __name__ == '__main__':
    ds = ClassificationDataset('/app/Oba_卒業研究A/2023年度歩容測定実験/venus3d_data/normalization_data')
    data = ds()
    print(data)
    # if data:
    #     tf_dataset = tf.data.Dataset.from_tensor_slices(data[0])  # Assuming data[0] holds the desired data
    #     print("TensorFlow Dataset elements:")
    #     for element in tf_dataset.take(5):  # Taking 5 elements for demonstration
    #         print(element)
    # else:
    #     print("No data retrieved.")
