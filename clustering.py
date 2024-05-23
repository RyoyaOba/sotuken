import os
import numpy as np
from sklearn.cluster import KMeans
from tqdm import tqdm
import shutil

import matplotlib.pyplot as plt
import multiprocessing
import tensorflow as tf


def load_and_preprocess_data(directory, num_samples=10436):
    data = []
    labels = []

    files = os.listdir(directory)
    for file_num in range(1, num_samples + 1):
        file_path = os.path.join(directory, f"SotukenA_data_{file_num}.csv")
        loaded_data = np.genfromtxt(file_path, delimiter=',', skip_header=1, encoding="latin-1")
        print(f"Loaded file: {file_path}")
        data.append(loaded_data[:, :-1])  # 時間系列データ
        labels.append(loaded_data[0, 20])  # 21列目

    data = np.array(data)
    labels = np.array(labels)
    return data, labels# データ読み込みと前処理


def build_autoencoder(input_shape):
    input_data = tf.keras.Input(shape=input_shape)
    encoded = tf.keras.layers.Dense(128, activation='relu')(input_data)  # エンコード層
    encoded = tf.keras.layers.Dense(64, activation='relu')(encoded)
    encoded = tf.keras.layers.Dense(32, activation='relu')(encoded)  # 潜在空間の次元数を32と仮定

    decoded = tf.keras.layers.Dense(64, activation='relu')(encoded)  # デコード層
    decoded = tf.keras.layers.Dense(128, activation='relu')(decoded)
    decoded = tf.keras.layers.Dense(input_shape[0], activation='sigmoid')(decoded)

    autoencoder = tf.keras.Model(input_data, decoded)
    autoencoder.compile(optimizer='adam', loss='mean_squared_error')

    return autoencoder


def train_autoencoder(data):
    input_shape = data.shape[1:]  # データの形状から入力形状を取得
    autoencoder = build_autoencoder(input_shape)
    
    autoencoder.fit(data, data, epochs=200, batch_size=64)  # 仮定されたパラメータで10エポック学習

    return autoencoder

# クラスタごとにデータを保存
def save_clustered_data(data_directory, clusters):
    output_directory ='/app/hinann/2023年度歩容測定実験/venus3d_data/normalization_data/sotukenB_clustering/'
    os.makedirs(output_directory, exist_ok=True)

    for cluster_label in np.unique(clusters):
        cluster_indices = np.where(clusters == cluster_label)[0]

        cluster_output_directory = os.path.join(output_directory, f"sotukenB_thumb_{cluster_label + 1}")
        os.makedirs(cluster_output_directory, exist_ok=True)

        for cluster_idx in cluster_indices:
            src_file = os.path.join(data_directory, f"SotukenA_data_{cluster_idx + 1}.csv")
            dest_file = os.path.join(cluster_output_directory, os.path.basename(src_file))
            shutil.copy(src_file, dest_file)

# 適切なクラスタ数を選択する関数
def choose_optimal_clusters(dtw_distances, output_dir):
    sse = []

    for n_clusters in tqdm(range(1, 9), desc="Elbow Method Progress", unit="clusters", ascii=True, ncols=100):
        kmeans = KMeans(n_clusters=n_clusters, random_state=0)
        kmeans.fit(dtw_distances)
        sse.append(kmeans.inertia_)

    diff_sse = np.diff(sse)
    diff2_sse = np.diff(diff_sse)
    elbow_index = np.argmax(diff2_sse) + 2

    sse = [val * 1e-9 for val in sse] # 10^-9 

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, 9), sse, marker='o', color = 'blue', linewidth = 1.5)
    plt.xlabel('Number of clusters', fontsize=18)
    plt.ylabel('Sum of squared errors', fontsize=18)
    plt.xticks(range(1, 9), fontsize=14)
    plt.yticks(fontsize=14)

    plt.grid()
    plt.ticklabel_format(style='plain', axis='y')

    graph_file_path = os.path.join(output_dir, 'elbow_method.png')
    plt.savefig(graph_file_path)  # グラフを保存

    print("Selected number of clusters:", elbow_index)

    return elbow_index


# メイン関数
if __name__ == "__main__":
    data_directory = '/app/hinann/2023年度歩容測定実験/venus3d_data/normalization_data/sotukenB_clustering/'
    num_samples = 2000  # サンプル数
    max_clusters = 9

    # データ読み込み
    data, labels = load_and_preprocess_data(data_directory, num_samples=num_samples)

    # 21列目の時系列データを抽出（仮定）
    time_series_data = data[:, :, 20]

    # 自己エンコーダの学習
    autoencoder = train_autoencoder(time_series_data)
    # クラスタリングに使う特徴量の抽出
    encoded_data = autoencoder.predict(time_series_data)

        # 適切なクラスタ数を選択
    # 適切なクラスタ数を選択
    optimal_clusters = choose_optimal_clusters(encoded_data, output_dir=data_directory)

    # KMeansクラスタリングを実行
    kmeans = KMeans(n_clusters=max_clusters)
    clusters = kmeans.fit_predict(encoded_data)

    print("Cluster labels:", clusters)

    # クラスタリング結果に基づいてデータを保存
    save_clustered_data(data_directory, clusters)
