
import argparse

parser = argparse.ArgumentParser(
                    prog='ProgramName',
                    description='What the program does',
                    epilog='Text at the bottom of help')
parser.add_argument('filename')           # positional argument

args = parser.parse_args()

import os
import numpy as np
from fastdtw import fastdtw
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from tqdm import tqdm
import shutil
import multiprocessing

def load_and_preprocess_data(directory, num_samples=5837):
    data = []
    labels = []

    files = os.listdir(directory)
    for file_num in range(1, num_samples + 1):
        file_path = os.path.join(directory, f"SotukenA_data_{file_num}.csv")
        loaded_data = np.genfromtxt(file_path, delimiter=',', skip_header=1, encoding="shift-jis")
        print(f"Loaded file: {file_path}")
        data.append(loaded_data[:, :-1])  # 時間系列データ
        labels.append(loaded_data[0, 19])  # 21列目

    data = np.array(data)
    labels = np.array(labels)
    return data, labels# データ読み込みと前処理

def calculate_dtw_distance(args):
    i, data = args
    distances = []
    for j in range(len(data)):
        distance, _ = fastdtw(data[i], data[j])
        distances.append(distance)
    return distances

def calculate_dtw_distance_matrix_parallel(data):
    num_samples = len(data)
    distance_matrix = np.zeros((num_samples, num_samples))

    with multiprocessing.Pool() as pool:
        args_list = [(i, data) for i in range(num_samples)]
        results = list(tqdm(pool.imap(calculate_dtw_distance, args_list), total=num_samples, desc="Calculating DTW Distances"))

    for i, distances in enumerate(results):
        distance_matrix[i, :] = distances

    return distance_matrix


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


# sseの値を表示
    print("Sum of Squared Errors (SSE) for each number of clusters:")
    for n_clusters, sse_value in enumerate(sse, start=1):
        print(f"Clusters = {n_clusters}: SSE = {sse_value:.4f}")

# エルボー法で選択されたクラスタ数を表示
    print("Selected number of clusters (Elbow Method):", elbow_index)

    # Times New Romanフォントを設定
    plt.rcParams["font.family"] = "Times New Roman"


    plt.figure(figsize=(10, 6))
    plt.plot(range(1, 9), sse, marker='o', color = 'blue', linewidth = 1.4)
    plt.xlabel('Number of clusters', fontsize=18)
    plt.ylabel('Sum of squared errors', fontsize=18)
    plt.xticks(range(1, 9), fontsize=14)
    plt.yticks(fontsize=14)

    plt.grid()
    plt.ticklabel_format(style='plain', axis='y')

    graph_file_path = os.path.join(output_dir, 'elbow_method_graph_R_Fz.png')
    plt.savefig(graph_file_path)  # グラフを保存

    print("Selected number of clusters:", elbow_index)

    return elbow_index

# クラスタごとにデータを保存
def save_clustered_data(data_directory, clusters):
    output_directory = args.filename + '/Oba_卒業研究A/2023年度歩容測定実験/venus3d_data/normalization_data'
    os.makedirs(output_directory, exist_ok=True)

    for cluster_label in np.unique(clusters):
        cluster_indices = np.where(clusters == cluster_label)[0]

        cluster_output_directory = os.path.join(output_directory, f"R_Fz_cluster_{cluster_label + 1}")
        os.makedirs(cluster_output_directory, exist_ok=True)

        for cluster_idx in cluster_indices:
            src_file = os.path.join(data_directory, f"SotukenA_data_{cluster_idx + 1}.csv")
            dest_file = os.path.join(cluster_output_directory, os.path.basename(src_file))
            shutil.copy(src_file, dest_file)

# KMeansクラスタリングを実行
def perform_kmeans_clustering(distance_matrix, num_clusters):
    kmeans = KMeans(n_clusters=num_clusters)
    kmeans.fit(distance_matrix)
    clusters = kmeans.labels_
    return clusters

# メイン関数
if __name__ == "__main__":
    data_directory = args.filename +'/Oba_卒業研究A/2023年度歩容測定実験/venus3d_data/normalization_data/クラスタリング用'
    num_samples = 5837
    max_clusters = 9

    # データ読み込み
    data, labels = load_and_preprocess_data(data_directory, num_samples=num_samples)

        # グラフを正しく保存するための修正
    output_graph_dir = os.path.join(args.filename, 'Oba_卒業研究A/2023年度歩容測定実験/venus3d_data/normalization_data')
    os.makedirs(output_graph_dir, exist_ok=True)

    # 21列目の時系列データを使用してDTW距離行列を計算
    distance_matrix = calculate_dtw_distance_matrix_parallel(data[:, :, 19])

    # 適切なクラスタ数を選択
    optimal_clusters = choose_optimal_clusters(distance_matrix, output_dir=output_graph_dir)

    # KMeansクラスタリングを実行
    clusters = perform_kmeans_clustering(distance_matrix, num_clusters=optimal_clusters)

    print("Cluster labels:", clusters)

    # クラスタリング結果に基づいてデータを保存
    save_clustered_data(data_directory, clusters)
