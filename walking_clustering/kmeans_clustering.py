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
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.cluster import KMeans  # KMeansクラスタリングを追加
import matplotlib.pyplot as plt
import re
from tqdm import tqdm
import shutil

# ...
# データ読み込みと前処理
def load_and_preprocess_data(directory, num_samples=4411):
    data = []
    labels = []

    # ファイル名から数字部分を抽出するための正規表現
    pattern = r"data_(\d+)\.csv"

    # ファイル名と数字の対応を格納する辞書
    file_number_map = {}

    # ディレクトリ内のファイルを順に処理
    files = os.listdir(directory)
    for file in files:
        match = re.match(pattern, file)
        if match:
            file_number = int(match.group(1))
            file_number_map[file_number] = file

    # ファイル番号を昇順にソート
    sorted_file_numbers = sorted(file_number_map.keys())

    # ソートされたファイル番号の順番にファイルを読み込む
    for file_number in sorted_file_numbers:
        file = file_number_map[file_number]
        file_path = os.path.join(directory, file)
        loaded_data = np.genfromtxt(file_path, delimiter=',', skip_header=1, encoding="shift-jis")
        print(f"Loaded file: {file_path}")
        data.append(loaded_data[:, :-1])  # 時間系列データ
        labels.append(loaded_data[0, 20])  # 21列目

        if len(data) == num_samples:
            break


    data = np.array(data)
    labels = np.array(labels)
    return data, labels



def determine_num_clusters(linkage_matrix):
        # エルボー法によるクラスタ数の決定
    last_merges = linkage_matrix[-10:, 2]  # 最後の10回のマージの距離
    differences = np.diff(last_merges)  # 直前のマージの距離の差分
    differences_ratio = differences[:-1] / differences[1:]  # 距離の差分の比率

    # デバッグ用：differences_ratioの値を表示
    print("differences_ratio:", differences_ratio)
    # エルボー法の結果をプロット
    font_size_title = 16
    font_size_label = 14
    font_size_ticks = 12

    # エルボー法の結果をプロット
    plt.figure(figsize=(12, 8))
    plt.plot(range(1, len(differences_ratio) + 1), differences_ratio, marker='o')
    #plt.title("Sum of squared errors", fontsize=font_size_title)
    plt.xlabel("Number of Clusters", fontsize=font_size_label)
    plt.ylabel("Ratio of Differences in Distances", fontsize=font_size_label)
    plt.xticks(range(1, len(differences_ratio) + 1), fontsize=font_size_ticks)
    plt.grid(True)

    output_directory = args.filename + '/Oba_卒業研究A/2023年度歩容測定実験/venus3d_data/normalization_data'

    # プロットを保存
    output_file_name = "elbow_plot_kmeans.png"
    output_file_path = os.path.join(output_directory, output_file_name)
    plt.savefig(output_file_path)

    
    plt.show()

    #plt.savefig(os.path.join(output_directory, "elbow_plot.png"))
    #plt.show()


    # 比率が最も急峻に変化するクラスタ数を返す
    return np.argmax(differences_ratio) + 2

# クラスタリングとCSV出力
def cluster_and_output(data_directory, output_directory):
    # データ読み込みと前処理
    data, labels = load_and_preprocess_data(data_directory, num_samples=4411)
    
    # ファイル名の連番をリセット
    idx = 1
    
    # 距離行列を計算
    distance_matrix = np.zeros((len(data), len(data)))
    for i in tqdm(range(len(data)), desc="Calculating Distance Matrix"):
        for j in range(len(data)):
            distance_matrix[i, j], _ = fastdtw(data[i], data[j])
    
    # KMeansクラスタリングを実行:
    num_clusters = determine_num_clusters(distance_matrix)  # エルボー法を使ってクラスタ数を決定
    kmeans = KMeans(n_clusters=num_clusters)
    kmeans.fit(distance_matrix)
    clusters = kmeans.labels_
    
    # クラスタごとにデータを保存
    for cluster_label in range(num_clusters):
        cluster_indices = np.where(clusters == cluster_label)[0]
        
        # クラスタごとのディレクトリを作成
        cluster_output_directory = os.path.join(output_directory, f"cluster_{cluster_label + 1}")
        os.makedirs(cluster_output_directory, exist_ok=True)
        
        # クラスタごとのデータを保存
        for cluster_idx in cluster_indices:
            # クラスタデータをそのままコピーして保存
            src_file = os.path.join(data_directory, f"data_{cluster_idx + 1}.csv")  # ファイル名の連番が1始まりなら +1 不要
            dest_file = os.path.join(cluster_output_directory, os.path.basename(src_file))
            shutil.copy(src_file, dest_file)
            
            # クラスタのラベルをファイル名に追加
            new_file_name = f"data_{idx}_cluster{cluster_label + 1}.csv"
            new_file_path = os.path.join(cluster_output_directory, new_file_name)
            os.rename(dest_file, new_file_path)
            
            idx += 1

# メイン関数
if __name__ == "__main__":
    data_directory = args.filename +  '/Oba_卒業研究A/2023年度歩容測定実験/venus3d_data/normalization_data/クラスタリング用'
    output_directory = args.filename + '/Oba_卒業研究A/2023年度歩容測定実験/venus3d_data/normalization_data'
    
    cluster_and_output(data_directory, output_directory)
