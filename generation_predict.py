
import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import japanize_matplotlib
import tensorflow as tf
from PIL import Image
from utils.make_GanDataset import GanDatasets

#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

ds = GanDatasets('/app/Walking_Clustering/sotukenB_clustering')
ds10_samples, ds01_samples, _, _ = ds.load_ds(sample_size=10560)
ds.split_data(ds10_samples, ds01_samples)
ds10_val = ds.dataset_10_val  # dataset_10_trainを取得
ds01_val = ds.dataset_01_val

#-------------
#  load model"./generator_f/model_cycle_ver0_alldata
#-------------
#generator = tf.keras.models.load_model("./generator/ver1.h5")
generator= tf.keras.models.load_model('/app/Oba_卒業研究A/generator_f/model_c21_epoch20')
#coatnet = tf.keras.models.load_model("./coatnet/model")
tf.keras.utils.plot_model(generator, to_file ='cyclegan_generator_model4.png')
#tf.keras.utils.plot_model(coatnet, to_file = "coatnet_model.png")

output_path = "/app/Oba_卒業研究A/aa/"

def save_plot_with_directory_check(path):
    directory = os.path.dirname(path)
    if not os.path.exists(directory):
        os.makedirs(directory)
    plt.savefig(path)

for i in range(len(ds01_val)):
    # 生成歩容データのcsv化
    gen_data = generator(ds01_val[i:i+1])
    gen_tensor = gen_data[0].numpy()
    reshaped_tensor = gen_tensor.reshape(399, 20)
    #gen_outputs = f"{output_path}gen_data_{i+1}.csv"
    #np.savetxt(gen_outputs, reshaped_tensor, delimiter=",")
    
    # 入力歩容データのcsv化
    no_ideal_data = ds01_val[i:i+1]
    reshaped_non_ideal_tensor = no_ideal_data.reshape(399, 20)
    #non_ideal_outputs = f"{output_path}no_ideal_data_{i+1}.csv"
    #np.savetxt(non_ideal_outputs, reshaped_non_ideal_tensor, delimiter=",")
    #cd Oba_卒業研究A
    # gen_outputsとnon_ideal_outputsをつなげて1つのcsvにする
    combined_output = f"{output_path}combined_data_{i+1}.csv"
    combined_data = np.concatenate((reshaped_tensor, reshaped_non_ideal_tensor), axis=1)
    np.savetxt(combined_output, combined_data, delimiter=",")


selected_file_number = np.random.randint(1, len(ds01_val))

selected_combined_data = pd.read_csv(f"{output_path}combined_data_{selected_file_number}.csv", header=None)

# タイトルのリスト
column_labels_japanize = ['体幹角度_矢状面','体幹角度_前額面',\
                '右股関節角度_矢状面','右股関節角度_前額面', \
                '左股関節角度_矢状面', '左股関節角度_前額面',\
                '右膝関節角度_矢状面', '右膝関節角度_前額面', \
                '左膝関節角度_矢状面','左膝関節角度_前額面',\
                '右足関節角度_矢状面','右足関節角度_前額面',\
                '左足関節角度_矢状面', '左足関節角度_前額面',\
                'L_FP1-Fx(N/kg)', 'L_FP1-Fy(N/kg)',' L_FP1-Fz(N/kg)',\
                'R_FP2-Fx(N/kg)', 'R_FP2-Fy(N/kg)','R_FP2-Fz(N/kg)']

column_labels = [
    "Normalized change\ntrunk angle yz [-]", \
    "Normalized change\ntrunk angle zx [-]", \
    "Normalized change\nR hip angle yz [-]",\
    "Normalized change\nR hip angle zx [-]",\
    "Normalized change\nL hip angle yz [-]",\
    "Normalized change\nL hip angle zx [-]", \
    "Normalized change\nR knee angle yz [-]", \
    "Normalized change\nR knee angle zx [-]", \
    "Normalized change\nL knee angle yz [-]", \
    "Normalized change\nL knee angle zx [-]",\
    "Normalized change\nR ankle angle yz [-]",\
    "Normalized change\nR ankle angle zx [-]",\
    "Normalized change\nL ankle angle yz [-]",\
    "Normalized change\nL ankle angle zx [-]", \
    "Normalized\nL ground reaction force x [-]",\
    "Normalized\nL ground reaction force y [-]", \
    "Normalized\nL ground reaction force z [-]", \
    "Normalized\nR ground reaction force x [-]", \
    "Normalized\nR ground reaction force y [-]", \
    "Normalized\nR ground reaction force z [-]",\
    "Normalized change of right\nthumb-to-grond distance [-]",
    "Normlaized change of left\nthumb-to-ground distance [-]"
]

column_labels_2 = ["change in\ntrunk angle YZ [-]",
                "change in\ntrunk angle XZ [-]",
                "change in\nhip angle YZ\nof the target leg [-]", "change in\nhip angle XZ\nof the target leg [-]", 
                "change in\nhip angle YZ\nof the non-target leg [-]", "change in\nhip angle XZ\nof the non-target leg [-]", "change in\nknee angle YZ\nof the target leg [-]", "change in\nknee angle XZ\nof the target leg [-]", 
                "change in\nknee angle YZ\nof the non-target leg [-]", "change in\nknee angle XZ\nof the non-target leg [-]", "change in\nankle angle YZ\nof the target leg [-]", "change in\nankle angle XZ\nof the target leg [-]", 
                "change in\nankle angle YZ\nof the non-target leg [-]", "change in\nankle angle XZ\nof the non-target leg [-]", 
                "\nground reaction force X\nof the non-target leg [-]", "\nground reaction force Y\nof the non-target leg [-]", "\nground reaction force Z\nof the non-target leg [-]",
                "\nground reaction force X\nof the target leg [-]", "\nground reaction force Y\nof the target leg [-]", 
                "\nground reaction force Z\nof the target leg [-]", "Normalized change of right\nthumb-to-grond distance [-]",
                "Normlaized change of left\nthumb-to-ground distance [-]"]
def clean_filename(name):
    return re.sub(r'[^\w\-_\.]', '', name)
#---------------------------------
# データ1つだけを可視化（横線時間軸）
#---------------------------------
for i in range(20):
    plt.figure(figsize=(12, 6))

    
    plt.plot(
        selected_combined_data.index, selected_combined_data[i + 20],
        label='non_ideal_gait',
        color='blue'
    )

    plt.plot(
        selected_combined_data.index, selected_combined_data[i],
        label='ideal_gait',
        #linestyle='--',
        color='red'
    )    
    plt.legend(fontsize=18, frameon=False)
    plt.xlabel('Walking cycle', fontsize=24)
    plt.ylabel(f'{column_labels[i]}', fontsize=24)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.grid(True, color='black', linestyle='--', linewidth=0.5)
    plt.gca().spines['top'].set_color('black')
    plt.gca().spines['bottom'].set_color('black')
    plt.gca().spines['left'].set_color('black')
    plt.gca().spines['right'].set_color('black')
    plt.tight_layout()
    plt.xlim(0, 400)
    plt.ylim(0, 1)
    #plt.title(f'{column_labels[i]}')
    cleaned_label = clean_filename(column_labels[i])
 
    plt.savefig(f"{output_path}Time_series_{selected_file_number}_lineplot_{cleaned_label}_{i+1}.png", dpi=300)
    plt.close()


#-----------------------------
#データ1だけを可視化（歩行周期）
#-----------------------------
for i in range(20):
    plt.figure(figsize=(12, 6))
    total_steps = len(selected_combined_data) - 1
    percentage_steps = [i * 100 / total_steps for i in range(len(selected_combined_data))]
    
    plt.plot(
        percentage_steps, selected_combined_data[i + 20],
        label='non_ideal_gait',
        color='blue',
        linewidth = 3
    )
    plt.plot(
        percentage_steps, selected_combined_data[i],
        label='ideal_gait',
        #linestyle='--',
        color='red',
        linewidth = 3
    )
    plt.legend(fontsize=18, frameon=False)
    plt.xlabel('Walking Cycle [%]', fontsize=24)
    plt.ylabel(f'{column_labels[i]}', fontsize=24)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.grid(True, color='black', linestyle='--', linewidth=0.5)
    plt.gca().spines['top'].set_color('black')
    plt.gca().spines['bottom'].set_color('black')
    plt.gca().spines['left'].set_color('black')
    plt.gca().spines['right'].set_color('black')
    plt.tight_layout()
    plt.xlim(0, 100)
    plt.ylim(0, 1)
    plt.tight_layout()
    #plt.title(f'{column_labels[i]}')
    cleaned_label = clean_filename(column_labels[i])
 
    plt.savefig(f"{output_path}walking_cycle_{selected_file_number}_lineplot_{cleaned_label}_{i+1}.png")
    plt.clf()


# #------------------------
# # 全データの平均値を可視化
# #------------------------
file_paths = [os.path.join(output_path, file) for file in os.listdir(output_path) if file.endswith('.csv')]

all_column_data = {}  

for file_path in file_paths:
    df = pd.read_csv(file_path, encoding='shift-jis')
    for col in df.columns:
        if col not in all_column_data:
            all_column_data[col] = []
        all_column_data[col].append(df[col])

mean_column_data = {}
for col, data_list in all_column_data.items():
    mean_column_data[col] = np.mean(data_list, axis=0)

mean_df = pd.DataFrame(mean_column_data)
mean_df.to_csv(os.path.join(output_path, 'mean_all_columns.csv'), index=False)
# Load mean data from the CSV file

mean_df_glaph = pd.read_csv(os.path.join(output_path, 'mean_all_columns.csv'))

for i in range(20):
    plt.figure(figsize=(12, 6))
    total_steps = len(selected_combined_data) - 1
    percentage_steps = [i * 100 / total_steps for i in range(len(selected_combined_data))]

    #-----------------------------------
    #つまずきやすい歩容（入力データ）
    #-----------------------------------
    plt.plot(
        percentage_steps, selected_combined_data[i + 20],
        label=f'{selected_file_number}_non_ideal_gait',
        color='blue',
        zorder = 3,
        linewidth = 3
    )

    #-----------------------------------
    # 生成つまずきにくい歩容（任意のデータ）
    #-----------------------------------
    plt.plot(
        percentage_steps, selected_combined_data[i],
        label=f'{selected_file_number}_ideal_gait',
        #linestyle='--',
        color='red',
        zorder = 3,
        linewidth = 3
    )    
    #-----------------------------------
    #生成つまずきにくい歩容（平均のデータ）
    #-----------------------------------
    total_steps = len(mean_df) - 1
    percentage_steps = [i * 100 / total_steps for i in range(len(mean_df))]


    #つまずきやすい歩容（平均データ）
    #-----------------------------------
    plt.plot(
        percentage_steps, mean_df_glaph.iloc[:, i + 20],
        label='mean_non_ideal_gait',
        linestyle='--',
        #linewidth = 3,
        color='blue',
        zorder = 3,
        linewidth = 1.5
    )
    plt.plot(
        percentage_steps, mean_df_glaph.iloc[:, i],
        label='mean_ideal_gait',
        linestyle='--',
        color='red',
        zorder = 3,
        linewidth = 1.5
    )
    #-----------------------------------
    
    plt.legend(fontsize=18, frameon=False)
    plt.xlabel('Walking Cycle [%]', fontsize=26)
    plt.ylabel(f'Normalized {column_labels_2[i]}', fontsize=28)
    plt.xticks(np.linspace(0,100,6))     
    plt.yticks(np.linspace(0,1,6)) 
    plt.grid(True, color='black', linestyle='--', linewidth=0.5)

    plt.tick_params(labelsize=22)
    plt.gca().spines['top'].set_color('black')
    plt.gca().spines['bottom'].set_color('black')
    plt.gca().spines['left'].set_color('black')
    plt.gca().spines['right'].set_color('black')
    plt.tight_layout()
    plt.xlim(0, 100)
    plt.ylim(0.0, 1)
    plt.tight_layout()
    #plt.title(f'{column_labels[i]}')
    cleaned_label = clean_filename(column_labels[i])
    plt.savefig(f"all_{i+1}.png")
    plt.clf()
    #save_plot_with_directory_check(f"{output_path}Mean_{cleaned_label}_{i+1}.png")

#----------
# クラスタごとの平均
# ----------
#つまずきにくい歩容

# cluster0_path= "/app/Oba_卒業研究A/2023年度歩容測定実験/venus3d_data/normalization_data/thumb_2/"
# cluster0 = [os.path.join(cluster0_path, file) for file in os.listdir(cluster0_path) if file.endswith('.csv')]

# all_cluster0_data = {}  

# for file_path in cluster0_path:
#     df = pd.read_csv(file_path, encoding='latin-1')
#     for col in df.columns:
#         if col not in all_cluster0_data:
#             all_cluster0_data[col] = []
#         all_cluster0_data[col].append(df[col])

# mean_cluster0_data = {}
# for col, data_list in all_cluster0_data.items():
#     mean_cluster0_data[col] = np.mean(data_list, axis=0)

# mean_cluster0_df = pd.DataFrame(mean_cluster0_data)
# mean_cluster0_df.to_csv(os.path.join(output_path, 'cluster0_mean_all_columns.csv'), index=False)
# # Load mean data from the CSV file

# mean_cluster0_df_glaph = pd.read_csv(os.path.join(cluster0_path, 'cluster0_mean_all_columns.csv'))


# #つまずきやすい歩容
# cluster1_path= "/app/Oba_卒業研究A/2023年度歩容測定実験/venus3d_data/normalization_data/thumb_1/"
# cluster1 = [os.path.join(cluster1_path, file) for file in os.listdir(cluster1_path) if file.endswith('.csv')]

# all_cluster1_data = {}  

# for file_path in cluster1_path:
#     df = pd.read_csv(file_path, encoding='latin-1')
#     for col in df.columns:
#         if col not in all_cluster1_data:
#             all_cluster1_data[col] = []
#         all_cluster1_data[col].append(df[col])

# mean_cluster1_data = {}
# for col, data_list in all_cluster1_data.items():
#     mean_cluster1_data[col] = np.mean(data_list, axis=0)

# mean_cluster1_df = pd.DataFrame(mean_cluster1_data)
# mean_cluster1_df.to_csv(os.path.join(output_path, 'cluster1_mean_all_columns.csv'), index=False)
# # Load mean data from the CSV file

# mean_cluster1_df_glaph = pd.read_csv(os.path.join(cluster1_path, 'cluster1_mean_all_columns.csv'))

def process_cluster_data(cluster_path, out):
    all_cluster_data = {}
    file_paths = [os.path.join(cluster_path, file) for file in os.listdir(cluster_path) if file.endswith('.csv')]

    for file_path in file_paths:
        df = pd.read_csv(file_path, encoding='latin-1')
        for col in df.columns:
            if col not in all_cluster_data:
                all_cluster_data[col] = []
            all_cluster_data[col].append(df[col])

    mean_cluster_data = {}
    for col, data_list in all_cluster_data.items():
        mean_cluster_data[col] = np.mean(data_list, axis=0)

    mean_cluster_df = pd.DataFrame(mean_cluster_data)
    mean_cluster_df.to_csv(out, index=False)
    return mean_cluster_df

# クラスタ0の処理
# クラスタ0の処理
cluster0_path = "/app/Walking_Clustering/sotukenB_clustering/sotukenB_thumb_2"
#output_path = "/app/Oba_卒業研究A/cc/"  # あなたの出力先パス
output_path = "/app/Oba_卒業研究A/aa/"  # 出力先のパス
mean_cluster0_output = os.path.join(output_path, 'cluster0_mean_all_columns.csv')
mean_cluster0_df = process_cluster_data(cluster0_path, mean_cluster0_output)

# 21列目と22列目を含まないDataFrameを作成する
mean_cluster0_df_filtered = mean_cluster0_df.drop(columns=[mean_cluster0_df.columns[20], mean_cluster0_df.columns[21]])

# CSVファイルに保存する際、1行目を含めないで保存する
mean_cluster0_df_filtered.to_csv(mean_cluster0_output, index=False, header=False)

#mean_cluster0_df.to_csv(mean_cluster0_output, header=False)


cluster1_path = "/app/Walking_Clustering/sotukenB_clustering/sotukenB_thumb_1"

mean_cluster1_output = os.path.join(output_path, 'cluster1_mean_all_columns.csv')
mean_cluster1_df = process_cluster_data(cluster1_path, mean_cluster1_output)
mean_cluster1_df.to_csv(mean_cluster1_output, index = False,header=None)

# 21列目と22列目を含まないDataFrameを作成する
mean_cluster1_df_filtered = mean_cluster1_df.drop(columns=[mean_cluster1_df.columns[20], mean_cluster1_df.columns[21]])

# CSVファイルに保存する際、1行目を含めないで保存する
mean_cluster1_df_filtered.to_csv(mean_cluster1_output,index=False, header=None)

mean_cluster0_df_glaph = pd.read_csv(os.path.join(output_path, 'cluster0_mean_all_columns.csv'))
mean_cluster1_df_glaph = pd.read_csv(os.path.join(output_path, 'cluster1_mean_all_columns.csv'))
print(mean_cluster0_df_glaph.head())
print(mean_cluster1_df_glaph.head())
print(len(mean_cluster0_df_glaph.columns))
print(len(mean_cluster1_df_glaph.columns))
print(mean_cluster0_df_glaph.shape)
print(mean_cluster1_df_glaph.shape)
print(i, i + 20)

# column_labels = [
#     "Normalized change\ntrunk angle yz [-]", \
#     "Normalized change\ntrunk angle zx [-]", \
#     "Normalized change\nR hip angle yz [-]",\
#     "Normalized change\nR hip angle zx [-]",\
#     "Normalized change\nL hip angle yz [-]",\
#     "Normalized change\nL hip angle zx [-]", \
#     "Normalized change\nR knee angle yz [-]", \
#     "Normalized change\nR knee angle zx [-]", \
#     "Normalized change\nL knee angle yz [-]", \
#     "Normalized change\nL knee angle zx [-]",\
#     "Normalized change\nR ankle angle yz [-]",\
#     "Normalized change\nR ankle angle zx [-]",\
#     "Normalized change\nL ankle angle yz [-]",\
#     "Normalized change\nL ankle angle zx [-]", \
#     "Normalized\nL ground reaction force x [-]",\
#     "Normalized\nL ground reaction force y [-]", \
#     "Normalized\nL ground reaction force z [-]", \
#     "Normalized\nR ground reaction force x [-]", \
#     "Normalized\nR ground reaction force y [-]", \
#     "Normalized\nR ground reaction force z [-]",\
#     "Normalized change of right\nthumb-to-grond distance [-]",
#     "Normlaized change of left\nthumb-to-ground distance [-]"
# ]


column_labels_3 = ["change in\ntrunk angle YZ [-]",
                   "change in\ntrunk angle XZ [-]",
                   "change in\nhip angle YZ\nof the target leg [-]", "change in\nhip angle XZ\nof the target leg [-]", 
                "change in\nhip angle YZ\nof the non-target leg [-]", "change in\nhip angle XZ\nof the non-target leg [-]", "change in\nknee angle YZ\nof the target leg [-]", "change in\nknee angle XZ\nof the target leg [-]", 
                "change in\nknee angle YZ\nof the non-target leg [-]", "change in\nknee angle XZ\nof the non-target leg [-]", "change in\nankle angle YZ\nof the target leg [-]", "change in\nankle angle XZ\nof the target leg [-]", 
                "change in\nankle angle YZ\nof the non-target leg [-]", "change in\nankle angle XZ\nof the non-target leg [-]", 
                "\nground reaction force X\nof the non-target leg [-]", "\nground reaction force Y\nof the non-target leg [-]", "\nground reaction force Z\nof the non-target leg [-]",
                "\nground reaction force X\nof the target leg [-]", "\nground reaction force Y\nof the target leg [-]", 
                "\nground reaction force Z\nof the target leg [-]", "Normalized change of right\nthumb-to-grond distance [-]",
                "Normlaized change of left\nthumb-to-ground distance [-]"]
# Visualize mean data for all columns
for i in range(20):
    plt.figure(figsize=(12, 6))
    total_steps = len(mean_cluster0_df_glaph) - 1
    percentage_steps = [i * 100 / total_steps for i in range(len(mean_cluster0_df_glaph))]

    # #生成と元データの平均値
    # total_steps = len(selected_combined_data) - 1
    # percentage_steps = [i * 100 / total_steps for i in range(len(selected_combined_data))]
    # #つまずきにくい歩容クラス
    # cluster0_total_steps =  len(mean_cluster0_df)
    # cluster0_percentage_steps=[i * 100 / cluster0_total_steps for i in range(len(mean_cluster0_df))]
    # #つまずきやすい歩容クラス
    # cluster1_total_steps =  len(mean_cluster1_df)
    # cluster1_percentage_steps=[i * 100 / cluster1_total_steps for i in range(len(mean_cluster1_df))]

 
    
    plt.plot(
        percentage_steps, mean_cluster1_df_glaph.iloc[:,i],
        label='つまずきやすい歩容',
        zorder =3,
        linewidth = 3,
        linestyle='--',
        color=(0, 0, 0.6),  # RGB値を少し暗めに調整
    )

    plt.plot(
        percentage_steps, mean_cluster0_df_glaph.iloc[:,i],
        label='つまずきにくい歩容',
        linestyle='--',
        zorder =3,
        linewidth = 3,
        color='red',  # 赤い色を暗めに調整
    )
    # plt.plot(
    #     percentage_steps, mean_df_glaph.iloc[:, i],
    #     label='mean_ideal_gait',
    #     linestyle='--',
    #     color='red'
    # )

    plt.plot(
        percentage_steps, selected_combined_data[i + 20],
        label=f'{selected_file_number}_non_ideal_gait',
        color='blue',
        zorder = 3,
        linewidth=3
    )
    plt.plot(
        percentage_steps, selected_combined_data[i],
        label=f'{selected_file_number}_ideal_gait',
        color= 'green',
        zorder = 3,
        linewidth=3
        #alpha=0.9

    )
    plt.legend(fontsize=18, frameon=False)
    plt.xlabel('Walking Cycle [%]', fontsize=26)
    plt.ylabel(f'Normalized {column_labels_3[i]}', fontsize=28)
    plt.xticks(np.linspace(0,100,6))     
    plt.yticks(np.linspace(0,1,6)) 
    plt.grid(True, color='black', linestyle='--', linewidth=0.5)
    
    #plt.grid(b=True, which='major', axis='both', zorder = 1)
    plt.tick_params(labelsize=22)
    plt.gca().spines['top'].set_color('black')
    plt.gca().spines['bottom'].set_color('black')
    plt.gca().spines['left'].set_color('black')
    plt.gca().spines['right'].set_color('black')
    plt.tight_layout()
    plt.xlim(0, 100)
    plt.ylim(0.0, 1)
    plt.tight_layout()
    cleaned_label = clean_filename(column_labels[i])

    plt.savefig(f"Mean_{i+1}.png")
    #lt.savefig(f"{output_path}Mean_{cleaned_label}_{i+1}.png")
    plt.clf()
    
#-------------------------
# 多変量生成歩容を1つに表示
#------------------------
im_files = [
    'Mean_1.png', 'Mean_2.png', 'Mean_3.png', 'Mean_4.png',
    'Mean_5.png', 'Mean_6.png', 'Mean_7.png', 'Mean_8.png',
    'Mean_9.png', 'Mean_10.png', 'Mean_11.png', 'Mean_12.png',
    'Mean_13.png', 'Mean_14.png', 'Mean_18.png', 'Mean_19.png',
    'Mean_20.png', 'Mean_15.png', 'Mean_16.png', 'Mean_17.png'
]



# image_files = [
#     'Mean_1.png', 'Mean_2.png', 'Mean_3.png', 'Mean_4.png',
#     'Mean_5.png', 'Mean_6.png', 'Mean_7.png', 'Mean_8.png',
#     'Mean_9.png', 'Mean_10.png', 'Mean_11.png', 'Mean_12.png',
#     'Mean_13.png', 'Mean_14.png', 'Mean_18.png', 'Mean_19.png',
#     'Mean_20.png', 'Mean_15.png', 'Mean_16.png', 'Mean_17.png'
# ]

images = [Image.open(filename).resize((1300, 700), Image.BICUBIC) for filename in im_files]

#連結
max_width = max(img.width for img in images)
total_height = sum(img.height for img in images)

#作成
new_image = Image.new('RGB', (max_width * 5, total_height))


x_offset = 0
y_offset = 0
for idx, img in enumerate(images, start=1):
    new_image.paste(img, (x_offset, y_offset))
    y_offset += img.height

    # 4の倍数の場合、次の行に移る
    if idx % 4 == 0:
        x_offset += img.width
        y_offset = 0

bbox = new_image.getbbox() 
new_image = new_image.crop(bbox)

# new_image.show() 
# new_image.save('2024transe1.png',quality = 100)  

last_out = '/app/Oba_卒業研究A/for_emotional_engineering'

last_out_path = os.path.join(last_out, f'data_{selected_file_number}.png')
new_image.save(last_out_path, quality = 100)
