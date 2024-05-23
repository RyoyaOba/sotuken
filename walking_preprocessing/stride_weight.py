
import os
import csv

def extract_floor_reaction_force(input_folder, output_folder):
    # フォースプレート1およびフォースプレート2のX軸、Y軸、Z軸方向の力のカラム番号を指定
    fp1_fx_col = 3
    fp1_fy_col = 4
    fp1_fz_col = 5
    fp2_fx_col = 11
    fp2_fy_col = 12
    fp2_fz_col = 13

    # 指定したフォルダ内の全てのCSVファイルを処理
    for filename in os.listdir(input_folder):
        if filename.endswith("FP.csv"):
            file_path = os.path.join(input_folder, filename)
            output_filename = filename.replace("sort", "force", 1)  # sortをforceに置換
            output_file_path = os.path.join(output_folder, output_filename)  # ここを修正

            # 新しいCSVデータを格納するリスト
            new_data = []

            with open(file_path, 'r') as csvfile:
                reader = csv.reader(csvfile)
                # 最初の75行をスキップ
                for _ in range(75):
                    next(reader)
                headers = next(reader)  # ヘッダー行をスキップ
                for row in reader:
                    # フォースプレート1およびフォースプレート2のX軸、Y軸、Z軸方向の力のデータを抽出
                    fp1_fx = float(row[fp1_fx_col])
                    fp1_fy = float(row[fp1_fy_col])
                    fp1_fz = float(row[fp1_fz_col])
                    fp2_fx = float(row[fp2_fx_col])
                    fp2_fy = float(row[fp2_fy_col])
                    fp2_fz = float(row[fp2_fz_col])

                    # 新しいCSVデータに追加
                    new_row = [fp1_fx, fp1_fy, fp1_fz, fp2_fx, fp2_fy, fp2_fz]
                    new_data.append(new_row)

            # 新しいCSVファイルにデータを書き込み
            with open(output_file_path, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                # ヘッダー行を書き込む
                writer.writerow(['FP1-Fx(N)', 'FP1-Fy(N)', 'FP1-Fz(N)', 'FP2-Fx(N)', 'FP2-Fy(N)', 'FP2-Fz(N)'])
                # データを書き込む
                writer.writerows(new_data)

            print("新しいCSVファイルが作成されました : ", output_file_path)

def normalize_floor_reaction_force(force_input_folder, normalization_output_folder):
    # フォースプレート1およびフォースプレート2のX軸、Y軸、Z軸方向の力のカラム番号を指定
    fp1_fx_col = 0
    fp1_fy_col = 1
    fp1_fz_col = 2
    fp2_fx_col = 3
    fp2_fy_col = 4
    fp2_fz_col = 5

    # 個人識別アルファベットと体重の対応表
    weight_dict = {
        'DLVI': 48.9,
        'SESV': 61.9,
        'IFCN': 65.3,
        'OSRZ': 55.5,
        'RVGK': 66.4,
        'MXNI': 59.1,
        'ZUIO': 66.3,
        'NVIX': 71.9,
        'NJUA': 84.8,
        'UTWG': 54.4,
        'BHVL': 54.3,
        # 8/23以降追加データ
        'SZZY':63.7,
        'CUJT':52.8,
        'BUAL':53.9,
        'HAWC':58.9,
        'KDDO':68.0,
        'HKGP':46.0,
        'DIPJ':57.7,
        'JAQJ':42.6,
        'WWOT':54.3,
        'DOMG':49.0,
        'YXRA':57.8,
        'FTND':50.5,
        'CXIP':56.7,
        'XDIO':87.0,
        'KILB':45.2,
        'BAEO':66.2,
        'AZDJ':49.8,
        'KSKV':79.8,
        'GUIK':89.3,
        'IOOP':58.5,
    }

    # 体重を1にするためのスケールファクターを計算
    weight_sum = sum(weight_dict.values())
    scale_factor = len(weight_dict) / weight_sum

    # 指定したフォルダ内の全てのCSVファイルを処理
    for filename in os.listdir(force_input_folder):
        if filename.startswith("force") and filename.endswith(".csv"):
            file_path = os.path.join(force_input_folder, filename)

            # 個人識別アルファベットを取得
            identifier = filename[6:10]

            # 対応する体重を取得
            weight = weight_dict.get(identifier, 0.0)
            print("Identifier:", identifier, "Weight:", weight)  # デバッグ用の出力

            # シリアル番号の記載にミスがあれば、ここで気が付ける
            if weight == 0.0:
                print("Warning: Weight for", identifier, "is 0.0. Skipping normalization.")
                continue

            # 新しいCSVデータを格納するリスト
            new_data = []

            with open(file_path, 'r') as csvfile:
                reader = csv.reader(csvfile)
                # ヘッダー行をスキップ
                headers = next(reader)
                for row in reader:
                    try:
                        # フォースプレート1およびフォースプレート2のX軸、Y軸、Z軸方向の力のデータを抽出
                        fp1_fx = float(row[fp1_fx_col])
                        fp1_fy = float(row[fp1_fy_col])
                        fp1_fz = float(row[fp1_fz_col])
                        fp2_fx = float(row[fp2_fx_col])
                        fp2_fy = float(row[fp2_fy_col])
                        fp2_fz = float(row[fp2_fz_col])

                        # 重力加速度 (m/s^2)
                        gravity_acceleration = 9.81
                        # 体重を1にするためにスケールファクターを乗算（重力加速度かけるよ）
                        normalized_fp1_fx = fp1_fx * scale_factor * gravity_acceleration
                        normalized_fp1_fy = fp1_fy * scale_factor * gravity_acceleration
                        normalized_fp1_fz = fp1_fz * scale_factor * gravity_acceleration
                        normalized_fp2_fx = fp2_fx * scale_factor * gravity_acceleration
                        normalized_fp2_fy = fp2_fy * scale_factor * gravity_acceleration
                        normalized_fp2_fz = fp2_fz * scale_factor * gravity_acceleration

                        # 新しいCSVデータに追加
                        new_row = [normalized_fp1_fx, normalized_fp1_fy, normalized_fp1_fz,
                                   normalized_fp2_fx, normalized_fp2_fy, normalized_fp2_fz]
                        new_data.append(new_row)

                    except IndexError:
                        # データが欠落している行をスキップ
                        print("データが欠落している行をスキップしました：", file_path, row)

            # 新しいCSVファイルにデータを書き込み
            output_filename = "normalization_" + filename
            output_file_path = os.path.join(normalization_output_folder, output_filename)

            with open(output_file_path, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                # ヘッダー行を書き込む
                writer.writerow(['FP1-Fx(N/kg)', 'FP1-Fy(N/kg)', 'FP1-Fz(N/kg)', 'FP2-Fx(N/kg)', 'FP2-Fy(N/kg)', 'FP2-Fz(N/kg)'])
                # データを書き込む
                writer.writerows(new_data)

            print("新しいnormalization_floor_reaction_force_CSVファイル：", output_file_path)

# ここから実行
if __name__ == "__main__":
    input_folder = "C:/Users/human/Oba_卒業研究A/2023年度歩容測定実験/venus3d_data/sorted_data"
    output_folder = "C:/Users/human/Oba_卒業研究A/2023年度歩容測定実験/venus3d_data/sorted_data/floor_reaction_force"
    extract_floor_reaction_force(input_folder, output_folder)

    normalization_output_folder = "C:/Users/human/Oba_卒業研究A/2023年度歩容測定実験/venus3d_data/normalization_floor_reaction_force"
    normalize_floor_reaction_force(output_folder, normalization_output_folder)
