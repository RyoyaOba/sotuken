
# from PIL import Image
# import os
# # 対象のフォルダ
# folder_path = '/app/Oba_卒業研究A/クラスタリング結果の可視化/'

# image_files = [
#     'cluster_体幹角度_矢状面.png.png', 
#     'cluster_体幹角度_前額面.png.png', 
#     'cluster_右股関節角度_矢状面.png.png', 
#     'cluster_右股関節角度_前額面.png.png',
#     'cluster_左股関節角度_矢状面.png.png', 
#     'cluster_左股関節角度_前額面.png.png',
#     'cluster_右膝関節角度_矢状面.png.png', 
#     'cluster_右膝関節角度_前額面.png.png',
#     'cluster_左膝関節角度_矢状面.png.png', 
#     'cluster_左膝関節角度_前額面.png.png', 
#     'cluster_.png', 
#     'cluster_右足関節角度_前額面.png.png',
#     'cluster_左足関節角度_矢状面.png.png', 
#     'cluster_左足関節角度_前額面.png.png', 
#     'cluster_R_FP2-FxNkg.png.png', 'cluster_R_FP2-FyNkg.png.png',

#     'cluster_R_FP2-FzNkg.png.png', 'cluster_L_FP1-FxNkg.png.png',
#     'cluster_L_FP1-FyNkg.png.png', 'cluster_L_FP1-FzNkg.png.png'
# ]

# # 1つの画像にまとめる
# images = [Image.open(os.path.join(folder_path, filename)).resize((1800, 500), Image.BICUBIC) for filename in image_files]

# # 1つの画像に連結する
# max_width = max(img.width for img in images)
# total_height = sum(img.height for img in images)

# # 新しい画像を作成
# new_image = Image.new('RGB', (max_width * 4, total_height))

# # 画像を配置
# x_offset = 0
# y_offset = 0
# for idx, img in enumerate(images, start=1):
#     new_image.paste(img, (x_offset, y_offset))
#     y_offset += img.height

#     # 4の倍数の場合、次の行に移る
#     if idx % 5 == 0:
#         x_offset += img.width
#         y_offset = 0


# # 黒い余白をトリミングして新しい画像を作成
# bbox = new_image.getbbox()  # 画像を囲む境界ボックスを取得
# new_image = new_image.crop(bbox)

# # 新しい画像を表示するか保存する
# new_image.show()  # 画像を表示する場合
# new_image.save('clustering_combined_image.png')#,quality = 95)  # 画像を保存する場


#----------
#先にこっち
#--------
from PIL import Image
import os

# 対象のフォルダ
folder_path = '/app/Oba_卒業研究A/クラスタリング結果の可視化/'

# 一個目のアンダーバー以降の名前ごとに画像をまとめる辞書を作成
image_dict = {}
for filename in os.listdir(folder_path):
    if filename.endswith('.png'):
        key = '_'.join(filename.split('_')[1:])  # 一個目のアンダーバー以降の名前をキーとして取得
        if key not in image_dict:
            image_dict[key] = []
        image_dict[key].append(filename)

# Print the keys to identify the issue
print("Keys:", image_dict.keys())

# キーでソート
#sorted_image_dict = dict(sorted(image_dict.items(), key=lambda x: (not x[0].startswith(('L_', 'R_')), int(x[0].split('_')[0]))))
sorted_image_dict = dict(sorted(image_dict.items(), key=lambda x: (not x[0].startswith(('L_', 'R_')), int(x[0].split('_')[0]) if x[0].split('_')[0].isdigit() else float('inf'))))

# Print the sorted keys
print("Sorted Keys:", sorted_image_dict.keys())

# それぞれのキーごとに画像を横に連結し保存
for key, filenames in sorted_image_dict.items():
    images = [Image.open(os.path.join(folder_path, fname)) for fname in filenames]
    widths, heights = zip(*(i.size for i in images))

    total_width = sum(widths)
    max_height = max(heights)

    new_image = Image.new('RGB', (total_width, max_height))

    x_offset = 0
    for img in images:
        new_image.paste(img, (x_offset, 0))
        x_offset += img.size[0]

    save_name = f"cluster_{key}.png"
    new_image.save(os.path.join(folder_path, f"{save_name}"))
