
import os
from PIL import Image

# 画像ファイル名のリスト
image_files = [
    'Mean_1.png', 'Mean_2.png', 'Mean_3.png', 'Mean_4.png',
    'Mean_5.png', 'Mean_6.png', 'Mean_7.png', 'Mean_8.png',
    'Mean_9.png', 'Mean_10.png', 'Mean_11.png', 'Mean_12.png',
    'Mean_13.png', 'Mean_14.png', 'Mean_18.png', 'Mean_19.png',
    'Mean_20.png', 'Mean_15.png', 'Mean_16.png', 'Mean_17.png'
]


# # 画像ファイル名のリスト
# image_files = [
#     'Mean_1.png', 'Mean_2.png', 'Mean_3.png', 'Mean_4.png',
#     'Mean_5.png', 'Mean_6.png', 'Mean_7.png', 'Mean_8.png',
#     'Mean_9.png', 'Mean_10.png', 'Mean_11.png', 'Mean_12.png',
#     'Mean_13.png', 'Mean_14.png', 'Mean_18.png', 'Mean_19.png',
#     'Mean_20.png', 'Mean_15.png', 'Mean_16.png', 'Mean_17.png'
# ]

# 1つの画像として出力
images = [Image.open(filename).resize((1300, 700), Image.BICUBIC) for filename in image_files]

# 1つの画像に連結する
max_width = max(img.width for img in images)
total_height = sum(img.height for img in images)

# 新しい画像を作成
new_image = Image.new('RGB', (max_width * 5, total_height))

# 画像を配置
x_offset = 0
y_offset = 0
for idx, img in enumerate(images, start=1):
    new_image.paste(img, (x_offset, y_offset))
    y_offset += img.height

    # 4の倍数の場合、次の行に移る
    if idx % 4 == 0:
        x_offset += img.width
        y_offset = 0


# 黒い余白をトリミングして新しい画像を作成
bbox = new_image.getbbox()  # 画像を囲む境界ボックスを取得
new_image = new_image.crop(bbox)

# 新しい画像を表示するか保存する
new_image.show()  # 画像を表示する場合
new_image.save('2024transe1.png',quality = 100)  # 画像を保存する場

out = '/app/Oba_卒業研究A/for_emotional_engineering'

out_path = os.path.join(out, '')