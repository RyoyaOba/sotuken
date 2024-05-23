
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from utils.make_dataset import CustomDataset

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# データセットの初期化
ds = CustomDataset('/app/hinann/sotukenB_clustering')
train_ds, val_ds, test_ds = ds(size=ds.__len__(), batch_size=1)

# モデルのロード
loaded_model = tf.keras.models.load_model("/app/Oba_卒業研究A/checkpoints/classification_2/best_model2.h5")

test_loss, test_accuracy = loaded_model.evaluate(test_ds, verbose=1)
print(f"Test Accuracy: {test_accuracy*100:.2f}%")

predictions = loaded_model.predict(test_ds)

# csvの名前とラベルを取得
test_csv_names = [data_info["csv_name"] for data_info in ds.datasets[-len(test_ds):]]
original_labels = [data_info["label"] for data_info in ds.datasets[-len(test_ds):]]

# 結果の出力するための[]
output_data = []

# Initialize variables for calculating accuracy
correct_predictions = 0
total_samples = len(test_csv_names)


# CSVファイル名、元のラベル、対応するソフトマックス出力を表示
for csv_name, label, prediction in zip(test_csv_names, original_labels, predictions):
    predicted_label = np.argmax(prediction)
    softmax_output_str = "[" + ", ".join(map(str, prediction)) + "]"

    print(f"CSV File Name: {csv_name}")
    print(f"Original Label: {label}")
    print("Softmax Output:", prediction)
    print(f"Predicted Label : {predicted_label}")
    print()

    # Check if the prediction is correct
    if predicted_label == label:
        correct_predictions += 1

    # Append information to the output_data list
    output_data.append({
        "CSV File Name": csv_name,
        "Original Label": label,
        "Softmax Output": softmax_output_str,
        "Predicted Label": predicted_label,
        "Probability Label 1": prediction[1]  # Probability of label 1
    })

# Create a DataFrame from the output_data list
output_df = pd.DataFrame(output_data)

# Sort the DataFrame based on the probability of label 1
output_df = output_df.sort_values(by="Probability Label 1", ascending=False)

# Specify the output directory
output_dir = "/app/Oba_卒業研究A/for_emotional_engineering"
os.makedirs(output_dir, exist_ok=True)

# Save the DataFrame to a CSV file with float formatting
output_csv_path = os.path.join(output_dir, "softmax_test2_results.csv")
output_df.to_csv(output_csv_path, index=False, float_format='%.20f')  # Adjust the precision as needed

# Calculate and print test accuracy
test_accuracy = (output_df["Original Label"] == output_df["Predicted Label"]).mean()
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
# print(f"Output saved to: {output_csv_path}")
#     output.append({
#         "CSV Name": csv_name,
#         "Original Label": label,
#         "Softmax Output": softmax_output_str,
#         "Predicted Label": predicted_label
#     })
# # Create a DataFrame from the output_data list
# output_df = pd.DataFrame(output)

# # Save the DataFrame to a CSV file
# output_csv_path = "./output_results.csv"  # Specify your desired output path
# output_df.to_csv(output_csv_path, index=False)

# # Calculate and print test accuracy
# test_accuracy = correct_predictions / total_samples
# print(f"Test Accuracy: {test_accuracy * 100:.2f}%")



# # import os
# # import numpy as np
# # import pandas as pd
# # import tensorflow as tf
# # from utils.make_dataset import CustomDataset

# # os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# # # データセットの初期化
# # ds = CustomDataset('/app/Oba_卒業研究A/2023年度歩容測定実験/venus3d_data/normalization_data/sotukenB_clustering')
# # train_ds, val_ds, test_ds = ds(size=ds.__len__(), batch_size=1)

# # # モデルのロード
# # loaded_model = tf.keras.models.load_model("/app/Oba_卒業研究A/checkpoints/classification_1/best_model.h5")

# # # Obtain predictions and corresponding CSV file names
# # predictions = loaded_model.predict(test_ds)

# # # Get the CSV file names, original labels, and softmax outputs for the test data
# # test_csv_names = [data_info["csv_name"] for data_info in ds.datasets[-len(test_ds):]]
# # original_labels = [data_info["label"] for data_info in ds.datasets[-len(test_ds):]]

# # printed_csv_count = 0

# # # Display CSV file names, original labels, and softmax outputs in descending order
# # for csv_name, label, prediction in zip(test_csv_names, original_labels, predictions):
# #     print(f"CSV File Name: {csv_name}")
# #     print(f"Original Label: {label}")
    
# #     # # Sort the softmax outputs in descending order
# #     # sorted_indices = np.argsort(prediction)[::-1]
# #     # sorted_softmax = prediction[sorted_indices]
    
# #     # print("Sorted Softmax Output:", sorted_softmax)
# #     # print("Sorted Indices:", sorted_indices)
# #     # print()

# #     printed_csv_count += 1

# # # Print the total number of printed CSV files
# # print(f"Total Printed CSV Count: {printed_csv_count}")
