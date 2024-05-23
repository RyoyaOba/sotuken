
# import tensorflow as tf

# from utils.make_dataset import CustomDataset

# # Load dataset

# ds = CustomDataset('/app/Oba_卒業研究A/2023年度歩容測定実験/venus3d_data/normalization_data/sotukenB_clustering')


# from utils.make_dataset import CustomDataset

# # Assuming 'folder_path_cluster0' and 'folder_path_cluster1' are the respective folder paths for the datasets

# # The instances cluster0 and cluster1 will have data with labels [1, 0] and [0, 1] respectively.


# train_ds, val_ds, test_ds = ds(size=ds.__len__(), batch_size=1)
# train_x, _ = next(iter(train_ds))
# # input_shape = train_x.shape
# input_shape = (399, 20)

# model_name = "202309132315"

# #別のpythonファイルにて，from utils.make_dataset import CustomDataset　を用い，
# #すでにCustomDatasetで作成した[1,0]にラベル付けされたデータと[0,1]にラベル付けされたデータをそれぞれ読み込むことはできますか。
# # 
# # Tensorboard settings
# log_dir = f"logs/fit/{model_name}"
# tensorboard_callback = tf.keras.callbacks.TensorBoard(
#     log_dir=log_dir, histogram_freq=1
# )

# # Train Model
# model = tf.keras.models.Sequential([
#     tf.keras.layers.Flatten(input_shape=input_shape),
#     tf.keras.layers.Dense(2048, activation = 'relu'),
#     tf.keras.layers.Dense(2048, activation="relu"),
#     tf.keras.layers.Dense(2048, activation='relu'),
#     tf.keras.layers.Dense(2, activation="softmax")
# ])


# model.compile(optimizer='SGD',
#             # loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
#               loss=tf.keras.losses.BinaryCrossentropy(),
#             # loss=tf.keras.losses.MeanAbsoluteError(),
#               metrics=['accuracy'])

# model.fit(train_ds, validation_data=val_ds, epochs=200, callbacks=[tensorboard_callback])


# test_loss, test_acc = model.evaluate(test_ds, verbose=2)

# print('\nTest accuracy:', test_acc)


import os
import numpy as np
import pandas as pd
import tensorflow as tf
from utils.make_dataset import CustomDataset

# Load dataset
ds = CustomDataset('/app/hinann/sotukenB_clustering')

train_ds, val_ds, test_ds = ds(size=ds.__len__(), batch_size=1)
train_x, _ = next(iter(train_ds))
input_shape = (399, 20)

model_name = "202309132315"

log_dir = f"logs/fit/{model_name}"
tensorboard_callback = tf.keras.callbacks.TensorBoard(
    log_dir=log_dir, histogram_freq=1
)

def make_classification(train_data, val_data):
    inputs = tf.keras.layers.Input(shape=(399, 20))
    
    cnn_outs = []
    for i in range(20):
        _input = inputs[:, :, i:i+1] 
        x = tf.keras.layers.Conv1D(32, 25, strides=1, padding="causal", activation='tanh')(_input)
        x = tf.keras.layers.MaxPool1D(pool_size=2, padding='valid')(x)
        x = tf.keras.layers.Conv1D(64, 10, strides=1, padding="causal", activation='tanh')(x)
        x = tf.keras.layers.MaxPool1D(pool_size=2, padding='valid')(x)
        out = tf.keras.layers.Flatten()(x)
        cnn_outs.append(out)

    merged = tf.keras.layers.concatenate()(cnn_outs)#Concatenate?
    merged = tf.keras.layers.Dense(256, activation='tanh')(merged)
    merged = tf.keras.layers.Dense(64, activation='tanh')(merged)
    merged = tf.keras.layers.Dense(16, activation='tanh')(merged)
    output = tf.keras.layers.Dense(2, activation='softmax')(merged)

    model = tf.keras.Model(inputs=[inputs], outputs=[output])
    
    opt = tf.keras.optimizers.SGD(learning_rate=1e-5, momentum=0.9)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    
    model.summary()

    es_cb = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=200, verbose=1, mode="auto")

    class PrintEpochNumber(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            print(f"Epoch {epoch + 1}/{self.params['epochs']}")

    checkpoint_path = "./checkpoints/classification_2"
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(checkpoint_path, "best_model2.h5"),
        monitor='val_loss',
        mode='min',
        save_best_only=True,
        save_weights_only=False,
        verbose=1
    )

    class SaveLossCallback(tf.keras.callbacks.Callback):
        def on_train_begin(self, logs=None):
            self.loss_history = []

        def on_epoch_end(self, epoch, logs=None):
            row = {
                'epoch': epoch + 1,
                'loss': logs['loss'],
                'accuracy': logs['accuracy'],
                'val_loss': logs['val_loss'],
                'val_accuracy': logs['val_accuracy']
            }
            self.loss_history.append(row)
            columns = ['epoch', 'loss', 'accuracy', 'val_loss', 'val_accuracy']
            loss_df = pd.DataFrame(self.loss_history, columns=columns)
            loss_df.to_csv('loss_history2.csv', index=False)

    model.fit(train_data, validation_data=val_data, 
              epochs=1000,
              batch_size=64,
              callbacks=[tensorboard_callback, checkpoint_callback, PrintEpochNumber(), SaveLossCallback(), es_cb])

    return model

classification_model = make_classification(train_ds, val_ds)
tf.keras.utils.plot_model(classification_model, to_file='classification_model.png')
tf.keras.models.save_model(classification_model, "./classification/classification_2")

