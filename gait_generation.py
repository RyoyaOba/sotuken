
import os
import time
import random
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
#-------------------#
# GPU使いたくないとき 
#-------------------#
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf
#from tensorflow.python.keras import backend as K
from utils.make_GanDataset import GanDatasets
#----------------------
# 分散型トレーニング

AUTOTUNE = tf.data.AUTOTUNE

#データセットの読み込み
ds = GanDatasets('/app/Walking_Clustering/sotukenB_clustering')

ds10_samples, ds01_samples, _, _ = ds.load_ds(sample_size=10560)
ds.split_data(ds10_samples, ds01_samples)

ds10_train = ds.dataset_10_train  # dataset_10_trainを取得
ds01_train = ds.dataset_01_train 

# サイズを出力
print(f"Dataset 10 Train size: {len(ds10_train)}")
print(f"Dataset 01 Train size: {len(ds01_train)}")

# print(ds10_train.shape)

def senet_block(inputs, channels):
    se = tf.keras.layers.GlobalAveragePooling1D()(inputs)
    se = tf.keras.layers.Reshape((1, channels))(se)

    se = tf.keras.layers.Dense(channels // 16, activation='relu', kernel_initializer='he_normal')(se)
    se = tf.keras.layers.Dense(channels, activation='sigmoid', kernel_initializer='he_normal')(se)

    se = tf.keras.layers.Reshape((1, channels))(se)
    return tf.keras.layers.multiply([inputs, se])

# 使用例:
def temporal_senet_block_16(inputs):
    return senet_block(inputs, 16)

def temporal_senet_block_32(inputs):
    return senet_block(inputs, 32)

def joint_cnn_block(inputs, selected_joint):
    cnn_outs = []
    for i in selected_joint:
        _input = inputs[:, :, i:i+1]
        x = tf.keras.layers.Conv1D(16, kernel_size=25, strides=1, padding="same", activation = 'tanh')(_input)
        x = tf.keras.layers.MaxPool1D(pool_size=2, padding='valid')(x)
        x= tf.keras.layers.Conv1D(32, kernel_size=10, strides=1, padding="same", activation='tanh')(x)
        x = tf.keras.layers.MaxPool1D(pool_size=2, padding='valid')(x)
        out = tf.keras.layers.Flatten()(x)

        cnn_outs.append(out)
    
    merged = tf.keras.layers.Concatenate()(cnn_outs)
    merged = tf.keras.layers.Dense(256, activation='tanh')(merged)
    merged = tf.keras.layers.Dense(64, activation='tanh')(merged)
    joint_weight = tf.keras.layers.Dense(16, activation='tanh')(merged)
    
    return joint_weight

#-------------------------------------------------------------------------
# 0: 体幹_矢状，1: 体幹_前額，2: 右股関節角度_矢状面，3: 右股関節角度_前額面
# 4: 左股関節角度_矢状面, 5: 左股関節角度_前額面, 6: 右膝関節角度_矢状面, 7: 右膝関節角度_前額面
# 8: 左膝関節角度_矢状面, 9: 左膝関節角度_前額面, 10: 右足関節角度_矢状面, 11: 右足関節角度_前額面
# 12: 左足関節角度_矢状面, 13: 左足関節角度_前額面, 14: L_FP1-Fx(N/kg)，15: L_FP1-Fy(N/kg)
# 16: L_FP1-Fz(N/kg), 17: R_FP2-Fx(N/kg), 18: R_FP2-Fy(N/kg), 19: R_FP2-Fz(N/kg)
#-------------------------------------------------------------------------
# def make_generator():
#     inputs = tf.keras.layers.Input(shape=(399, 20))
#     cnn_outs = []

#     # 選択された関節ごとに異なる重みを取得
#     selected_joints = [
#                        [2, 6, 10], #右脚_矢状面(股関節，膝関節，足関節)
#                        [3, 7, 11], #右脚_前額面(股関節，膝関節，足関節)
#                        [4, 8, 12], #左脚_矢状面(股関節，膝関節，足関節)
#                        [5, 9, 13]  #左脚_前額面(股関節，膝関節，足関節)
#                        ] 
    
#     # selected_joints = [
#     #     [0,1],#体幹
#     #     [2,6],[4,8],#右股，右膝矢状面
#     #     [3,7],[5,9],#前額面
#     #     [6,10],[8,12],#膝，足
#     #     [7,11],[9,13]
#     # ]

#     # 関節ごとの重みをランダムに初期化
   
#     joint_weights = [joint_cnn_block(inputs, selected_joint) for selected_joint in selected_joints]

#     for i in range(20):
#         _input = inputs[:, :, i:i+1]

#         # # 関節ごとの重みを適用
#         # weighted_inputs = [tf.keras.layers.Dense(32, activation='tanh')(joint_weight) for joint_weight in joint_weights]
#         # x = tf.keras.layers.Multiply()([_input] + weighted_inputs)

#         gates = [tf.keras.layers.Dense(1, activation='sigmoid')(joint_weight) for joint_weight in joint_weights]
#         weighted_inputs = [_input * gate for gate, _input in zip(gates, inputs)]
#         x = tf.keras.layers.Add()(weighted_inputs)

#         x = tf.keras.layers.Conv1DTranspose(256, kernel_size=5, strides=1, padding="same")(x)
#         x = tf.keras.layers.LayerNormalization()(x)
#         x = tf.keras.layers.LeakyReLU(alpha=0.01)(x)
  
#         x = tf.keras.layers.Conv1DTranspose(128, kernel_size=5, strides=1, padding='same')(x)
#         x = tf.keras.layers.LayerNormalization()(x)
#         x = tf.keras.layers.LeakyReLU(alpha=0.01)(x)

#         x = tf.keras.layers.Conv1DTranspose(64, kernel_size=7, strides=1, padding='same')(x)
#         x = tf.keras.layers.LayerNormalization()(x)
#         x = tf.keras.layers.LeakyReLU(alpha=0.01)(x)

#         x = tf.keras.layers.Conv1DTranspose(32, kernel_size=9, strides=1, padding='same')(x)
#         #x = tf.keras.layers.LayerNormalization(axis=-1)(x)
#         x = tf.keras.layers.LeakyReLU(alpha=0.01)(x)

#         x = tf.keras.layers.Conv1DTranspose(1, kernel_size=9, strides=1, padding='same', activation = 'tanh')(x)

#         #x = tf.keras.activations.tanh(x)

#         cnn_outs.append(x)

#     merged = tf.keras.layers.Concatenate()(cnn_outs)
#     model = tf.keras.Model(inputs=inputs, outputs=merged)
#     model.summary()
#     return model



def make_generator():
    inputs = tf.keras.layers.Input(shape=(399, 20))
    cnn_outs = []

    # 選択された関節ごとに異なる重みを取得
    selected_joints = [
                       [0, 1],
                       [2, 6, 10], #右脚_矢状面(股関節，膝関節，足関節)
                       [3, 7, 11], #右脚_前額面(股関節，膝関節，足関節)
                       [4, 8, 12], #左脚_矢状面(股関節，膝関節，足関節)
                       [5, 9, 13]  #左脚_前額面(股関節，膝関節，足関節)
                       ] 
    
    # selected_joints = [
    #     [0,1],#体幹
    #     [2,6],[4,8],#右股，右膝矢状面
    #     [3,7],[5,9],#前額面
    #     [6,10],[8,12],#膝，足
    #     [7,11],[9,13]
    # ]

    # 関節ごとの重みをランダムに初期化
   
    joint_weights = [joint_cnn_block(inputs, selected_joint) for selected_joint in selected_joints]

    for i in range(20):
        _input = inputs[:, :, i:i+1]

        # # 関節ごとの重みを適用
        weighted_inputs = [tf.keras.layers.Dense(1)(joint_weight) for joint_weight in joint_weights]
        x = tf.keras.layers.Multiply()([_input] + weighted_inputs)

        x = tf.keras.layers.Conv1DTranspose(256, kernel_size=3, strides=1, padding="same")(x)
        x = tf.keras.layers.LayerNormalization(axis=-1)(x)
        x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
  
        x = tf.keras.layers.Conv1DTranspose(128, kernel_size=5, strides=1, padding='same')(x)
        x = tf.keras.layers.LayerNormalization(axis=-1)(x)
        x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
  
        #x = tf.keras.layers.LeakyReLU(alpha=0.01)(x)

        x = tf.keras.layers.Conv1DTranspose(64, kernel_size=7, strides=1, padding='same')(x)
        x = tf.keras.layers.LayerNormalization(axis=-1)(x)
        x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
  
        #x = tf.keras.layers.LeakyReLU(alpha=0.01)(x)

        x = tf.keras.layers.Conv1DTranspose(32, kernel_size=9, strides=1, padding='same')(x)
        x = tf.keras.layers.LayerNormalization(axis=-1)(x)
        x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
  
        #x = tf.keras.layers.LeakyReLU(alpha=0.01)(x)

        x = tf.keras.layers.Conv1DTranspose(1, kernel_size=12, strides=1, padding='same', activation = 'tanh')(x)

        #x = tf.keras.activations.tanh(x)

        cnn_outs.append(x)

    merged = tf.keras.layers.Concatenate()(cnn_outs)
    model = tf.keras.Model(inputs=inputs, outputs=merged)
    model.summary()
    return model



def make_discriminator():
    inputs = tf.keras.layers.Input(shape=(399, 20))
    cnn_outs = []

    for i in range(20):
        _input = inputs[:, :, i:i+1] 

        x = tf.keras.layers.Conv1D(32, kernel_size=20, strides=1, padding="causal")(_input)
        x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)

        x = tf.keras.layers.Dropout(0.25)(x)

        x = temporal_senet_block_32(x)
        x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)

        x = tf.keras.layers.GlobalMaxPooling1D()(x)
        
        out = tf.keras.layers.Dropout(0.25)(x)
        cnn_outs.append(out)

    merged = tf.keras.layers.Concatenate()(cnn_outs)
    merged = tf.keras.layers.Dense(64)(merged)
    merged = tf.keras.layers.LeakyReLU(alpha=0.2)(merged)
    merged = tf.keras.layers.Dropout(0.10)(merged)
    output = tf.keras.layers.Dense(1)(merged) 

    model = tf.keras.Model(inputs=inputs, outputs=output)
    model.summary()
    return model

# 使い方
generator_model = make_generator()
discriminator_model = make_discriminator()

tf.keras.utils.plot_model(generator_model, to_file = "generator_model.png")

tf.keras.utils.plot_model(discriminator_model, to_file = "discriminator_model.png")
#------------------
# 損失関数
#=---------------
#cd Oba_卒業研究
C_LAMBDA = 10
I_LAMBDA = 5
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
loss_obj_hinge = tf.keras.losses.Hinge()
#cd Oba_卒業研究A
loss_obj_mse = tf.keras.losses.MeanSquaredError()
# 交差エントロピー
# Hinge Loss
loss_obj_hinge =tf.keras.losses.Hinge() 
# 平均絶対誤差
loss_obj_mae = tf.keras.losses.MeanAbsoluteError()

def discriminator_loss(real, generated):
    real_loss = cross_entropy(tf.ones_like(real), real)
    
    generated_loss = cross_entropy(tf.zeros_like(generated), generated)

    total_disc_loss = real_loss + generated_loss

    return total_disc_loss * 0.5

def generator_loss(generated):
    return cross_entropy(tf.ones_like(generated), generated)

def calc_cycle_loss(real_image, cycled_image):
    loss_1 = tf.reduce_mean(tf.abs(real_image - cycled_image))
    return C_LAMBDA * loss_1

def identity_loss(real_image, same_image):
    loss_2 = tf.reduce_mean(tf.abs(real_image - same_image))
    return I_LAMBDA * loss_2

generator_g = make_generator()
generator_f = make_generator()#pix2pix.unet_generator(OUTPUT_CHANNELS, norm_type='instancenorm')
#generator_f = make_generator()#pix2pix.unet_generator(OUTPUT_CHANNELS, norm_type='instancenorm')
discriminator_x = make_discriminator()
discriminator_y = make_discriminator()

generator_g_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
generator_f_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

discriminator_x_optimizer = tf.keras.optimizers.Adam(1e-4, beta_1=0.5)
discriminator_y_optimizer = tf.keras.optimizers.Adam(1e-4, beta_1=0.5)

checkpoint_dir = "./checkpoints/cycle_23"
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_g=generator_g,
                           generator_f=generator_f,
                           discriminator_x=discriminator_x,
                           discriminator_y=discriminator_y,
                           generator_g_optimizer=generator_g_optimizer,
                           generator_f_optimizer=generator_f_optimizer,
                           discriminator_x_optimizer=discriminator_x_optimizer,
                           discriminator_y_optimizer=discriminator_y_optimizer)

BUFFER_SIZE = 50  # Number of images to store in the buffer
buffer_A = []
buffer_B = []

def update_buffer(buffer, image, BUFFER_SIZE):
    if len(buffer) < BUFFER_SIZE:
        buffer.append(image)
    else:
        # Replace the oldest image with the new one
        idx = random.randint(0, BUFFER_SIZE - 1)
        buffer[idx] = image

EPOCHS =1000 #200
BATCH_SIZE = 32 #32がきのう

@tf.function
def train_step(ds10_train, ds01_train):
  # persistent is set to True because the tape is used more than
  # once to calculate the gradients.
    with tf.GradientTape(persistent=True) as tape:
        # Generator G translates X -> Y
        # Generator F translates Y -> X.

        fake_y = generator_g(ds10_train, training=True)
        cycled_x = generator_f(fake_y, training=True)

        fake_x = generator_f(ds01_train, training=True)
        cycled_y = generator_g(fake_x, training=True)

        ds10_train= tf.cast(ds10_train, tf.float32)
        ds01_train = tf.cast(ds01_train, tf.float32)
        cycled_x = tf.cast(cycled_x, tf.float32)
        cycled_y = tf.cast(cycled_y, tf.float32)

        # same_x and same_y are used for identity loss.
        same_x = generator_f(ds10_train, training=True)
        same_y = generator_g(ds01_train, training=True)

        disc_real_x = discriminator_x(ds10_train, training=True)
        disc_real_y = discriminator_y(ds01_train, training=True)

        disc_fake_x = discriminator_x(fake_x, training=True)
        disc_fake_y = discriminator_y(fake_y, training=True)

        # calculate the loss
        gen_g_loss = generator_loss(disc_fake_y)
        gen_f_loss = generator_loss(disc_fake_x)

        total_cycle_loss = calc_cycle_loss(ds10_train, cycled_x) + calc_cycle_loss(ds01_train, cycled_y)

        # Total generator loss = adversarial loss + cycle loss
        total_gen_g_loss = gen_g_loss + total_cycle_loss + identity_loss(ds01_train, same_y) 
        total_gen_f_loss = gen_f_loss + total_cycle_loss + identity_loss(ds10_train, same_x) 

        disc_x_loss = discriminator_loss(disc_real_x, disc_fake_x)
        disc_y_loss = discriminator_loss(disc_real_y, disc_fake_y)
                

    # Calculate the gradients for generator and discriminator
    
    generator_g_gradients = tape.gradient(total_gen_g_loss, 
                                        generator_g.trainable_variables)
    generator_f_gradients = tape.gradient(total_gen_f_loss, 
                                        generator_f.trainable_variables)

    discriminator_x_gradients = tape.gradient(disc_x_loss, 
                                            discriminator_x.trainable_variables)
    discriminator_y_gradients = tape.gradient(disc_y_loss, 
                                            discriminator_y.trainable_variables)

    # Apply the gradients to the optimizer
    generator_g_optimizer.apply_gradients(zip(generator_g_gradients, 
                                            generator_g.trainable_variables))

    generator_f_optimizer.apply_gradients(zip(generator_f_gradients, 
                                            generator_f.trainable_variables))

    discriminator_x_optimizer.apply_gradients(zip(discriminator_x_gradients,
                                                discriminator_x.trainable_variables))

    discriminator_y_optimizer.apply_gradients(zip(discriminator_y_gradients,
                                                discriminator_y.trainable_variables))
    
    generated_image_A = generator_g(ds10_train, training=True)
    generated_image_B = generator_f(ds01_train, training=True)
    update_buffer(buffer_A, generated_image_A, BUFFER_SIZE)
    update_buffer(buffer_B, generated_image_B, BUFFER_SIZE)

    return gen_g_loss, gen_f_loss, disc_x_loss, disc_y_loss

def train(dataset_A, dataset_B, epochs, csv_filename, save_generator_every=25):
    losses = {
    "gen_g_loss": [],
    "gen_f_loss":[],
    "disc_x_loss":[],
    "disc_y_loss": []
    }
    with open(csv_filename, mode='w', newline='') as csv_file:
        fieldnames = ["Epoch", "Gen G Loss", "Gen F Loss", "Disc X Loss", "Disc Y Loss"]
        csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        csv_writer.writeheader()

    # CSVファイルを追記モードで開く
    with open(csv_filename, mode='a', newline='') as csv_file:
        fieldnames = ["Epoch", "Gen G Loss", "Gen F Loss", "Disc X Loss", "Disc Y Loss"]
        csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

        # ファイルが空の場合はヘッダーを書き込む
        if csv_file.tell() == 0:
            csv_writer.writeheader()

        for epoch in tqdm(range(epochs)):
            start = time.time()

            for image_A, image_B in zip(dataset_A, dataset_B):
                gen_g_loss, gen_f_loss, disc_x_loss, disc_y_loss = train_step(image_A, image_B)

            if (epoch+1) % 10 == 0:
                checkpoint.save(file_prefix=checkpoint_prefix + f"_epoch{epoch + 1}")

            if (epoch) % save_generator_every == 0:
                save_epoch = (epoch) // save_generator_every
                generator_f.save(f"./generator_f/model_c23_epoch{save_epoch}")
                generator_g.save(f"./generator_g/model_c23_epoch{save_epoch}")

            tqdm.write('Time for epoch {} is {} sec'.format(epoch + 1, time.time() - start))

            # 現在のエポックの損失を追加
            losses["gen_g_loss"].append(gen_g_loss.numpy().item())
            losses["gen_f_loss"].append(gen_f_loss.numpy().item())
            losses["disc_x_loss"].append(disc_x_loss.numpy().item())
            losses["disc_y_loss"].append(disc_y_loss.numpy().item())
            
            #losses_np = {key: value.numpy() for key, value in losses.items()}
            #tqdm.write(f"Losses: {losses}")
            #値のみcsvデータに追加
            gen_g_loss_value = gen_g_loss.numpy().item() \
                if isinstance(gen_g_loss, tf.Tensor) else gen_g_loss
            gen_f_loss_value = gen_f_loss.numpy().item() \
                if isinstance(gen_f_loss, tf.Tensor) else gen_f_loss
            disc_x_loss_value = disc_x_loss.numpy().item()\
                if isinstance(disc_x_loss, tf.Tensor) else disc_x_loss
            disc_y_loss_value = disc_y_loss.numpy().item()\
                if isinstance(disc_y_loss, tf.Tensor) else disc_y_loss
            
            csv_writer.writerow({
                "Epoch": epoch + 1,
                "Gen G Loss": losses["gen_g_loss"][-1],
                "Gen F Loss": losses["gen_f_loss"][-1],
                "Disc X Loss": losses["disc_x_loss"][-1],
                "Disc Y Loss": losses["disc_y_loss"][-1]
            })

            # losses["gen_g_loss"].append(gen_g_loss_value)
            # losses["gen_f_loss"].append(gen_f_loss_value)
            # losses["disc_x_loss"].append(disc_x_loss_value)
            # losses["disc_y_loss"].append(disc_y_loss_value)

            # エポックごとの損失を表示
                        # 現在のエポックの損失を表示
            tqdm.write(f"Losses at epoch {epoch + 1}:")
            tqdm.write(f"Gen G Loss: {losses['gen_g_loss'][-1]}")
            tqdm.write(f"Gen F Loss: {losses['gen_f_loss'][-1]}")
            tqdm.write(f"Disc X Loss: {losses['disc_x_loss'][-1]}")
            tqdm.write(f"Disc Y Loss: {losses['disc_y_loss'][-1]}")

    return losses

csv_filename = "gen_losses_model_c23.csv"
# データセットの準備
dataset_A = tf.data.Dataset.from_tensor_slices(ds10_train).shuffle(10000).batch(BATCH_SIZE)
dataset_B = tf.data.Dataset.from_tensor_slices(ds01_train).shuffle(10000).batch(BATCH_SIZE)

losses = train(dataset_A, dataset_B, EPOCHS, csv_filename, save_generator_every = 25)
#losses = train(dist_dataset_A, dist_dataset_B, EPOCHS)

# generator.save("./generator/ver1.h5")
tf.keras.models.save_model(generator_f, "./generator_f/model_c23")
tf.keras.models.save_model(generator_g, "./generator_g/model_c23")
tf.keras.models.save_model(discriminator_x,"./discriminator_x/model_c23")
tf.keras.models.save_model(discriminator_y,"./discriminator_y/model_c23")

