from tqdm import tqdm

import pandas as pd
import numpy as np
import cv2
import os

from tensorflow.keras import models, layers, activations, initializers, optimizers, metrics, losses, callbacks


class AnnoDataModule:
    def __init__(self, dataset_dir_path):
        self.DATASET_DIR_PATH = dataset_dir_path
        print(f"Find {len(os.listdir(self.DATASET_DIR_PATH))} class(es).")
        print(os.listdir(self.DATASET_DIR_PATH))

    def img_to_np(self, rescaling_ratio=1):
        fnames = list()
        imgs = list()
        for label_name in tqdm(os.listdir(self.DATASET_DIR_PATH), desc="img_to_np"):
            for img_name in tqdm(os.listdir(f"{self.DATASET_DIR_PATH}/{label_name}/lit/")):
                fname = f"{self.DATASET_DIR_PATH}/{label_name}/lit/{img_name}"
                img = cv2.imread(filename=fname)
                img = cv2.resize(src=img, dsize=(0, 0), fx=rescaling_ratio, fy=rescaling_ratio)
                img = cv2.cvtColor(img, code=cv2.COLOR_BGR2RGB)

                imgs.append(img)
                fnames.append(fname)

        return np.array(fnames), np.array(imgs)

    def label_to_np(self, rescaling_ratio=1):
        fnames = list()
        chars = list()
        landmarks = list()
        for label_name in tqdm(os.listdir(self.DATASET_DIR_PATH), desc="label_to_np"):
            for json_fname in tqdm(os.listdir(f"{self.DATASET_DIR_PATH}/{label_name}/annotation/")):
                fname = f"{self.DATASET_DIR_PATH}/{label_name}/annotation/{json_fname}"
                pd_json = pd.read_json(path_or_buf=fname)
                raw_landmark = pd_json["Landmarks"].to_numpy()
                landmark = list()
                for mark in raw_landmark:
                    landmark.append([int(mark[0] * rescaling_ratio), int(mark[1] * rescaling_ratio)])

                fnames.append(fname)
                chars.append(label_name)
                landmarks.append(landmark)

        return np.array(fnames), np.array(chars), np.array(landmarks)


class AnnoTrainModule:
    def __init__(self, input_shape):
        self.INPUT_SHAPE = input_shape

    def create_model(self, num_conv_blocks):
        input_layer = layers.Input(shape=self.INPUT_SHAPE)
        conv_layer_77 = layers.Conv2D(filters=64, kernel_size=(7, 7), padding="same", activation=activations.relu,
                                      kernel_initializer=initializers.he_uniform(), name=f"conv2d_77")(input_layer)
        x = conv_layer_77

        block_cnt = 1
        for i in range(1, num_conv_blocks + 1):
            num_conv_filters = 2 ** (5 + block_cnt)
            x_ = layers.Conv2D(filters=num_conv_filters, kernel_size=(3, 3), padding="same", activation=activations.relu if i <= 2 else activations.selu,
                               kernel_initializer=initializers.he_uniform(), name=f"conv2d_{i}_1")(x)
            x = layers.BatchNormalization(name=f"bn_{i}_1")(x_)
            x = layers.Conv2D(filters=num_conv_filters, kernel_size=(3, 3), padding="same", activation=activations.relu if i <= 2 else activations.selu,
                              kernel_initializer=initializers.he_uniform(), name=f"conv2d_{i}_2")(x)
            x = layers.BatchNormalization(name=f"bn_{i}_2")(x)
            x = layers.Add()([x_, x])
            if i == num_conv_blocks:
                x = layers.AvgPool2D(padding="same", name=f"avg_pool_2d")(x)
                break
            x = layers.MaxPooling2D(padding="same", name=f"max_pool_2d_{i}")(x)

            if i % 2 == 0:
                block_cnt += 1

        x = layers.Flatten()(x)

        x = layers.Dense(1024, activation=activations.selu, kernel_initializer=initializers.he_uniform())(x)

        cls_out = layers.Dense(24, activation=activations.softmax, kernel_initializer=initializers.he_uniform(), name="cls_out")(x)
        landmark_out = layers.Dense(52, activation=activations.softmax, kernel_initializer=initializers.he_uniform(), name="lndmrk_out")(x)

        model = models.Model(input_layer, [cls_out, landmark_out])
        model.compile(
            optimizer=optimizers.Adam(),
            metrics={
                "cls_out": metrics.categorical_accuracy,
                "lndmrk_out": metrics.categorical_accuracy
            },
            loss={
                "cls_out": metrics.categorical_crossentropy,
                "lndmrk_out": metrics.MSE
            },
            run_eagerly=True
        )

        return model
