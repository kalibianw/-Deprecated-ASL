from utils import AnnoTrainModule, AnnoDataModule

from sklearn.model_selection import train_test_split

import numpy as np
import os


def mkdir(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return path


NPZ_PATH = "npz/AnnoNumpy_0.5.npz"
CKPT_PATH = mkdir("ckpt/anno_0.5/anno_0.5.ckpt")
MODEL_PATH = mkdir("model/anno_0.5.h5")
LOG_DIR = mkdir("log/anno_0.5/")
BATCH_SIZE = 8
VALID_SIZE = 0.2

NPZ_LOADER = np.load(NPZ_PATH)
for key in NPZ_LOADER:
    print(key)

# Load dataset
x_train, y_cls_train, y_lndmrk_train = NPZ_LOADER["x_train"], NPZ_LOADER["y_cls_train"], NPZ_LOADER["y_lndmrk_train"]

print(x_train.shape, y_cls_train.shape, y_lndmrk_train.shape)
print(np.unique(y_cls_train, return_counts=True))

adm = AnnoDataModule(
    dataset_dir_path=None,
    rescaling_ratio=0.5,
    img_height=x_train.shape[1],
    img_width=x_train.shape[2]
)

# Data Normalization (Image will be normalized during the training)
y_cls_train, y_lndmrk_train = adm.label_normalization(chars=y_cls_train, landmarks=y_lndmrk_train)
print(x_train.shape, y_cls_train.shape, y_lndmrk_train.shape)

# Reshape y_lndmrk
y_lndmrk_train = np.reshape(y_lndmrk_train, newshape=(y_lndmrk_train.shape[0], y_lndmrk_train.shape[1] * y_lndmrk_train.shape[2]))

# Train and Valid split
x_train, x_valid, y_cls_train, y_cls_valid, y_lndmrk_train, y_lndmrk_valid = train_test_split(
    x_train, y_cls_train, y_lndmrk_train, test_size=0.2, stratify=y_cls_train
)
print(x_train.shape, x_valid.shape)
print(y_cls_train.shape, y_cls_valid.shape)
print(y_lndmrk_train.shape, y_lndmrk_valid.shape)

atm = AnnoTrainModule(
    input_shape=x_train.shape[1:],
    batch_size=BATCH_SIZE,
    ckpt_path=CKPT_PATH,
    model_path=MODEL_PATH,
    log_dir=LOG_DIR
)
model = atm.create_model(num_conv_blocks=6)
model.summary()
model.save(filepath=MODEL_PATH)

atm.train(
    model=model,
    x_train=x_train,
    y_cls_train=y_cls_train,
    y_lndmrk_train=y_lndmrk_train,
    x_valid=x_valid,
    y_cls_valid=y_cls_valid,
    y_lndmrk_valid=y_lndmrk_valid
)
