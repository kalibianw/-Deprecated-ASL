from utils import AnnoTrainModule, AnnoDataModule

from sklearn.model_selection import train_test_split

import numpy as np
import shutil
import os


def mkdir(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return path


def rmkdir(path):
    if os.path.exists(os.path.dirname(path)):
        shutil.rmtree(os.path.dirname(path))
        os.makedirs(os.path.dirname(path), exist_ok=True)
    else:
        os.makedirs(os.path.dirname(path), exist_ok=True)

    return path


RESCALING_RATIO = 0.5
NPZ_PATH = f"npz/AnnoNumpy_{RESCALING_RATIO}.npz"
CKPT_PATH = rmkdir(f"ckpt/anno_{RESCALING_RATIO}/anno_{RESCALING_RATIO}.ckpt")
MODEL_PATH = mkdir(f"model/anno_{RESCALING_RATIO}.h5")
LOG_DIR = rmkdir(f"log/anno_{RESCALING_RATIO}/")
BATCH_SIZE = 32
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
    rescaling_ratio=RESCALING_RATIO,
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
    y_lndmrk_valid=y_lndmrk_valid,
    CALLBACKS_MONITOR="val_lndmrk_out_mean_absolute_error"
)
