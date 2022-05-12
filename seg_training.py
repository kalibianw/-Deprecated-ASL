from utils import SegTrainModule, SegDataModule

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
NPZ_PATH = f"npz/SegNumpy_{RESCALING_RATIO}.npz"
CKPT_PATH = rmkdir(f"ckpt/seg_{RESCALING_RATIO}/seg_{RESCALING_RATIO}.ckpt")
MODEL_PATH = mkdir(f"model/seg_{RESCALING_RATIO}.h5")
LOG_DIR = rmkdir(f"log/seg_{RESCALING_RATIO}/")
BATCH_SIZE = 32
VALID_SIZE = 0.2

CALLBACKS_MONITOR = "val_loss"

npz_loader = np.load(NPZ_PATH)
for key in npz_loader:
    print(key)

# Load dataset
x_train, y_cls_train, y_seg_train = npz_loader["x_train"], npz_loader["y_cls_train"], npz_loader["y_seg_train"]

print(x_train.shape, y_cls_train.shape, y_seg_train.shape)
print(np.unique(y_cls_train, return_counts=True))

sdm = SegDataModule(
    dataset_dir_path=None,
    rescaling_ratio=RESCALING_RATIO,
    img_height=x_train.shape[1],
    img_width=x_train.shape[2]
)

# Data Normalization (Image will be normalized during the training)
y_cls_train = sdm.cls_normalization(chars=y_cls_train)

print(x_train.shape, y_cls_train.shape, y_seg_train.shape)

# Train and Valid split
x_train, x_valid, y_cls_train, y_cls_valid, y_seg_train, y_seg_valid = train_test_split(
    x_train, y_cls_train, y_seg_train, test_size=0.2, stratify=y_cls_train
)
print(x_train.shape, x_valid.shape)
print(y_cls_train.shape, y_cls_valid.shape)
print(y_seg_train.shape, y_seg_valid.shape)

stm = SegTrainModule(
    input_shape=x_train.shape[1:],
    batch_size=BATCH_SIZE,
    ckpt_path=CKPT_PATH,
    model_path=MODEL_PATH,
    log_dir=LOG_DIR
)
model = stm.create_model(num_conv_blocks=4)
model.summary()
model.save(filepath=MODEL_PATH)

stm.train(
    model=model,
    x_train=x_train,
    x_valid=x_valid,
    y_seg_train=y_seg_train,
    y_seg_valid=y_seg_valid,
    callbacks_monitor=CALLBACKS_MONITOR
)
