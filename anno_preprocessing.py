from utils import AnnoDataModule
from sklearn.model_selection import train_test_split
import numpy as np

DATASET_DIR_PATH = "D:/AI/data/ASL Alphabet Synthetic"
RESCALING_RATIO = 0.5
TEST_SIZE = 0.3
COMPRESSED = False

adm = AnnoDataModule(dataset_dir_path=DATASET_DIR_PATH, rescaling_ratio=RESCALING_RATIO)

img_fnames, imgs = adm.img_to_np()
print(img_fnames.shape, imgs.shape)

label_fnames, chars, landmarks = adm.label_to_np()
print(label_fnames.shape, chars.shape, landmarks.shape)

x_train, x_test, y_cls_train, y_cls_test, y_lndmrk_train, y_lndmrk_test = train_test_split(imgs, chars, landmarks, test_size=TEST_SIZE, stratify=chars)
print(x_train.shape, x_test.shape,
      y_cls_train.shape, y_cls_test.shape,
      y_lndmrk_train.shape, y_lndmrk_test.shape)

if COMPRESSED:
    np.savez_compressed(file=f"npz/AnnoNumpy_{RESCALING_RATIO}_compressed.npz",
                        x_train=x_train, x_test=x_test,
                        y_cls_train=y_cls_train, y_cls_test=y_cls_test,
                        y_lndmrk_train=y_lndmrk_train, y_lndmrk_test=y_lndmrk_test)
else:
    np.savez(file=f"npz/AnnoNumpy_{RESCALING_RATIO}.npz",
             x_train=x_train, x_test=x_test,
             y_cls_train=y_cls_train, y_cls_test=y_cls_test,
             y_lndmrk_train=y_lndmrk_train, y_lndmrk_test=y_lndmrk_test)
