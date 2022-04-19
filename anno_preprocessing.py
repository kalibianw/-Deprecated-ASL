from utils import AnnoDataModule
import numpy as np

DATASET_DIR_PATH = "D:/AI/data/ASL Alphabet Synthetic"
RESCALING_RATIO = 1

dm = AnnoDataModule(dataset_dir_path=DATASET_DIR_PATH)

img_fnames, imgs = dm.img_to_np(rescaling_ratio=RESCALING_RATIO)
print(img_fnames.shape, imgs.shape)
np.savez_compressed(file=f"npz/AnnoImgNumpy_{RESCALING_RATIO}.npz", fnames=img_fnames, imgs=imgs)
del img_fnames
del imgs

label_fnames, chars, landmarks = dm.label_to_np(rescaling_ratio=RESCALING_RATIO)
print(label_fnames.shape, chars.shape, landmarks.shape)
np.savez_compressed(file=f"npz/AnnoLabelNumpy_{RESCALING_RATIO}.npz", fnames=label_fnames, chars=chars, landmarks=landmarks)
