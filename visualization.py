from utils import AnnoDataModule, AnnoVisualModule

import numpy as np

npz_loader = np.load("npz/AnnoNumpy_0.5.npz")

for key in npz_loader:
    print(key)

imgs = npz_loader["x_train"]
chars = npz_loader["y_cls_train"]
lndmrks = npz_loader["y_lndmrk_train"]
print(imgs.shape, chars.shape, lndmrks.shape)
print(np.unique(chars, return_counts=True))

print(f"image height: {imgs.shape[1]}\nimage width: {imgs.shape[2]}")
adm = AnnoDataModule(dataset_dir_path=None, rescaling_ratio=0.5, img_height=imgs.shape[1], img_width=imgs.shape[2])
chars = adm.label_normalization(chars)

avm = AnnoVisualModule(is_chars_normalized=True)
avm.show_output(imgs[:5], chars[:5], lndmrks[:5])
