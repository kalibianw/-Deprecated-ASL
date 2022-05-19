from utils import AnnoDataModule, AnnoVisualModule

import numpy as np
import cv2

NUM_IMG = 0

npz_loader = np.load("npz/AnnoNumpy_0.5.npz")

for key in npz_loader:
    print(key)

imgs = npz_loader["x_train"]
chars = npz_loader["y_cls_train"]
lndmrks = npz_loader["y_lndmrk_train"]
print(imgs.shape, chars.shape, lndmrks.shape)
print(np.unique(chars, return_counts=True))

print(f"imgs height: {imgs.shape[1]}\nimgs width: {imgs.shape[2]}")
adm = AnnoDataModule(dataset_dir_path=None, rescaling_ratio=0.5, img_height=imgs.shape[1], img_width=imgs.shape[2])
chars = adm.cls_normalization(chars)

avm = AnnoVisualModule(is_chars_normalized=True)
avm.draw_line(imgs[NUM_IMG], lndmrks[NUM_IMG], color=(255, 255, 255))

cv2.imshow(f"{np.argmax(chars[NUM_IMG])}", mat=imgs[NUM_IMG])
cv2.waitKey(0)
cv2.destroyAllWindows()
