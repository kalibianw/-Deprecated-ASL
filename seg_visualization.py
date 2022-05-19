from utils import SegVisualModule

import numpy as np

npz_loader = np.load("npz/SegNumpy_0.5.npz")
for key in npz_loader:
    print(key)

imgs = npz_loader["x_train"]
chars = npz_loader["y_cls_train"]
seg_imgs = npz_loader["y_seg_train"]
print(imgs.shape, chars.shape, seg_imgs.shape)
print(np.unique(chars, return_counts=True))

svm = SegVisualModule(is_chars_normalized=False)
svm.draw_point(imgs[:5], chars[:5], seg_imgs[:5])
