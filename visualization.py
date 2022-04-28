from utils import AnnoDataModule

import numpy as np
import cv2

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
encoded_chars, lndmrks = adm.label_normalization(
    chars, lndmrks
)

for img, char, lndmrk in zip(imgs[:5], chars[:5], lndmrks[:5]):
    lndmrk[:, 0] *= img.shape[1]
    lndmrk[:, 1] *= img.shape[0]
    lndmrk = np.asarray(lndmrk, dtype=int)
    cv2.imshow(f"{char}", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    pts = np.reshape(a=lndmrk, newshape=(-1, 1, 2))
    cv2.polylines(img, pts=pts, isClosed=True, color=(255, 255, 255), thickness=5)
    cv2.imshow(f"{char}", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
