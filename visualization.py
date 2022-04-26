import numpy as np
import cv2

IMG_NPZ_PATH = "npz/AnnoImgNumpy_0.5.npz"
LABEL_NPZ_PATH = "npz/AnnoLabelNumpy_0.5.npz"

img_npz_loader = np.load(IMG_NPZ_PATH)
label_npz_loader = np.load(LABEL_NPZ_PATH)
for key in img_npz_loader:
    print(key)
for key in label_npz_loader:
    print(key)

img_fnames = img_npz_loader["fnames"]
imgs = img_npz_loader["imgs"]

label_fnames = label_npz_loader["fnames"]
chars = label_npz_loader["chars"]
lndmrks = label_npz_loader["landmarks"]

for img, lndmrk in zip(imgs[:5], lndmrks[:5]):
    print(img.shape, lndmrk.shape)
    cv2.imshow("test", img)
    cv2.waitKey(0)
    cv2.destroyWindow("test")

    pts = np.reshape(a=lndmrk, newshape=(-1, 1, 2))
    cv2.polylines(img, pts=pts, isClosed=True, color=(255, 255, 255), thickness=5)
    cv2.imshow("test", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
