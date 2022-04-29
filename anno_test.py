from utils import AnnoDataModule

from tensorflow.keras import models

import numpy as np

MODEL_PATH = "model/anno_0.5.h5"
NPZ_PATH = "npz/AnnoNumpy_0.5.npz"

npz_loader = np.load(NPZ_PATH)
for key in npz_loader:
    print(key)

x_test, y_cls_test, y_lndmrk_test = npz_loader["x_test"], npz_loader["y_cls_test"], npz_loader["y_lndmrk_test"]
print(x_test.shape, y_cls_test.shape, y_lndmrk_test.shape)


adm = AnnoDataModule(dataset_dir_path=None, rescaling_ratio=0.5, img_height=x_test.shape[1], img_width=x_test.shape[2])
y_cls_test = adm.cls_normalization(chars=y_cls_test)
y_lndmrk_test = np.reshape(y_lndmrk_test, newshape=(y_lndmrk_test.shape[0], y_lndmrk_test.shape[1] * y_lndmrk_test.shape[2]))

model = models.load_model(MODEL_PATH)
model.summary()
model.evaluate(
    x={"img": x_test}, y={"cls_out": y_cls_test, "lndmrk_out": y_lndmrk_test},
    batch_size=32
)
