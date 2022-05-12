from utils import SegVisualModule

from keras.api.keras import models

import numpy as np

MODEL_PATH = "model/seg_0.5.h5"
NPZ_PATH = "npz/SegNumpy_0.5.npz"
BATCH_SIZE = 32

npz_loader = np.load(NPZ_PATH)

x_test, y_cls_test, y_seg_test = npz_loader["x_test"][:5], npz_loader["y_cls_test"][:5], npz_loader["y_seg_test"][:5]
print(x_test.shape, y_cls_test.shape, y_seg_test.shape)

model = models.load_model(MODEL_PATH)
model.summary()

model.evaluate(
    x={"img": x_test}, y={"seg_out": y_seg_test},
    batch_size=BATCH_SIZE
)

seg_pred = model.predict(
    x={"img": x_test},
    verbose=1,
    batch_size=BATCH_SIZE
)
print(seg_pred.shape)
print(np.max(seg_pred), np.min(seg_pred))

seg_pred = np.asarray(seg_pred, dtype=int)

svm = SegVisualModule(is_chars_normalized=False)
svm.show_output(imgs=x_test, chars=y_cls_test, seg_imgs=seg_pred)
