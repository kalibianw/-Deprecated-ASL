from utils import AnnoDataModule, AnnoVisualModule

from keras.api.keras import models
from sklearn.metrics import confusion_matrix

import numpy as np
import string
import cv2

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
    verbose=1,
    batch_size=32
)

cls_pred, lndmrk_pred = model.predict(
    x={"img": x_test},
    verbose=1,
    batch_size=32
)
lndmrk_pred = np.reshape(lndmrk_pred, newshape=(lndmrk_pred.shape[0], int(lndmrk_pred.shape[1] / 2), 2))
print(cls_pred.shape)
print(lndmrk_pred.shape)

avm = AnnoVisualModule(is_chars_normalized=True, is_lndmrks_normalized=False)
avm.draw_line(img=x_test[0], lndmrk=lndmrk_pred[0], color=(255, 255, 255))

cv2.imshow(f"{np.argmax(cls_pred[0])}", x_test[0])
cv2.waitKey(0)
cv2.destroyAllWindows()

conf_mat = confusion_matrix(y_true=np.argmax(y_cls_test, axis=1), y_pred=np.argmax(cls_pred, axis=1))
avm.plot_confusion_matrix(
    conf_mat,
    target_names=list(string.ascii_lowercase),
    normalize=False
)
