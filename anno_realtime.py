from utils import AnnoDataModule, AnnoVisualModule

from keras.api.keras import models
import numpy as np
import string
import cv2

FONT = cv2.FONT_HERSHEY_SIMPLEX
COLOR_CODE = (255, 255, 255)

adm = AnnoDataModule(dataset_dir_path=None)
avm = AnnoVisualModule(is_chars_normalized=True)

model = models.load_model("model/anno_0.5.h5")

cap = cv2.VideoCapture(0)
cap.set(3, 320)
cap.set(4, 240)
while True:
    ret, frame = cap.read()
    frame = np.expand_dims(frame, axis=0)
    if ret:
        cls_pred, lndmrk_pred = model.predict(
            x={"img": frame},
            verbose=1,
        )
        lndmrk_pred = np.reshape(lndmrk_pred, newshape=(lndmrk_pred.shape[0], int(lndmrk_pred.shape[1] / 2), 2))

        avm.draw_line(frame[0], lndmrk_pred[0], color=COLOR_CODE)
        if cls_pred[0][np.argmax(cls_pred[0])] > 0.8:
            cv2.putText(frame[0], f"{string.ascii_lowercase[np.argmax(cls_pred[0])]} - {cls_pred[0][np.argmax(cls_pred[0])]}", (0, 50), fontFace=FONT, fontScale=2, color=COLOR_CODE)
        cv2.imshow(f"test", frame[0])
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        raise Exception("Hardware error occurred")

cap.release()
cv2.destroyAllWindows()
