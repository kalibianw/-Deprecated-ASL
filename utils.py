from tqdm import tqdm

import pandas as pd
import numpy as np
import cv2
import os


class AnnoDataModule:
    def __init__(self, dataset_dir_path):
        self.DATASET_DIR_PATH = dataset_dir_path
        print(f"Find {len(os.listdir(self.DATASET_DIR_PATH))} class(es).")
        print(os.listdir(self.DATASET_DIR_PATH))

    def img_to_np(self, rescaling_ratio=1):
        fnames = list()
        imgs = list()
        for label_name in tqdm(os.listdir(self.DATASET_DIR_PATH), desc="img_to_np"):
            for img_name in tqdm(os.listdir(f"{self.DATASET_DIR_PATH}/{label_name}/lit/")):
                fname = f"{self.DATASET_DIR_PATH}/{label_name}/lit/{img_name}"
                img = cv2.imread(filename=fname)
                img = cv2.resize(src=img, dsize=(0, 0), fx=rescaling_ratio, fy=rescaling_ratio)
                img = cv2.cvtColor(img, code=cv2.COLOR_BGR2RGB)

                imgs.append(img)
                fnames.append(fname)

        return np.array(fnames), np.array(imgs)

    def label_to_np(self, rescaling_ratio=1):
        fnames = list()
        chars = list()
        landmarks = list()
        for label_name in tqdm(os.listdir(self.DATASET_DIR_PATH), desc="label_to_np"):
            for json_fname in tqdm(os.listdir(f"{self.DATASET_DIR_PATH}/{label_name}/annotation/")):
                fname = f"{self.DATASET_DIR_PATH}/{label_name}/annotation/{json_fname}"
                pd_json = pd.read_json(path_or_buf=fname)
                raw_landmark = pd_json["Landmarks"].to_numpy()
                landmark = list()
                for mark in raw_landmark:
                    landmark.append([int(mark[0] * rescaling_ratio), int(mark[1] * rescaling_ratio)])

                fnames.append(fname)
                chars.append(label_name)
                landmarks.append(landmark)

        return np.array(fnames), np.array(chars), np.array(landmarks)


class AnnoTrainModule:
    def __init__(self, input_shape):
        self.INPUT_SHAPE = input_shape
