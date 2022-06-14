from __future__ import annotations
import pandas as pd
from sklearn import preprocessing
import os
from pathlib import Path
from sklearn.model_selection import train_test_split
import shutil

annotations = pd.read_csv('./annotations.csv', sep=' ', header=None) 

le = preprocessing.LabelEncoder()
all_labels = annotations[1]
le.fit(all_labels)
all_labels = le.transform(all_labels)

paths = pd.read_csv('./annotations.csv', sep=' ', header=None)[0]
paths = paths.apply(lambda x: './images/'+'/'.join(os.path.normpath(x).split(os.sep)[-3:]).replace('*','_'))  

path_train, path_val, y_train, y_val = train_test_split(paths, all_labels, test_size=0.33, random_state=42)

def save(paths, labels, root, name, ext):
    for path, label in zip(paths, labels):
        if os.path.exists(path):
            savepath = root + str(label) + "/"
            Path(savepath).mkdir(parents=True, exist_ok=True)
            shutil.copy(path, savepath + name + str(len(os.listdir(savepath))) + ext)
            """
            img = io.imread(path)
            io.imsave(savepath + name + str(len(os.listdir(savepath))) + ".png", img)
            """

save(path_train, y_train, './train/', 'train', ".png")
save(path_val, y_val, './valid/', 'valid', ".png")
save([p.replace("images", "poses")[:-4]+"_keypoints.json" for p in path_train], y_train, './train_poses/', 'train', ".json")
save([p.replace("images", "poses")[:-4]+"_keypoints.json" for p in path_val], y_val, './valid_poses/', 'valid', ".json")