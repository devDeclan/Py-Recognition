import os
import pandas as pd
import argparse
import lib.config as config
from tqdm import tqdm

if __name__ == '__main__':

  labeled_images = []
  for (root, dirs, files) in os.walk(config.TRAIN_FRAMES_ROOT, topdown=True):
    if len(files) > 0:
      for img in tqdm(range(len(files))):
        image_path = os.path.join(root, files[img])
        label = root.split('/')[-2]
        labeled_images.append([image_path, label])
  df = pd.DataFrame(labeled_images, columns=['image', 'label'])

  df.to_csv(config.TRAIN_DATA, encoding='utf-8', header=True, index=False)

  labeled_images = []
  for (root, dirs, files) in os.walk(config.TEST_FRAMES_ROOT, topdown=True):
    print(files)
    if len(files) > 0:
      for img in tqdm(range(len(files))):
        image_path = os.path.join(root, files[img])
        label = root.split('/')[-2]
        labeled_images.append([image_path, label])
  df = pd.DataFrame(labeled_images, columns=['image', 'label'])
  print(df)

  df.to_csv(config.TEST_DATA, encoding='utf-8', index=False)

