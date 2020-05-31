import os

DATASET_ROOT = "dataset"
RESOURCES_ROOT = "resources"
DATA_ROOT = "data"

TRAIN_ROOT = os.path.join(DATASET_ROOT, "train")
VALID_ROOT = os.path.join(DATASET_ROOT, "valid")
TEST_ROOT = os.path.join(DATASET_ROOT, "test")
TRAIN_FRAMES_ROOT = os.path.join(DATASET_ROOT, "train_frames")
VALID_FRAMES_ROOT = os.path.join(DATASET_ROOT, "valid_frames")
TEST_FRAMES_ROOT = os.path.join(DATASET_ROOT, "test_frames")

TRAIN_DATA = os.path.join(DATA_ROOT, "train.csv")
TEST_DATA = os.path.join(DATA_ROOT, "test.csv")

CATEGORIES_PATH = os.path.join(RESOURCES_ROOT, "categories.json")
CLASSES_PATH = os.path.join(RESOURCES_ROOT, "classes.json")
TRAIN_METADATA_PATH = os.path.join(RESOURCES_ROOT, "train.json")
VALID_METADATA_PATH = os.path.join(RESOURCES_ROOT, "valid.json")
TEST_METADATA_PATH = os.path.join(RESOURCES_ROOT, "test.json")
