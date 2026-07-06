import os
from pathlib import Path

# Paths
PROJECT_ROOT = Path(os.environ.get("PROJECT_ROOT", "/home/ec2-user/SageMaker/multimodal-deepfake-detection"))
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
FACES_DIR = PROCESSED_DATA_DIR / "faces"
RESULTS_DIR = PROJECT_ROOT / "results"
METRICS_DIR = RESULTS_DIR / "metrics"
HEATMAPS_DIR = RESULTS_DIR / "heatmaps"
CHECKPOINTS_DIR = RESULTS_DIR / "checkpoints"

# Face extraction settings
FACE_SIZE = 299  # XceptionNet input size
FACE_MARGIN = 0.3  # margin around detected face
FRAMES_PER_VIDEO = 30  # sample this many frames per video
MAX_VIDEOS_PER_PART = None  # set to integer to limit (e.g., 100 for testing)
MIN_FACE_CONFIDENCE = 0.95

# Training settings
BATCH_SIZE = 32
NUM_EPOCHS = 20
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5
NUM_WORKERS = 4
TRAIN_SPLIT = 0.7
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15
RANDOM_SEED = 42
EARLY_STOPPING_PATIENCE = 5

# Model settings
MODEL_NAME = "xceptionnet"
NUM_CLASSES = 2
PRETRAINED = True

# Evaluation settings
NUM_HEATMAP_SAMPLES = 50
GRADCAM_TARGET_LAYER = "block12"

# Device
DEVICE = "cuda"
