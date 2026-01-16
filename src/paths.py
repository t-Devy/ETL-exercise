from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)

TRANSFORM_DIR = PROJECT_ROOT / "artifacts" / "transform_artifacts"
TRANSFORM_DIR.mkdir(parents=True, exist_ok=True)

LOADER_DIR = PROJECT_ROOT / "artifacts" / "loader_artifacts"
LOADER_DIR.mkdir(parents=True, exist_ok=True)

# -- data outputs
RAW_CSV_PATH = DATA_DIR / "ccr_mock.csv"
RAW_PKL_PATH = DATA_DIR / "ccr_raw.pkl"


# --- transform outputs
X_TRAIN_PATH = TRANSFORM_DIR / "X_train.npy"
X_VAL_PATH = TRANSFORM_DIR / "X_val.npy"
Y_TRAIN_PATH = TRANSFORM_DIR / "y_train.npy"
Y_VAL_PATH = TRANSFORM_DIR / "y_val.npy"
SCALER_PATH = TRANSFORM_DIR / "scaler.pkl"
FEATURES_PATH = TRANSFORM_DIR / "feature_columns.json"

# --- load outputs: confirm loader config shapes
TRAIN_LOADER_INFO_PATH = LOADER_DIR / "train_loader_info.json"
VAL_LOADER_INFO_PATH = LOADER_DIR / "val_loader_info.json"


