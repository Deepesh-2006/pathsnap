from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = Path("/Users/deepeshm/Downloads/Student folder")
TRAIN_DIR = DATA_DIR / "train"
TEST_DIR = DATA_DIR / "student_test"

ARTIFACTS_DIR = BASE_DIR / "artifacts"
OUTPUTS_DIR = BASE_DIR / "outputs"
PLOTS_DIR = OUTPUTS_DIR / "plots"
MAPS_DIR = OUTPUTS_DIR / "maps"

for d in (ARTIFACTS_DIR, OUTPUTS_DIR, PLOTS_DIR, MAPS_DIR):
    d.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42
TOP_K_CANDIDATES = 10
CANDIDATE_RADIUS_M = 120.0
