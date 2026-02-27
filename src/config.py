"""
AI-Powered Network Intrusion Detection System (NIDS)
Project Configuration File

All paths, constants, and settings used across the project.
"""

from pathlib import Path
import os

# ============================================================
# DIRECTORY PATHS
# ============================================================

# Root project directory
ROOT_DIR = Path(r"C:\Capstone Project")

# Data directories
DATA_RAW = ROOT_DIR / "Dataset"
DATA_PROCESSED = ROOT_DIR / "processed_data"
MODELS_DIR = ROOT_DIR / "models"
REPORTS_DIR = ROOT_DIR / "reports"

# Create directories if they don't exist
for d in [DATA_PROCESSED, MODELS_DIR, REPORTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ============================================================
# DATASET FILES
# ============================================================

CSV_FILES = [
    "Monday-WorkingHours.pcap_ISCX.csv",
    "Tuesday-WorkingHours.pcap_ISCX.csv",
    "Wednesday-workingHours.pcap_ISCX.csv",
    "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
    "Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv",
    "Friday-WorkingHours-Morning.pcap_ISCX.csv",
    "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
    "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
]

# ============================================================
# ATTACK TYPES IN DATASET
# ============================================================

ATTACK_TYPES = [
    "BENIGN",
    "DoS Hulk",
    "PortScan",
    "DDoS",
    "DoS GoldenEye",
    "FTP-Patator",
    "SSH-Patator",
    "DoS slowloris",
    "DoS Slowhttptest",
    "Bot",
    "Web Attack \x96 Brute Force",
    "Web Attack \x96 XSS",
    "Infiltration",
    "Web Attack \x96 Sql Injection",
    "Heartbleed",
]

# ============================================================
# MODEL & TRAINING SETTINGS
# ============================================================

RANDOM_SEED = 42
TEST_SIZE = 0.15
VAL_SIZE = 0.15

# ============================================================
# VERIFY SETUP
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("AI-NIDS Configuration Verification")
    print("=" * 60)
    print(f"Root Directory:     {ROOT_DIR}")
    print(f"Dataset Directory:  {DATA_RAW}")
    print(f"Processed Data:     {DATA_PROCESSED}")
    print(f"Models Directory:   {MODELS_DIR}")
    print(f"Reports Directory:  {REPORTS_DIR}")
    print()

    print("Checking CSV files in Dataset folder:")
    found = 0
    for f in CSV_FILES:
        path = DATA_RAW / f
        if path.exists():
            size_mb = path.stat().st_size / (1024 * 1024)
            print(f"  ✅ {f} ({size_mb:.1f} MB)")
            found += 1
        else:
            print(f"  ❌ MISSING: {f}")

    print(f"\nFound {found}/{len(CSV_FILES)} files")
    print(f"Directories created: {DATA_PROCESSED.exists()}, {MODELS_DIR.exists()}, {REPORTS_DIR.exists()}")

    if found == len(CSV_FILES):
        print("\n✅ Setup verified — all files found!")
    else:
        print(f"\n❌ Missing {len(CSV_FILES) - found} files. Check your Dataset folder.")