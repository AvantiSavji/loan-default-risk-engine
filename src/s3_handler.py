import os
import shutil
import joblib
import pandas as pd
import io
from dotenv import load_dotenv

load_dotenv()

# ─────────────────────────────────────────
# LOCAL STORAGE SETUP
# Simulates S3 bucket structure locally
# Replace get_s3_client() with real boto3
# client when AWS credentials are ready
# ─────────────────────────────────────────

LOCAL_BUCKET_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    's3_local_bucket'
)

def _ensure_dir(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)

def _local_path(s3_key):
    return os.path.join(LOCAL_BUCKET_PATH, s3_key)


# ─────────────────────────────────────────
# UPLOAD FUNCTIONS
# ─────────────────────────────────────────

def upload_model(local_path, s3_key):
    """
    Upload a trained model file to local S3 simulation.
    In production: replace with s3.upload_file()
    """
    dest = _local_path(s3_key)
    _ensure_dir(dest)
    shutil.copy2(local_path, dest)
    print(f"Uploaded model    : {local_path}")
    print(f"Destination       : s3_local_bucket/{s3_key}")


def upload_dataframe(df, s3_key):
    """
    Upload a pandas DataFrame as CSV to local S3 simulation.
    In production: replace with s3.put_object()
    """
    dest = _local_path(s3_key)
    _ensure_dir(dest)
    df.to_csv(dest, index=False)
    print(f"Uploaded dataframe → s3_local_bucket/{s3_key}")


def upload_file(local_path, s3_key):
    """
    Upload any file to local S3 simulation.
    In production: replace with s3.upload_file()
    """
    dest = _local_path(s3_key)
    _ensure_dir(dest)
    shutil.copy2(local_path, dest)
    print(f"Uploaded file     : {os.path.basename(local_path)}")
    print(f"Destination       : s3_local_bucket/{s3_key}")


# ─────────────────────────────────────────
# DOWNLOAD FUNCTIONS
# ─────────────────────────────────────────

def download_model(s3_key, local_path):
    """
    Download a model from local S3 simulation.
    In production: replace with s3.download_file()
    """
    src = _local_path(s3_key)
    shutil.copy2(src, local_path)
    model = joblib.load(local_path)
    print(f"Downloaded model  : s3_local_bucket/{s3_key}")
    return model


def download_dataframe(s3_key):
    """
    Download a CSV from local S3 simulation into DataFrame.
    In production: replace with s3.get_object()
    """
    src = _local_path(s3_key)
    df  = pd.read_csv(src)
    print(f"Downloaded dataframe : s3_local_bucket/{s3_key}")
    return df


# ─────────────────────────────────────────
# UTILITY FUNCTIONS
# ─────────────────────────────────────────

def list_bucket_contents(prefix=''):
    """
    List all files in local S3 simulation.
    In production: replace with s3.list_objects_v2()
    """
    search_path = os.path.join(LOCAL_BUCKET_PATH, prefix)

    if not os.path.exists(search_path):
        print(f"No files found in s3_local_bucket/{prefix}")
        return []

    files = []
    print(f"\nFiles in s3_local_bucket/{prefix}")
    print("-" * 55)

    for root, dirs, filenames in os.walk(search_path):
        for filename in filenames:
            full_path   = os.path.join(root, filename)
            relative    = os.path.relpath(full_path, LOCAL_BUCKET_PATH)
            size_kb     = os.path.getsize(full_path) / 1024
            print(f"  {relative:<45} {size_kb:.1f} KB")
            files.append(relative)

    print(f"\nTotal files: {len(files)}")
    return files


def check_connection():
    """
    Verify local S3 simulation is accessible.
    In production: replace with s3.head_bucket()
    """
    try:
        os.makedirs(LOCAL_BUCKET_PATH, exist_ok=True)
        print("Storage Connection : SUCCESS (Local Simulation)")
        print(f"Bucket Path        : s3_local_bucket/")
        print("Note               : Switch to real AWS S3 by")
        print("                     adding credentials to .env")
        return True
    except Exception as e:
        print(f"Storage Connection : FAILED")
        print(f"Error              : {e}")
        return False