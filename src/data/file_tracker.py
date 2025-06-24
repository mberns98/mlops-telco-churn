import pandas as pd
import hashlib
import os

def hash_file(path):
    """
    Generate a unique MD5 hash for the file at the given path.

    Args:
        path (str): Path to the file.

    Returns:
        str: MD5 hash string representing the file contents.
    """
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()

def already_processed(path, tracker_path="data/file_tracker.csv"):
    """
    Check whether a file has already been processed, based on its hash.

    Args:
        path (str): Path to the file to check.
        tracker_path (str): Path to the file tracker CSV that logs processed files.

    Returns:
        bool: True if the file was already processed, False otherwise.
    """
    if not os.path.exists(tracker_path):
        return False
    df = pd.read_csv(tracker_path)
    file_hash = hash_file(path)
    return file_hash in df["file_hash"].values

def update_file_tracker(path, tracker_path="data/file_tracker.csv"):
    """
    Add a new entry to the file tracker for a processed file.

    Args:
        path (str): Path to the file that was processed.
        tracker_path (str): Path to the file tracker CSV.
    """
    file_hash = hash_file(path)

    # Build a new record with file metadata
    record = {
        "filename": os.path.basename(path), # Only keep the file name, not the full path
        "file_hash": file_hash,
        "timestamp": pd.Timestamp.now()  # Add current date
    }
    df_new = pd.DataFrame([record])

    # If tracker exists, append to it; otherwise, create new
    if os.path.exists(tracker_path):
        df_old = pd.read_csv(tracker_path)
        df = pd.concat([df_old, df_new], ignore_index=True)
    else:
        df = df_new

    # Save the updated tracker
    df.to_csv(tracker_path, index=False)

# Ejemplo de uso:
#if not already_processed(path):
#    process_file(path)
#    update_file_tracker(path)
#else:
#    print("Ya fue procesado")