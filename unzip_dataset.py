# unzip_dataset.py
import zipfile
import os

# Define the relative path to the archive.zip file
zip_file_path = 'archive.zip'

# Specify the directory to extract to (optional)
extract_dir = 'extracted_data'

# Create the directory if it doesn't exist
os.makedirs(extract_dir, exist_ok=True)

# Unzip the dataset
with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
    zip_ref.extractall(extract_dir)

print(f"Dataset extracted to {extract_dir}/")
