import zipfile
import pandas as pd
import os

dataset_dir = "./dataset"
files = ["SpamAssasin.csv.zip", "phishing_email.csv.zip"]

for file in files:
    filepath = os.path.join(dataset_dir, file)
    if os.path.exists(filepath):
        print(f"\n{'='*50}")
        print(f"Checking {file}")
        print('='*50)
        
        with zipfile.ZipFile(filepath, 'r') as zip_ref:
            print(f"Files in archive: {zip_ref.namelist()}")
            
            # Read first file
            csv_file = zip_ref.namelist()[0]
            with zip_ref.open(csv_file) as f:
                df = pd.read_csv(f, nrows=5)
                print(f"\nColumns: {list(df.columns)}")
                print(f"Shape: {df.shape}")
                print("\nFirst few rows:")
                print(df)