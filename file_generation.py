import os
import time
import pandas as pd
from imblearn.over_sampling import SMOTE

class StreamDataGenerator:
    def __init__(self, file_path = None, output_dir="stream_data", chunk_size=1000, drift_batch=20, drift_koef=1.25):
        self.file_path = file_path
        self.output_dir = output_dir
        self.chunk_size = chunk_size
        self.drift_batch = drift_batch
        self.drift_koef = drift_koef
        os.makedirs(self.output_dir, exist_ok=True)

    def read_and_prepare_data(self):
        # Read the data
        df = pd.read_csv(self.file_path)

        # Convert columns to numeric, forcing errors to NaN
        df = df.apply(pd.to_numeric, errors='coerce')

        # Drop rows with NaN values
        df = df.dropna()

        # Separate features and target
        X = df.drop(columns=['Class'])
        y = df['Class']

        # Apply SMOTE to balance the dataset
        smote = SMOTE(random_state=3445)
        X_resampled, y_resampled = smote.fit_resample(X, y)

        # Combine resampled features and target
        df_resampled = pd.DataFrame(X_resampled, columns=X.columns)
        df_resampled['Class'] = y_resampled

        return df_resampled

    def generate_files(self, data = None) -> None:
        if data is None:
            df_resampled = self.read_and_prepare_data()
        else:
            df_resampled = data.toPandas()
        # Split the resampled data into chunks
        for i, chunk in enumerate(df_resampled.groupby(df_resampled.index // self.chunk_size)):
            chunk_df = chunk[1]

            # Simulate data drift after a certain number of batches
            if i >= self.drift_batch:
                chunk_df['Amount'] *= self.drift_koef
                chunk_df['V11'] *= self.drift_koef
                chunk_df['V4'] *= self.drift_koef
                chunk_df['V2'] *= self.drift_koef

            # Save the chunk to a CSV file
            chunk_df.to_csv(f"{self.output_dir}/stream_data_{i}.csv", index=False)
            #time.sleep(0.1)
            print(f"Saved {i+1} files", end="\r")

        print("\nData generation completed.")