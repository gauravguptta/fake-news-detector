import pandas as pd
import os

def merge_and_save(true_path="data/True.csv", fake_path="data/Fake.csv", output_path="data/train.csv"):
    """Merge True/Fake news datasets into one CSV with labels."""

    # Load datasets
    df_true = pd.read_csv(true_path)
    df_fake = pd.read_csv(fake_path)

    # Assign labels: 1 = Real, 0 = Fake
    df_true["label"] = 1
    df_fake["label"] = 0

    # Combine & shuffle
    combined = pd.concat([df_true, df_fake], ignore_index=True)
    combined = combined.sample(frac=1, random_state=42).reset_index(drop=True)

    # Ensure output folder exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save merged dataset
    combined.to_csv(output_path, index=False)
    print(f"✅ Merged dataset saved at: {output_path}")
    print(combined.head())

if __name__ == "__main__":
    merge_and_save()
