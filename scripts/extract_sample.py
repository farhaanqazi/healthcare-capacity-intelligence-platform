import pandas as pd
from pathlib import Path

def extract_sample():
    print("Loading full dataset...")
    df = pd.read_parquet("data/processed/rtt_features.parquet")
    
    print(f"Original shape: {df.shape}")
    
    # To preserve the time-series nature and ensure we have enough data to pass the
    # walk-forward and AUC tests, we will select 25 random large providers that
    # likely have data for all months and specialties.
    providers = df["provider_code"].value_counts().head(25).index.tolist()
    
    sample_df = df[df["provider_code"].isin(providers)].copy()
    
    print(f"Sample shape: {sample_df.shape}")
    
    # Ensure tests/fixtures directory exists
    fixtures_dir = Path("tests/fixtures")
    fixtures_dir.mkdir(parents=True, exist_ok=True)
    
    sample_path = fixtures_dir / "sample_features.parquet"
    sample_df.to_parquet(sample_path, index=False)
    
    file_size_mb = sample_path.stat().st_size / (1024 * 1024)
    print(f"Successfully saved sample to {sample_path} ({file_size_mb:.2f} MB)")

if __name__ == "__main__":
    extract_sample()
