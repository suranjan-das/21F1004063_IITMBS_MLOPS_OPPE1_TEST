import os
import pandas as pd
from glob import glob
from datetime import datetime

# =========================
# CONFIGURATION
# =========================
DATA_DIR = "./data"
TRAIN_OUTPUT = "train.csv"
TEST_OUTPUT = "test.csv"
FEAST_FEATURES_OUTPUT = os.path.join(DATA_DIR, "stock_features.parquet")

ROLLING_WINDOW = "10min"   # past 10-minute rolling window
FUTURE_SHIFT = 5           # predict 5 minutes ahead
TEST_SIZE = 20             # last 20 samples per stock


# =========================
# FUNCTION: Process single stock file
# =========================
def process_stock_file(filepath):
    stock_name = os.path.basename(filepath).split("__")[0]
    print(f"\n🔹 Processing {stock_name} from {filepath}")

    # Load CSV
    df = pd.read_csv(filepath)

    # Convert timestamp and sort
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['stock_name'] = stock_name
    df = df.sort_values('timestamp').reset_index(drop=True)
    df.set_index('timestamp', inplace=True)

    # Fill missing data
    df.ffill(inplace=True)

    # Rolling features (10-minute lookback)
    df['rolling_avg_10'] = df['close'].rolling(window=ROLLING_WINDOW, min_periods=1).mean()
    df['volume_sum_10'] = df['volume'].rolling(window=ROLLING_WINDOW, min_periods=1).sum()

    # Drop NaN values from rolling calculations
    df.dropna(subset=['rolling_avg_10', 'volume_sum_10'], inplace=True)

    # Create target: 1 if price goes up after 5 mins, else 0
    df['close_5min_future'] = df['close'].shift(-FUTURE_SHIFT)
    df['target'] = (df['close_5min_future'] > df['close']).astype(int)
    df.drop(columns=['close_5min_future'], inplace=True)

    # Drop rows where future data unavailable
    df = df.dropna(subset=['target']).copy()

    # Split into train/test
    test_df = df.tail(TEST_SIZE).copy()
    train_df = df.iloc[:-TEST_SIZE].copy()

    print(f"   -> Train: {train_df.shape}, Test: {test_df.shape}")
    return train_df, test_df


# =========================
# MAIN PIPELINE
# =========================
if __name__ == "__main__":
    all_files = glob(os.path.join(DATA_DIR, "*.csv"))
    if not all_files:
        raise FileNotFoundError(f"No CSV files found in {DATA_DIR}")

    all_train, all_test, all_features = [], [], []

    for file in all_files:
        train_df, test_df = process_stock_file(file)

        all_train.append(train_df)
        all_test.append(test_df)

        # Prepare feature data for Feast
        feat_df = train_df.reset_index()[[
            'timestamp', 'stock_name', 'rolling_avg_10', 'volume_sum_10'
        ]]
        feat_df['created_timestamp'] = datetime.utcnow()
        all_features.append(feat_df)

    # Merge all stock data
    merged_train = pd.concat(all_train, ignore_index=True)
    merged_test = pd.concat(all_test, ignore_index=True)
    merged_features = pd.concat(all_features, ignore_index=True)

    # Save outputs
    merged_train.to_csv(TRAIN_OUTPUT, index=False)
    merged_test.to_csv(TEST_OUTPUT, index=False)
    merged_features.to_parquet(FEAST_FEATURES_OUTPUT, index=False)

    print("\n===============================")
    print("✅ Data processing complete!")
    print(f"Train data saved to: {TRAIN_OUTPUT} ({len(merged_train)} rows)")
    print(f"Test data saved to: {TEST_OUTPUT} ({len(merged_test)} rows)")
    print(f"Feast features saved to: {FEAST_FEATURES_OUTPUT} ({len(merged_features)} rows)")
    print("===============================")
    print("Columns in train:", merged_train.columns.tolist())
    print("Stocks processed:", merged_train['stock_name'].unique())
