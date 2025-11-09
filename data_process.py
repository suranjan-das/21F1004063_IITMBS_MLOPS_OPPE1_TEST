import os
import pandas as pd
from glob import glob

# =========================
# CONFIGURATION
# =========================
DATA_DIR = "./data"
TRAIN_OUTPUT = "train.csv"
TEST_OUTPUT = "test.csv"

ROLLING_WINDOW = "10min"   # past 10-minute rolling window
FUTURE_SHIFT = 5           # predict 5 minutes ahead
TEST_SIZE = 20             # last 20 samples for test


# =========================
# FUNCTION: Process single stock file
# =========================
def process_stock_file(filepath):
    stock_name = os.path.basename(filepath).split("__")[0]
    print(f"\nProcessing {stock_name} from file {filepath}")

    # Load CSV
    df = pd.read_csv(filepath)

    # Convert timestamp and sort
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['stock_name'] = stock_name
    df = df.sort_values('timestamp').reset_index(drop=True)
    df.set_index('timestamp', inplace=True)

    # Forward fill missing values
    df.ffill(inplace=True)

    # Rolling features (past 10 minutes)
    df['rolling_avg_10'] = df['close'].rolling(window=ROLLING_WINDOW, min_periods=1).mean()
    df['volume_sum_10'] = df['volume'].rolling(window=ROLLING_WINDOW, min_periods=1).sum()

    # Drop NaNs from rolling calculations
    df.dropna(subset=['rolling_avg_10', 'volume_sum_10'], inplace=True)

    # Create target: price up in next 5 minutes?
    df['close_5min_future'] = df['close'].shift(-FUTURE_SHIFT)
    df['target'] = (df['close_5min_future'] > df['close']).astype(int)
    df.drop(columns=['close_5min_future'], inplace=True)

    # Drop last rows where future value unavailable
    df = df.dropna(subset=['target']).copy()

    # Split into train/test
    test_df = df.tail(TEST_SIZE).copy()
    train_df = df.iloc[:-TEST_SIZE].copy()

    print(f"  -> Train shape: {train_df.shape}, Test shape: {test_df.shape}")
    return train_df, test_df


# =========================
# MAIN PIPELINE
# =========================
if __name__ == "__main__":
    all_files = glob(os.path.join(DATA_DIR, "*.csv"))
    if not all_files:
        raise FileNotFoundError(f"No CSV files found in {DATA_DIR}")

    train_dfs, test_dfs = [], []

    for file in all_files:
        train_df, test_df = process_stock_file(file)
        train_dfs.append(train_df)
        test_dfs.append(test_df)

    # Merge all stocks’ train/test data
    merged_train = pd.concat(train_dfs, ignore_index=True)
    merged_test = pd.concat(test_dfs, ignore_index=True)

    # Save processed data
    merged_train.to_csv(TRAIN_OUTPUT, index=False)
    merged_test.to_csv(TEST_OUTPUT, index=False)

    print("\n=========================")
    print("✅ Data processing complete!")
    print(f"Train data saved to: {TRAIN_OUTPUT} ({len(merged_train)} rows)")
    print(f"Test data saved to: {TEST_OUTPUT} ({len(merged_test)} rows)")
    print("=========================")
    print("Columns in train:", merged_train.columns.tolist())
    print("Stocks processed:", merged_train['stock_name'].unique())
