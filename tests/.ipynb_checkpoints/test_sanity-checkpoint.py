# tests/test_sanity.py
import pandas as pd

def test_no_missing_values():
    df = pd.read_csv("test.csv")
    assert df.isnull().sum().sum() == 0, "❌ Test data contains missing values!"

def test_target_binary():
    df = pd.read_csv("test.csv")
    assert set(df["target"].unique()).issubset({0, 1}), "❌ Target column is not binary!"

def test_feature_ranges():
    df = pd.read_csv("test.csv")
    assert df["rolling_avg_10"].between(0, 1e6).all(), "❌ rolling_avg_10 out of expected range!"
    assert df["volume_sum_10"].ge(0).all(), "❌ volume_sum_10 has negative values!"
