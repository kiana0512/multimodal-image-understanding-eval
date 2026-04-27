"""Dataset splitting utilities."""
from __future__ import annotations
import pandas as pd
from sklearn.model_selection import train_test_split

def stratified_split(df: pd.DataFrame, label_column: str = "label", val_size: float = 0.2, seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create train/val splits, stratified when possible."""
    stratify=df[label_column] if label_column in df.columns and df[label_column].nunique()>1 else None
    train_df,val_df=train_test_split(df,test_size=val_size,random_state=seed,stratify=stratify)
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True)
