import pandas as pd
from sklearn.preprocessing import StandardScaler

FEATURE_COLS = [
    "goals_per90", "assists_per90", "shots_per90", "TklW", "Int", "Min_x", "Age",
]

def build_feature_matrix(df):
    df = df.copy()
    df = df.rename(columns={
        "Gls_x": "goals_per90",
        "Ast": "assists_per90",
        "Sh": "shots_per90",
    })
    df = df[df["Min_x"] >= 90].copy()
    df = df.dropna(subset=FEATURE_COLS)
    names = df["Player"].reset_index(drop=True)
    feature_subset = df[FEATURE_COLS].reset_index(drop=True)
    return feature_subset, names

def scale_features(X):
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns = X.columns)
    return X_scaled, scaler

if __name__ == "__main__":
    from data_loader import load_sheets, merge_data, parse_wages, parse_position

    sheets = load_sheets()
    df = merge_data(*sheets)
    df = parse_wages(df)
    df = parse_position(df)

    X, names = build_feature_matrix(df)
    X_scaled, scaler = scale_features(X)

    print(f"Feature matrix: {X_scaled.shape}")
    print("\nMeans (should be ~0):")
    print(X_scaled.mean().round(4))
    print("\nStds (should be ~1):")
    print(X_scaled.std().round(4))