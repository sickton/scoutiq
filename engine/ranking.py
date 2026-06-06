import numpy as np
import pandas as pd

def rank_candidates(target_name, similarity_df, full_df):
    target_row = full_df[full_df["Player"] == target_name].iloc[0]
    target_wage = target_row["Weekly Wages"]
    target_age = target_row["Age"]

    max_minutes = full_df["Min_x"].max()

    candidates = similarity_df.copy()
    candidates = candidates.merge(
        full_df[["Player", "Weekly Wages", "Age", "Min_x"]],
        left_on="player", right_on="Player", how="left"
    ).drop(columns="Player")

    candidates["perf_score"] = candidates["similarity"]
    
    candidates["wage_score"] = np.where(
        candidates["Weekly Wages"].isna() | np.isnan(target_wage),
        np.nan,
        (1 - candidates["Weekly Wages"] / target_wage).clip(0, 1)
    )

    candidates["age_score"] = (
        1 - (candidates["Age"] - target_age).abs() / 10
    ).clip(0, 1)

    candidates["avail_score"] = (
        1 - candidates["Min_x"] / max_minutes
    ).clip(0, 1)

    weights = {
        "perf_score": 0.40,
        "wage_score": 0.25,
        "age_score": 0.20,
        "avail_score": 0.15,
    }

    def weighted_score(row):
        total_w = 0.0
        total_s = 0.0
        for col, w in weights.items():
            val = row[col]
            if not np.isnan(val):
                total_w += w
                total_s += w * val
        return total_s / total_w if total_w > 0 else np.nan

    candidates["final_score"] = candidates.apply(weighted_score, axis=1)

    cols = ["player", "perf_score", "wage_score", "age_score", "avail_score", "final_score"]
    return candidates[cols].sort_values("final_score", ascending=False).reset_index(drop=True)

if __name__ == "__main__":
    from data_loader import load_sheets, merge_data, parse_position, parse_wages
    from features import build_feature_matrix, scale_features
    from similarity import find_similar

    sheets = load_sheets()
    df = merge_data(*sheets)
    df = parse_position(df)
    df = parse_wages(df)

    X, names, positions = build_feature_matrix(df)
    X_scaled, scaler = scale_features(X)

    target = "Mohamed Salah"
    sim_df = find_similar(target, X_scaled, names, positions, top_n=10)
    ranked = rank_candidates(target, sim_df, df)

    pd.set_option("display.float_format", "{:.3f}".format)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 120)
    print(ranked.to_string(index=False))