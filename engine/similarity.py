import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

def find_similar(player_name, df_scaled, names, positions, top_n=10):
    idx = names[names == player_name].index[0]

    POS_GROUP = {
        "FW": "attacker",
        "MF": "midfielder",
        "DF": "defender",
        "GK": "goalkeeper",
    }

    target_group = POS_GROUP.get(positions.iloc[idx][0], "other")
    same_group = positions.apply(lambda p: POS_GROUP.get(p[0], "other") == target_group)

    filtered_scaled = df_scaled[same_group.values].reset_index(drop=True)
    filtered_names = names[same_group.values].reset_index(drop=True)

    player_vector = filtered_scaled[filtered_names == player_name].values.reshape(1, -1)

    score = cosine_similarity(player_vector, filtered_scaled.values).flatten()

    results = pd.DataFrame({"player": filtered_names, "similarity": score})
    results = results[results["player"] != player_name]
    results = results.sort_values("similarity", ascending=False).head(top_n)

    return results.reset_index(drop=True)

if __name__ == "__main__":
    from data_loader import load_sheets, merge_data, parse_position, parse_wages
    from features import build_feature_matrix, scale_features

    sheets = load_sheets()
    df = merge_data(*sheets)
    df = parse_position(df)
    df = parse_wages(df)

    X, names, positions = build_feature_matrix(df)
    X_scaled, scaler = scale_features(X)

    results = find_similar("Virgil van Dijk", X_scaled, names, positions, top_n=10)
    print(results)