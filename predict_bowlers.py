import pandas as pd
import joblib

# ============================
# LOAD MODEL BUNDLE
# ============================
bundle = joblib.load("bowler_model.pkl")
model = bundle["model"]
feature_columns = bundle["feature_columns"]

# ============================
# LOAD DATA
# ============================
df = pd.read_csv("Book2.csv")
df.columns = df.columns.str.strip()
df = df.dropna(how="all").copy()

# Make sure numeric cols are numeric (same cleaning style as training)
for col in ["wickets", "runs", "economy", "did_bowler_do_well"]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

df["wickets"] = df["wickets"].fillna(0)
df["runs"] = df["runs"].fillna(0)
df["economy"] = df["economy"].fillna(0)
df["did_bowler_do_well"] = df["did_bowler_do_well"].fillna(0).astype(int)


# ============================
# ROLE DETECTION
# ============================
def get_bowling_role(bowler_type: str) -> str:
    bt = str(bowler_type).lower()
    if any(k in bt for k in ["legbreak", "offbreak", "orthodox"]):
        return "spin"
    return "pace"


# ============================
# REGION DETECTION
# ============================
def detect_region(ground: str) -> str:
    g = str(ground).lower()

    if any(k in g for k in [
        "oval", "lords", "headingley", "manchester",
        "leeds", "birmingham", "london"
    ]):
        return "ENG"

    if any(k in g for k in [
        "premadasa", "pallekele", "galle", "chennai",
        "delhi", "dhaka", "karachi", "lahore", "colombo"
    ]):
        return "ASIA"

    return "NEUTRAL"


# ============================
# APPLY ROLE CONSTRAINTS (International mode only)
# ============================
def apply_role_constraints(ranked_df, region, final_count=4):
    spinners = ranked_df[ranked_df["role"] == "spin"]
    pacers = ranked_df[ranked_df["role"] == "pace"]

    selected = []

    if region == "ENG":
        selected += pacers.head(2).to_dict("records")
        remaining = ranked_df.drop(pacers.head(2).index)

    elif region == "ASIA":
        selected += spinners.head(2).to_dict("records")
        remaining = ranked_df.drop(spinners.head(2).index)

    else:
        remaining = ranked_df

    needed = final_count - len(selected)
    selected += remaining.head(needed).to_dict("records")

    return selected


# ============================
# PITCH SUMMARY
# ============================
def summarise_pitch_from_results(results):
    spin = sum(1 for r in results if get_bowling_role(r["bowler_type"]) == "spin")
    pace = len(results) - spin

    if spin > pace:
        return f"Spin-friendly conditions detected ({spin} spinners vs {pace} pacers)."
    elif pace > spin:
        return f"Pace-friendly conditions detected ({pace} pacers vs {spin} spinners)."
    return f"Balanced pitch conditions ({spin} spin, {pace} pace)."


# ============================
# EXPLANATION ENGINE
# ============================
def explain_score(name, bowler_type, score, pitch_type, weather, opposition_team):
    text = f"{name} ({bowler_type}) has a model score of {round(float(score), 3)}. "

    if score >= 0.7:
        text += "This is a strong confidence level. "
    elif score >= 0.5:
        text += "This is a moderate confidence level. "
    else:
        text += "This is a lower confidence level, but still competitive. "

    bt = str(bowler_type).lower()
    pt = str(pitch_type).lower()
    wt = str(weather).lower()

    if any(k in bt for k in ["legbreak", "offbreak", "orthodox"]):
        if pt in ["spinning", "slow", "dry"]:
            text += "Spin-friendly pitch conditions suit this bowler. "
    else:
        if pt == "green":
            text += "Green pitches support seam movement for fast bowlers. "
        if wt in ["cloudy", "overcast"]:
            text += "Overcast weather can assist swing bowling. "

    text += (
        f"This prediction is based on patterns learned from past matches under "
        f"similar conditions against opponents like {opposition_team}."
    )
    return text


# ============================
# FEATURE IMPORTANCE 
# ============================
def get_feature_importance_text(model, feature_columns, top_n=12):
    importances = getattr(model, "feature_importances_", None)
    if importances is None:
        return "Feature importance not available for this model."

    fi = pd.DataFrame({
        "feature": feature_columns,
        "importance": importances
    })

    fi = fi.sort_values("importance", ascending=False).head(top_n)

    lines = [f"MODEL FEATURE IMPORTANCE (Top {top_n}):"]
    for i, row in enumerate(fi.itertuples(index=False), 1):
        lines.append(f"{i}. {row.feature} → {round(float(row.importance), 4)}")

    return "\n".join(lines)


# ==========================================================
# STEP 4 — INTERNATIONAL MODE (Top 4 Bowlers)
# ==========================================================
def predict_best_bowlers(
    match_type,
    pitch_type,
    weather,
    ground,
    your_team,
    opposition_team,
    return_all=False,
    show_importance=False,
):
    # Filter dataset to only bowlers from the selected team
    team_df = df[df["your_team"] == your_team].copy()
    if team_df.empty:
        raise ValueError(f"No bowlers found for team '{your_team}'")

    # Replace match conditions with the NEW input conditions
    team_df["match_type"] = match_type
    team_df["pitch_type"] = pitch_type
    team_df["weather"] = weather
    team_df["ground"] = ground
    team_df["opposition_team"] = opposition_team

    # Define which columns are categorical and numeric (same as training)
    categorical_cols = [
        "match_type",
        "pitch_type",
        "weather",
        "ground",
        "your_team",
        "opposition_team",
        "bowler_type",
    ]
    numeric_cols = ["wickets", "runs", "economy"]

    # One-hot encode categorical features
    X_cat = pd.get_dummies(team_df[categorical_cols], drop_first=False)

    # Keep numeric features as they are
    X_num = team_df[numeric_cols]

    # Combine categorical and numeric features
    X = pd.concat([X_cat, X_num], axis=1)

    # Align columns with training-time feature list (very important for ML consistency)
    X = X.reindex(columns=feature_columns, fill_value=0)

    # Get probability that each bowler will perform well (class 1)
    probs = model.predict_proba(X)[:, 1]
    team_df["score"] = probs

    # Detect whether each bowler is spin or pace
    team_df["role"] = team_df["bowler_type"].apply(get_bowling_role)

    # Average scores per bowler and rank them from highest to lowest
    ranked = (
        team_df
        .groupby(["bowler_name", "bowler_type", "role"])["score"]
        .mean()
        .reset_index()
        .sort_values("score", ascending=False)
        .reset_index(drop=True)
    )

    # Detect region of the ground (Asia, England, etc.)
    region = detect_region(ground)

    # Apply cricket-based role constraints (e.g., more pacers in England)
    selected = apply_role_constraints(ranked, region)

    # Build final output list and explanations
    results, explanations = [], []
    for r in selected:
        results.append({
            "bowler_name": r["bowler_name"],
            "bowler_type": r["bowler_type"],
            "score": round(float(r["score"]), 3),
        })
        explanations.append(
            explain_score(
                r["bowler_name"],
                r["bowler_type"],
                r["score"],
                pitch_type,
                weather,
                opposition_team,
            )
        )

    # Generate pitch summary (spin-friendly, pace-friendly, etc.)
    pitch_msg = summarise_pitch_from_results(results)

    # Optionally include feature importance explanation
    fi_text = get_feature_importance_text(model, feature_columns) if show_importance else None

    # Return extra details if requested
    if return_all:
        return results, explanations, pitch_msg, ranked, fi_text

    # Default return: top bowlers + explanations + pitch summary
    return results, explanations, pitch_msg, fi_text


# ==========================================================
# STEP 5 — LOCAL MODE (Top 4 Bowler TYPES)
# Inputs: match_type, pitch_type, weather
# Output: Top 4 types + explanations + pitch summary
# ==========================================================
def predict_best_bowler_types_local(
    match_type,
    pitch_type,
    weather,
    return_all=False,
    show_importance=False,
):
    # Candidate types: use what's in the dataset (works even if you add new types later)
    candidate_types = sorted(df["bowler_type"].dropna().unique().tolist())
    if not candidate_types:
        raise ValueError("No bowler_type values found in dataset.")

    # Neutral numeric defaults (so local mode doesn't depend on a specific player's past wickets/runs)
    wickets_default = float(df["wickets"].median())
    runs_default = float(df["runs"].median())
    economy_default = float(df["economy"].median())

    # Build one row per bowler type
    local_rows = []
    for bt in candidate_types:
        local_rows.append({
            "match_type": match_type,
            "pitch_type": pitch_type,
            "weather": weather,
            # Local mode ignores these, but model expects them as features:
            "ground": "LOCAL",
            "your_team": "LOCAL",
            "opposition_team": "LOCAL",
            "bowler_type": bt,
            "wickets": wickets_default,
            "runs": runs_default,
            "economy": economy_default,
        })

    local_df = pd.DataFrame(local_rows)

    categorical_cols = [
        "match_type",
        "pitch_type",
        "weather",
        "ground",
        "your_team",
        "opposition_team",
        "bowler_type",
    ]
    numeric_cols = ["wickets", "runs", "economy"]

    X_cat = pd.get_dummies(local_df[categorical_cols], drop_first=False)
    X_num = local_df[numeric_cols]

    X = pd.concat([X_cat, X_num], axis=1)
    X = X.reindex(columns=feature_columns, fill_value=0)

    probs = model.predict_proba(X)[:, 1]
    local_df["score"] = probs
    local_df["role"] = local_df["bowler_type"].apply(get_bowling_role)

    ranked_types = (
        local_df
        .groupby(["bowler_type", "role"])["score"]
        .mean()
        .reset_index()
        .sort_values("score", ascending=False)
        .reset_index(drop=True)
    )

    top = ranked_types.head(4).to_dict("records")

    results, explanations = [], []
    for r in top:
        results.append({
            "bowler_type": r["bowler_type"],
            "score": round(float(r["score"]), 3),
        })
        # Use the same explanation engine (name will be the type label)
        explanations.append(
            explain_score(
                name=r["bowler_type"],
                bowler_type=r["bowler_type"],
                score=r["score"],
                pitch_type=pitch_type,
                weather=weather,
                opposition_team="local opponents",
            )
        )

    # Pitch summary uses bowler_type role detection
    pitch_msg = summarise_pitch_from_results([{"bowler_type": r["bowler_type"]} for r in results])
    fi_text = get_feature_importance_text(model, feature_columns) if show_importance else None

    if return_all:
        return results, explanations, pitch_msg, ranked_types, fi_text

    return results, explanations, pitch_msg, fi_text


# ============================
# DEMO
# ============================
if __name__ == "__main__":
    # -------- International demo --------
    bowlers, explanations, pitch_msg, all_rankings, fi_text = predict_best_bowlers(
        match_type="ODI",
        pitch_type="Spinning",
        weather="Sunny",
        ground="Shere Bangla Stadium",
        your_team="Sri Lanka",
        opposition_team="Bangladesh",
        return_all=True,
        show_importance=True,
    )

    print("\nTOP 4 BOWLERS (International):")
    for b, e in zip(bowlers, explanations):
        print(f"- {b['bowler_name']} ({b['bowler_type']}), score={b['score']}")
        print("  Explanation:", e)
        print()

    print("PITCH SUMMARY (International):")
    print(pitch_msg)

    print("\nALL BOWLERS RATING (International):")
    display_df = all_rankings.copy()
    display_df["score"] = display_df["score"].round(3)
    display_df.insert(0, "rank", range(1, len(display_df) + 1))
    print(display_df.to_string(index=False))

    print("\nFEATURE IMPORTANCE:")
    print(fi_text)

    # -------- Local demo --------
    local_types, local_explanations, local_pitch_msg, all_types_ranked, local_fi = predict_best_bowler_types_local(
        match_type="T20",
        pitch_type="Green",
        weather="Overcast",
        return_all=True,
        show_importance=False,
    )

    print("\nTOP 4 BOWLER TYPES (Local):")
    for t, e in zip(local_types, local_explanations):
        print(f"- {t['bowler_type']}, score={t['score']}")
        print("  Explanation:", e)
        print()

    print("PITCH SUMMARY (Local):")
    print(local_pitch_msg)

    print("\nALL TYPES RATING (Local):")
    show_types = all_types_ranked.copy()
    show_types["score"] = show_types["score"].round(3)
    show_types.insert(0, "rank", range(1, len(show_types) + 1))
    print(show_types.to_string(index=False))

