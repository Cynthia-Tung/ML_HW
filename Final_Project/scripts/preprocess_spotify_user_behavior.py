from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = ROOT / "dataset"
USER_INPUT = DATASET_DIR / "spotify_user_behavior_realistic_50000_rows.csv"
SONGS_INPUT = DATASET_DIR / "spotify_songs.csv"
ANALYSIS_OUTPUT = DATASET_DIR / "spotify_user_behavior_preprocessed.csv"
MODEL_OUTPUT = DATASET_DIR / "spotify_user_behavior_model_ready.csv"
GENRE_STATS_OUTPUT = DATASET_DIR / "spotify_genre_reference_stats.csv"
POPULARITY_INDEX_OUTPUT = DATASET_DIR / "spotify_genre_popularity_index.csv"
SUMMARY_OUTPUT = ROOT / "preprocessing_summary.md"

AUDIO_FEATURES = [
    "danceability",
    "energy",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence",
    "tempo",
    "loudness",
]

USER_TO_SPOTIFY_GENRE = {
    "Pop": "pop",
    "Rock": "rock",
    "Latin": "latin",
    "R&B": "r&b",
    "Hip-Hop": "rap",
    "Electronic": "edm",
}


def yes_no_to_int(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower().map({"yes": 1, "no": 0})


def add_age_groups(df: pd.DataFrame) -> pd.DataFrame:
    bins = [0, 19, 29, 39, 49, 59, 200]
    labels = ["10-19", "20-29", "30-39", "40-49", "50-59", "60+"]
    df["age_group"] = pd.cut(df["age"], bins=bins, labels=labels, right=True)
    return df


def build_genre_reference(songs: pd.DataFrame) -> pd.DataFrame:
    songs = songs.copy()
    songs["track_album_release_date"] = pd.to_datetime(
        songs["track_album_release_date"], errors="coerce"
    )
    songs["release_year"] = songs["track_album_release_date"].dt.year
    songs["release_decade"] = (songs["release_year"] // 10 * 10).astype("Int64")

    aggregations = {
        "track_id": "count",
        "track_popularity": ["mean", "median"],
        "release_year": ["mean", "median", "min", "max"],
    }
    for feature in AUDIO_FEATURES:
        aggregations[feature] = ["mean", "std", "var"]

    stats = songs.groupby("playlist_genre", dropna=False).agg(aggregations)
    stats.columns = ["_".join(col).strip("_") for col in stats.columns.to_flat_index()]
    stats = stats.rename(
        columns={
            "track_id_count": "genre_track_count",
            "track_popularity_mean": "genre_popularity_mean",
            "track_popularity_median": "genre_popularity_median",
            "release_year_mean": "genre_release_year_mean",
            "release_year_median": "genre_release_year_median",
            "release_year_min": "genre_release_year_min",
            "release_year_max": "genre_release_year_max",
        }
    )

    variance_cols = [f"{feature}_var" for feature in AUDIO_FEATURES]
    stats["genre_audio_total_variance"] = stats[variance_cols].sum(axis=1)
    stats["genre_audio_bubble_index"] = 1 / (1 + stats["genre_audio_total_variance"])

    popularity_q1 = stats["genre_popularity_mean"].quantile(0.25)
    popularity_q3 = stats["genre_popularity_mean"].quantile(0.75)
    stats["genre_popularity_segment"] = "middle"
    stats.loc[stats["genre_popularity_mean"] >= popularity_q3, "genre_popularity_segment"] = (
        "mainstream"
    )
    stats.loc[stats["genre_popularity_mean"] <= popularity_q1, "genre_popularity_segment"] = (
        "niche"
    )

    stats = stats.reset_index().rename(columns={"playlist_genre": "mapped_spotify_genre"})
    return stats


def preprocess_user_behavior(users: pd.DataFrame, genre_stats: pd.DataFrame) -> pd.DataFrame:
    df = users.copy()
    df.columns = df.columns.str.strip()
    df = df.drop_duplicates(subset=["user_id"]).reset_index(drop=True)

    text_columns = [
        column
        for column in df.columns
        if pd.api.types.is_object_dtype(df[column]) or pd.api.types.is_string_dtype(df[column])
    ]
    for column in text_columns:
        df[column] = df[column].astype(str).str.strip()

    df["signup_date"] = pd.to_datetime(df["signup_date"], errors="coerce")
    reference_date = df["signup_date"].max()
    df["signup_year"] = df["signup_date"].dt.year
    df["signup_month"] = df["signup_date"].dt.month
    df["signup_quarter"] = df["signup_date"].dt.quarter
    df["tenure_days"] = (reference_date - df["signup_date"]).dt.days
    df["tenure_months"] = (df["tenure_days"] / 30.4375).round(1)

    df = add_age_groups(df)
    df["ad_interaction_flag"] = yes_no_to_int(df["ad_interaction"])
    df["ad_conversion_flag"] = yes_no_to_int(df["ad_conversion_to_subscription"])
    df["is_paid_subscription"] = (df["subscription_type"] != "Free").astype(int)
    df["is_active_subscription"] = (df["subscription_status"] == "Active").astype(int)

    df["mapped_spotify_genre"] = df["favorite_genre"].map(USER_TO_SPOTIFY_GENRE)
    df["genre_stats_available"] = df["mapped_spotify_genre"].notna().astype(int)
    df = df.merge(genre_stats, on="mapped_spotify_genre", how="left")
    df["genre_popularity_segment"] = df["genre_popularity_segment"].fillna("unmatched")

    return df


def build_popularity_index(genre_stats: pd.DataFrame) -> pd.DataFrame:
    index = genre_stats[
        [
            "mapped_spotify_genre",
            "genre_track_count",
            "genre_popularity_mean",
            "genre_popularity_median",
            "genre_popularity_segment",
        ]
    ].copy()

    score_min = index["genre_popularity_mean"].min()
    score_max = index["genre_popularity_mean"].max()
    score_range = score_max - score_min
    if score_range == 0:
        index["popularity_score_0_100"] = 100.0
    else:
        index["popularity_score_0_100"] = (
            (index["genre_popularity_mean"] - score_min) / score_range * 100
        )

    index["popularity_rank"] = index["genre_popularity_mean"].rank(
        method="dense", ascending=False
    ).astype(int)
    index["popularity_percentile"] = index["genre_popularity_mean"].rank(pct=True)
    index["listener_type"] = index["genre_popularity_segment"].map(
        {
            "mainstream": "Mainstreamer",
            "middle": "Balanced",
            "niche": "Discovery-oriented",
        }
    )
    index["listener_type_zh"] = index["genre_popularity_segment"].map(
        {
            "mainstream": "主流型用戶",
            "middle": "中間型用戶",
            "niche": "小眾挖掘者",
        }
    )
    index["indicator_rule"] = index["genre_popularity_segment"].map(
        {
            "mainstream": "Top 25% by genre_popularity_mean",
            "middle": "Middle 50% by genre_popularity_mean",
            "niche": "Bottom 25% by genre_popularity_mean",
        }
    )

    ordered_columns = [
        "mapped_spotify_genre",
        "genre_track_count",
        "genre_popularity_mean",
        "genre_popularity_median",
        "popularity_score_0_100",
        "popularity_rank",
        "popularity_percentile",
        "genre_popularity_segment",
        "listener_type",
        "listener_type_zh",
        "indicator_rule",
    ]
    return index[ordered_columns].sort_values("popularity_rank").reset_index(drop=True)


def build_model_ready(df: pd.DataFrame) -> pd.DataFrame:
    model = df.copy()
    model["signup_date"] = model["signup_date"].dt.strftime("%Y-%m-%d")

    drop_columns = [
        "user_id",
        "signup_date",
        "ad_interaction",
        "ad_conversion_to_subscription",
        "subscription_status",
    ]
    model = model.drop(columns=[col for col in drop_columns if col in model.columns])

    numeric_columns = model.select_dtypes(include=["number"]).columns.tolist()
    for column in numeric_columns:
        if model[column].isna().any():
            model[column] = model[column].fillna(model[column].median())

    categorical_columns = [
        column
        for column in model.columns
        if (
            pd.api.types.is_object_dtype(model[column])
            or pd.api.types.is_string_dtype(model[column])
            or isinstance(model[column].dtype, pd.CategoricalDtype)
        )
    ]
    for column in categorical_columns:
        model[column] = model[column].where(model[column].notna(), "Unknown").astype(str)

    model = pd.get_dummies(model, columns=categorical_columns, dummy_na=False)

    scale_columns = [
        column
        for column in numeric_columns
        if column not in {"inactive_3_months_flag", "ad_interaction_flag", "ad_conversion_flag",
                          "is_paid_subscription", "is_active_subscription", "genre_stats_available"}
    ]
    for column in scale_columns:
        std = model[column].std(ddof=0)
        if std and pd.notna(std):
            model[f"{column}_z"] = (model[column] - model[column].mean()) / std

    return model


def write_summary(raw_users: pd.DataFrame, analysis: pd.DataFrame, genre_stats: pd.DataFrame) -> None:
    unmatched = (
        analysis.loc[analysis["genre_stats_available"] == 0, "favorite_genre"]
        .value_counts()
        .sort_index()
    )
    matched_rate = analysis["genre_stats_available"].mean()
    overview = genre_stats[
        [
            "mapped_spotify_genre",
            "genre_track_count",
            "genre_popularity_mean",
            "genre_popularity_segment",
            "genre_audio_bubble_index",
        ]
    ].sort_values("genre_popularity_mean", ascending=False)
    overview_lines = [
        "| mapped_spotify_genre | genre_track_count | genre_popularity_mean | genre_popularity_segment | genre_audio_bubble_index |",
        "| --- | ---: | ---: | --- | ---: |",
    ]
    for row in overview.itertuples(index=False):
        overview_lines.append(
            f"| {row.mapped_spotify_genre} | {row.genre_track_count:,} | "
            f"{row.genre_popularity_mean:.2f} | {row.genre_popularity_segment} | "
            f"{row.genre_audio_bubble_index:.4f} |"
        )

    lines = [
        "# Spotify User Behavior Dataset preprocessing summary",
        "",
        f"- Input rows: {len(raw_users):,}",
        f"- Output analysis rows: {len(analysis):,}",
        f"- Output model rows: {len(analysis):,}",
        f"- Duplicate user_id rows removed: {len(raw_users) - len(analysis):,}",
        f"- Missing values after preprocessing: {int(analysis.isna().sum().sum()):,}",
        f"- User genres matched to spotify_songs.csv: {matched_rate:.1%}",
        "",
        "## Created files",
        "",
        f"- `{ANALYSIS_OUTPUT.relative_to(ROOT)}`: cleaned, enriched, analysis-friendly data.",
        f"- `{MODEL_OUTPUT.relative_to(ROOT)}`: one-hot encoded and median-imputed model table.",
        f"- `{GENRE_STATS_OUTPUT.relative_to(ROOT)}`: genre-level popularity/audio reference table.",
        f"- `{POPULARITY_INDEX_OUTPUT.relative_to(ROOT)}`: genre popularity indicator table.",
        "",
        "## Important transformations",
        "",
        "- Parsed `signup_date` and added `signup_year`, `signup_month`, `signup_quarter`, `tenure_days`, and `tenure_months`.",
        "- Added `age_group` cohorts: 10-19, 20-29, 30-39, 40-49, 50-59, 60+.",
        "- Converted Yes/No fields into `ad_interaction_flag` and `ad_conversion_flag`.",
        "- Added subscription flags: `is_paid_subscription` and `is_active_subscription`.",
        "- Mapped available user genres to `spotify_songs.csv` playlist genres and joined genre popularity/audio statistics.",
        "- Added `genre_popularity_segment` using the top/bottom quartiles of genre mean popularity.",
        "- Added `genre_audio_bubble_index`, where larger values mean the genre has a tighter audio-feature space.",
        "",
        "## Unmatched user genres",
        "",
    ]
    if unmatched.empty:
        lines.append("- None")
    else:
        lines.extend([f"- {genre}: {count:,}" for genre, count in unmatched.items()])

    lines.extend(
        [
            "",
            "## Genre reference overview",
            "",
            "\n".join(overview_lines),
            "",
        ]
    )
    SUMMARY_OUTPUT.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    users = pd.read_csv(USER_INPUT)
    songs = pd.read_csv(SONGS_INPUT)

    genre_stats = build_genre_reference(songs)
    popularity_index = build_popularity_index(genre_stats)
    analysis = preprocess_user_behavior(users, genre_stats)
    model = build_model_ready(analysis)

    genre_stats.to_csv(GENRE_STATS_OUTPUT, index=False)
    popularity_index.to_csv(POPULARITY_INDEX_OUTPUT, index=False)
    analysis.to_csv(ANALYSIS_OUTPUT, index=False)
    model.to_csv(MODEL_OUTPUT, index=False)
    write_summary(users, analysis, genre_stats)

    print(f"Wrote {ANALYSIS_OUTPUT}")
    print(f"Wrote {MODEL_OUTPUT}")
    print(f"Wrote {GENRE_STATS_OUTPUT}")
    print(f"Wrote {POPULARITY_INDEX_OUTPUT}")
    print(f"Wrote {SUMMARY_OUTPUT}")


if __name__ == "__main__":
    main()
