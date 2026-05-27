from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = ROOT / "dataset"
USER_INPUT = DATASET_DIR / "spotify_user_behavior_realistic_50000_rows.csv"
SONGS_INPUT = DATASET_DIR / "spotify_songs.csv"
POPULARITY_INDEX_INPUT = DATASET_DIR / "spotify_genre_popularity_index.csv"
USER_LISTENER_OUTPUT = DATASET_DIR / "spotify_user_listener_type_mapping.csv"
REPRESENTATIVE_SONGS_OUTPUT = DATASET_DIR / "spotify_representative_songs_by_genre.csv"

USER_TO_SPOTIFY_GENRE = {
    "Pop": "pop",
    "Rock": "rock",
    "Latin": "latin",
    "R&B": "r&b",
    "Hip-Hop": "rap",
    "Electronic": "edm",
}


def add_age_groups(df: pd.DataFrame) -> pd.DataFrame:
    bins = [0, 19, 29, 39, 49, 59, 200]
    labels = ["10-19", "20-29", "30-39", "40-49", "50-59", "60+"]
    df["age_group"] = pd.cut(df["age"], bins=bins, labels=labels, right=True)
    return df


def build_user_listener_mapping(users: pd.DataFrame, popularity_index: pd.DataFrame) -> pd.DataFrame:
    users = users.copy()
    users["mapped_spotify_genre"] = users["favorite_genre"].map(USER_TO_SPOTIFY_GENRE)
    users = add_age_groups(users)

    mapping_columns = [
        "mapped_spotify_genre",
        "genre_popularity_mean",
        "popularity_rank",
        "genre_popularity_segment",
        "listener_type",
        "listener_type_zh",
    ]
    users = users.merge(popularity_index[mapping_columns], on="mapped_spotify_genre", how="left")
    users["genre_popularity_segment"] = users["genre_popularity_segment"].fillna("unmatched")
    users["listener_type"] = users["listener_type"].fillna("Unmatched")
    users["listener_type_zh"] = users["listener_type_zh"].fillna("資料集2無對應曲風")

    output_columns = [
        "user_id",
        "age",
        "age_group",
        "country",
        "favorite_genre",
        "mapped_spotify_genre",
        "genre_popularity_mean",
        "popularity_rank",
        "genre_popularity_segment",
        "listener_type",
        "listener_type_zh",
        "subscription_type",
        "primary_device",
        "avg_listening_hours_per_week",
        "avg_skips_per_day",
    ]
    return users[output_columns]


def build_representative_songs(
    songs: pd.DataFrame, popularity_index: pd.DataFrame, top_n: int = 10
) -> pd.DataFrame:
    songs = songs.copy()
    songs = songs.merge(
        popularity_index[
            [
                "mapped_spotify_genre",
                "genre_popularity_segment",
                "listener_type",
                "listener_type_zh",
            ]
        ],
        left_on="playlist_genre",
        right_on="mapped_spotify_genre",
        how="inner",
    )
    songs = songs.sort_values(
        ["playlist_genre", "track_popularity", "track_name"],
        ascending=[True, False, True],
    )
    songs["genre_song_rank"] = songs.groupby("playlist_genre").cumcount() + 1
    songs = songs[songs["genre_song_rank"] <= top_n]

    output_columns = [
        "playlist_genre",
        "genre_popularity_segment",
        "listener_type_zh",
        "genre_song_rank",
        "track_name",
        "track_artist",
        "track_popularity",
        "playlist_subgenre",
        "track_album_release_date",
    ]
    return songs[output_columns].reset_index(drop=True)


def main() -> None:
    users = pd.read_csv(USER_INPUT)
    songs = pd.read_csv(SONGS_INPUT)
    popularity_index = pd.read_csv(POPULARITY_INDEX_INPUT)

    user_mapping = build_user_listener_mapping(users, popularity_index)
    representative_songs = build_representative_songs(songs, popularity_index)

    user_mapping.to_csv(USER_LISTENER_OUTPUT, index=False, encoding="utf-8-sig")
    representative_songs.to_csv(REPRESENTATIVE_SONGS_OUTPUT, index=False, encoding="utf-8-sig")
    print(f"Wrote {USER_LISTENER_OUTPUT}")
    print(f"Wrote {REPRESENTATIVE_SONGS_OUTPUT}")


if __name__ == "__main__":
    main()
