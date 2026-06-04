from __future__ import annotations

from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]

LASTFM_DIR = ROOT / "lastfm"
OUTPUT_DIR = ROOT / "dataset"
FIGURE_DIR = ROOT / "figures"

TAGS_CLEANED = LASTFM_DIR / "tags_cleaned.dat"
USER_TAGGED_ARTISTS = LASTFM_DIR / "user_taggedartists.dat"
USER_ARTISTS = LASTFM_DIR / "user_artists.dat"

GENRE_INDEX_OUTPUT = OUTPUT_DIR / "lastfm_genre_popularity_index.csv"
USER_LABEL_OUTPUT = OUTPUT_DIR / "lastfm_user_genre_labels.csv"
SUMMARY_OUTPUT = OUTPUT_DIR / "lastfm_listener_type_summary.csv"

POPULARITY_CHART_OUTPUT = (
    FIGURE_DIR / "lastfm_genre_popularity_index.png"
)

DISTRIBUTION_CHART_OUTPUT = (
    FIGURE_DIR /
    "lastfm_genre_distribution.png"
)

FONT_REGULAR = Path("C:/Windows/Fonts/arial.ttf")
FONT_BOLD = Path("C:/Windows/Fonts/arialbd.ttf")

INVALID_GENRES = {
    "girl singer",
    "songwriter",
    "instrumental",
    "soundtrack",
    "piano",
    "guitar",
    "ballad",
    "female vocalists",
    "male vocalists",
    "favorites",
    "favorite",
    "favourite",
    "awesome",
    "good",
    "love",
    "cool",
    "beautiful",
    "american",
    "british",
    "seen live",
    "albums i own",
    "favorite songs",
    "favourites",
    "my favorites",
    "under 2000 listeners",
    "under 1000 listeners",
    "under 500 listeners",
    "under 100 listeners",
    "under 50 listeners",
    "seen live",
    "download",
    "misc",
    "unknown",
    "rock female",
    "ninja tune",
    "boyband",
    "gaga",
    "pop queen",
    "music",
    "violin",
    "idol",
    "duet",
    "dance mix",
    "groove",
    "band",
    "albums",
    "beatles",
    "breakbeat",
    "cello",
    "flamenco",
    "harmonica",
    "guns n roses",
    "nice song",
    "dance queen",
    "radio",
    "singer",
    "concert",
    "flute",
    "guitar god",
    "music to cry",
    "songs on repeat",
    "composer",
    "eurovision",
    "fashion",
    "melodia",
    "dj",
    "bass",
    "drum",
    "rhythm",
    "sax",
    "pop 90",
    "rock 80s",
    "melodic",
    "jam",
    "timbaland",
    "rpg music",
    
    # 廣泛
    "symphonic",
    "glam",
    "prog",
    
    # 展現手法
    "acoustic",
    "musical",
    "romantica",
    "traditional",
    "lyrical",
    "dat riff",
    "slide guitar",
    "turntablism",
}

GENRE_MAPPING = {

    # metal
    "death metal": "metal",
    "thrash metal": "metal",
    "nu metal": "metal",
    "metalcore": "metal",
    "proto metal": "metal",
    "melodic metal": "metal",
    "progressive metal": "metal",
    "math metal": "metal",
    "stoner metal": "metal",
    "groove metal": "metal",
    "power metal": "metal",
    "black metal": "metal",
    "doom metal": "metal",
    "deathcore": "metal",
    "german metal": "metal",
    "speed metal": "metal",
    "metal church": "metal",
    "avant-garde metal": "metal",
    "suicidal black metal": "metal",
    "us metal": "metal",
    "kill metal": "metal",
    "mexican metal": "metal",
    "true metal": "metal",
    "extreme metal": "metal",
    "finnish death metal": "metal",
    "war metal": "metal",
    "thrash": "metal",
    "epic metal": "metal",
    "hair metal": "metal",

    # rock
    "alternative rock": "rock",
    "hard rock": "rock",
    "progressive rock": "rock",
    "prog-rock": "rock",
    "classic rock": "rock",
    "melodic rock": "rock",
    "art rock": "rock",
    "noise rock": "rock",
    "kulfrock": "rock",
    "rock n roll": "rock",
    "psychedelic rock": "rock",
    "garage rock": "rock",
    "souther rock": "rock",
    "space-rock": "rock",
    "math rock": "rock",
    "piano rock": "rock",
    "brit rock": "rock",
    "britrock": "rock",
    "noisepop": "rock",
    "slowcore": "rock",
    "rock espanol": "rock",
    "album rock": "rock",
    "experimental rock": "rock",
    "kultrock": "rock",
    "italian rock": "rock",
    "nz rock": "rock",
    "j-rock": "rock",
    "jrock": "rock",
    "retro rock": "rock",
    "russian rock": "rock",
    "riot grrl": "rock",
    "dark rock": "rock",
    "shock rock": "rock",
    "deutschrock": "rock",
    "chick rock": "rock",
    "glam rock": "rock",
    "rock brasil": "rock",
    "finnish rock": "rock",
    "rock argento": "rock",
    "rock brasil": "rock",
    
    # 可獨立
    #"new wave": "rock",
    #"grunge": "rock",
    #"shoegaze": "rock",
    
    # punk -> rock
    "punk rock": "rock",
    "french punk": "rock",
    "pirate punk": "rock",
    "punk n roll": "rock",
    "punkrock": "rock",
    "skate-punk": "rock",

    # pop
    "synthpop": "pop",
    "dreampop": "pop",
    "pop music": "pop",
    "uk pop": "pop",
    "dark pop": "pop",
    "french pop": "pop",
    "latin pop": "pop",
    
    # k-pop
    "kpop": "k-pop",
    
    # 可獨立
    #"britpop": "pop/rock",
    #"j-pop": "pop",
    
    # 可再討論
    #"dance": "pop",
    "dancehall": "reggae",
    
    # dance pop
    "dancepop": "dance pop",
    
    # pop rock
    "powerpop": "pop rock",
    "power pop": "pop rock",
    "poprock": "pop rock",

    # electronic
    "trance": "electronic",
    "house": "electronic",
    "techno": "electronic",
    "trip hop": "electronic",
    "tri hop": "electronic",
    "deep house": "electronic",
    "downtempo": "electronic",
    "dark ambient": "electronic",
    "psytrance": "electronic",
    "dubstep": "electronic",
    "drum bass": "electronic",
    "breakcore": "electronic",
    "progressive house": "electronic",
    "electro house": "electronic",
    "tech house": "electronic",
    "eletronic": "electronic",
    "synth": "electronic",

    # rap
    "hip hop": "rap",
    "hiphop": "rap",
    "japanese rap": "rap",
    "hardcore rap": "rap",
    "gangsta": "rap",
    "bay rap": "rap",
    "polish rap": "rap",
    "crunk": "rap",
    
    # blues
    "texas blues": "blues",
    
    # r&b
    "rhythm & blues": "r&b",
    
    # jazz
    "swing": "jazz",
    "jazzy": "jazz",
    "bebop": "jazz",
    
    # chanson
    "chanson francaise": "chanson",
    
    # gothic
    "goth": "gothic",
    
    # brazilian music
    "samba": "brazilian music",
    "bossa nova": "brazilian music",
    
    # classical
    "baroque": "classical",
    "opera": "classical",
    "orchestra": "classical",
    
    # latin
    "salsa": "latin",
    
    # country
    "alt country": "country",
    "modern country": "country",
}

def load_font(path: Path, size: int):
    if path.exists():
        return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()

def clean_genres(df: pd.DataFrame) -> pd.DataFrame:

    df = df.copy()

    df["canonical_genre"] = (
        df["canonical_genre"]
        .str.lower()
        .str.strip()
    )

    df = df[
        ~df["canonical_genre"]
        .isin(INVALID_GENRES)
    ]

    df["canonical_genre"] = (
        df["canonical_genre"]
        .replace(GENRE_MAPPING)
    )

    return df

def segment_genres(index: pd.DataFrame) -> pd.DataFrame:

    index = index.copy()

    index["popularity_rank"] = (
        index["genre_popularity_score"]
        .rank(
            method="first",
            ascending=False,
        )
        .astype(int)
    )

    total = len(index)

    mainstream_cutoff = max(
        1,
        int(total * 0.25),
    )

    niche_cutoff = int(total * 0.75)

    index["genre_popularity_segment"] = "middle"

    index.loc[
        index["popularity_rank"]
        <= mainstream_cutoff,
        "genre_popularity_segment",
    ] = "mainstream"

    index.loc[
        index["popularity_rank"]
        > niche_cutoff,
        "genre_popularity_segment",
    ] = "niche"
    
    index["listener_type"] = index[
        "genre_popularity_segment"
    ].map(
        {
            "mainstream": "Mainstream",
            "middle": "Middle",
            "niche": "Niche",
        }
    )

    return index


def build_genre_popularity_index(
    user_artist_genres: pd.DataFrame,
) -> pd.DataFrame:

    genre_index = (
        user_artist_genres.groupby("canonical_genre")
        .agg(
            tag_assignment_count=("tagID", "count"),
            unique_users=("userID", "nunique"),
            unique_artists=("artistID", "nunique"),
            listening_weight_sum=("weight", "sum"),
            listening_weight_mean=("weight", "mean"),
        )
        .reset_index()
    )
    
    genre_index = genre_index[
        genre_index["unique_users"] >= 20
    ]
    
    print(
        f"Remaining genres: {len(genre_index)}"
    )

    genre_index["genre_popularity_score"] = (
        genre_index["unique_users"]
    )

    genre_index = segment_genres(genre_index)

    return genre_index.sort_values(
        "popularity_rank"
    )


def build_user_labels(
    user_artist_genres: pd.DataFrame,
    genre_index: pd.DataFrame,
) -> pd.DataFrame:

    user_genre_scores = (
        user_artist_genres.groupby(
            ["userID", "canonical_genre"]
        )
        .agg(
            genre_listening_weight=("weight", "sum"),
            genre_artist_count=("artistID", "nunique"),
            genre_tag_assignment_count=("tagID", "count"),
        )
        .reset_index()
    )

    user_top_genre = (
        user_genre_scores.sort_values(
            [
                "userID",
                "genre_listening_weight",
                "genre_artist_count",
                "genre_tag_assignment_count",
            ],
            ascending=[True, False, False, False],
        )
        .drop_duplicates("userID")
    )

    user_top_genre = user_top_genre.merge(
        genre_index[
            [
                "canonical_genre",
                "genre_popularity_score",
                "popularity_rank",
                "genre_popularity_segment",
                "listener_type",
            ]
        ],
        on="canonical_genre",
        how="left",
    )

    return user_top_genre.rename(
        columns={
            "canonical_genre": "top_canonical_genre"
        }
    )


def build_listener_summary(
    user_labels: pd.DataFrame,
) -> pd.DataFrame:

    summary = (
        user_labels.groupby(
            [
                "genre_popularity_segment",
                "listener_type",
            ]
        )
        .agg(
            user_count=("userID", "count")
        )
        .reset_index()
    )

    summary["percentage"] = (
        summary["user_count"]
        / summary["user_count"].sum()
        * 100
    ).round(2)

    return summary.sort_values(
        "user_count",
        ascending=False,
    )


def draw_popularity_chart(
    genre_index: pd.DataFrame,
) -> None:

    FIGURE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    df = (
        genre_index
        .sort_values(
            "genre_popularity_score",
            ascending=False,
        )
        .head(10)
        .reset_index(drop=True)
    )

    if len(df) == 0:

        print(
            "No genres available for chart."
        )

        return

    #df = df.sort_values(
    #    "genre_popularity_score",
    #    ascending=True,
    #).reset_index(drop=True)

    width = 1600
    height = 700

    margin_left = 320
    margin_right = 120
    margin_top = 150
    margin_bottom = 100

    plot_width = (
        width
        - margin_left
        - margin_right
    )

    colors = {
        "mainstream": "#2F80ED",
        "middle": "#7B8794",
        "niche": "#D64545",
    }

    image = Image.new(
        "RGB",
        (width, height),
        "#F8FAFC",
    )

    draw = ImageDraw.Draw(image)

    font_title = load_font(
        FONT_BOLD,
        40,
    )

    font_label = load_font(
        FONT_BOLD,
        22,
    )

    font_small = load_font(
        FONT_REGULAR,
        18,
    )

    draw.text(
        (margin_left, 40),
        "Top 10 Genre Popularity Ranking",
        fill="#102A43",
        font=font_title,
    )

    draw.text(
        (margin_left, 95),
        "Popularity Score = Unique Users",
        fill="#52616B",
        font=font_small,
    )

    max_score = (
        df["genre_popularity_score"]
        .max()
    )

    bar_height = 30
    gap = 14

    for idx, row in df.iterrows():

        y = (
            margin_top
            + idx * (bar_height + gap)
        )

        score = row[
            "genre_popularity_score"
        ]

        bar_width = int(
            score
            / max_score
            * plot_width
        )

        segment = row[
            "genre_popularity_segment"
        ]

        color = colors.get(
            segment,
            "#7B8794"
        )

        draw.text(
            (
                20,
                y,
            ),
            f"#{idx+1} {row['canonical_genre']}",
            fill="#102A43",
            font=font_label,
        )

        draw.rounded_rectangle(
            (
                margin_left,
                y,
                margin_left + bar_width,
                y + bar_height,
            ),
            radius=6,
            fill=color,
        )

        draw.text(
            (
                margin_left
                + bar_width
                + 10,
                y,
            ),
            str(
                int(score)
            ),
            fill="#102A43",
            font=font_small,
        )

    image.save(
        POPULARITY_CHART_OUTPUT
    )


def draw_distribution_chart(
    genre_index: pd.DataFrame,
) -> None:

    distribution = (
        genre_index
        .groupby(
            "genre_popularity_segment"
        )
        .size()
        .reset_index(
            name="genre_count"
        )
    )

    width = 900
    height = 700

    image = Image.new(
        "RGB",
        (width, height),
        "#F8FAFC",
    )

    draw = ImageDraw.Draw(image)

    font_title = load_font(
        FONT_BOLD,
        36,
    )

    font_label = load_font(
        FONT_BOLD,
        24,
    )

    font_small = load_font(
        FONT_REGULAR,
        20,
    )

    colors = {
        "mainstream": "#2F80ED",
        "middle": "#7B8794",
        "niche": "#D64545",
    }

    draw.text(
        (60, 40),
        "Genre Popularity Distribution",
        fill="#102A43",
        font=font_title,
    )

    max_count = (
        distribution["genre_count"]
        .max()
    )

    start_x = 180
    base_y = 550

    bar_width = 120
    gap = 120

    for i, row in distribution.iterrows():

        count = row["genre_count"]

        height_ratio = (
            count / max_count
        )

        bar_height = int(
            height_ratio * 350
        )

        x = start_x + i * (
            bar_width + gap
        )

        y = base_y - bar_height

        draw.rectangle(
            (
                x,
                y,
                x + bar_width,
                base_y,
            ),
            fill=colors[
                row[
                    "genre_popularity_segment"
                ]
            ],
        )

        draw.text(
            (
                x + 20,
                y - 35,
            ),
            str(count),
            fill="#102A43",
            font=font_small,
        )

        draw.text(
            (
                x - 10,
                base_y + 15,
            ),
            row[
                "genre_popularity_segment"
            ].capitalize(),
            fill="#102A43",
            font=font_label,
        )

    image.save(
        DISTRIBUTION_CHART_OUTPUT
    )


def main() -> None:

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("Loading datasets...")

    tags = pd.read_csv(
        TAGS_CLEANED,
        sep="\t",
        encoding="latin-1",
    )

    user_tags = pd.read_csv(
        USER_TAGGED_ARTISTS,
        sep="\t",
        encoding="latin-1",
    )

    user_artists = pd.read_csv(
        USER_ARTISTS,
        sep="\t",
        encoding="latin-1",
    )

    tagged_genres = user_tags.merge(
        tags,
        on="tagID",
        how="inner",
    )

    tagged_genres = clean_genres(tagged_genres)

    user_artist_genres = tagged_genres.drop_duplicates(
        subset=[
            "userID",
            "artistID",
            "canonical_genre",
        ]
    )

    user_artist_genres = user_artist_genres.merge(
        user_artists,
        on=["userID", "artistID"],
        how="left",
    )

    user_artist_genres["weight"] = (
        user_artist_genres["weight"]
        .fillna(0)
    )

    print("Building genre popularity index...")

    genre_index = build_genre_popularity_index(
        user_artist_genres
    )

    genre_index.to_csv(
        GENRE_INDEX_OUTPUT,
        index=False,
        encoding="utf-8-sig",
    )

    print("Building user labels...")

    user_labels = build_user_labels(
        user_artist_genres,
        genre_index,
    )

    user_labels.to_csv(
        USER_LABEL_OUTPUT,
        index=False,
        encoding="utf-8-sig",
    )

    print("Building listener summary...")

    summary = build_listener_summary(
        user_labels
    )

    summary.to_csv(
        SUMMARY_OUTPUT,
        index=False,
        encoding="utf-8-sig",
    )

    print("Drawing chart...")

    draw_popularity_chart(
        genre_index
    )
    
    draw_distribution_chart(
        genre_index
    )

    print(f"Wrote {GENRE_INDEX_OUTPUT}")
    print(f"Wrote {USER_LABEL_OUTPUT}")
    print(f"Wrote {SUMMARY_OUTPUT}")
    print(f"Wrote {POPULARITY_CHART_OUTPUT}")
    print(f"Wrote {DISTRIBUTION_CHART_OUTPUT}")

    print(
        "Note: Last.fm dataset does not contain age information."
    )
    
    pd.set_option(
        "display.max_rows",
        None
    )

    print(
        genre_index[
            [
                "canonical_genre",
                "unique_users"
            ]
        ].sort_values(
            "unique_users"
        )
    )

if __name__ == "__main__":
    main()