from __future__ import annotations

from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = ROOT / "dataset"
FIGURE_DIR = ROOT / "figures"
GENRE_STATS_INPUT = DATASET_DIR / "spotify_genre_reference_stats.csv"
POPULARITY_INDEX_OUTPUT = DATASET_DIR / "spotify_genre_popularity_index.csv"
POPULARITY_CHART_OUTPUT = FIGURE_DIR / "spotify_genre_popularity_index.png"
FONT_REGULAR = Path("C:/Windows/Fonts/msjh.ttc")
FONT_BOLD = Path("C:/Windows/Fonts/msjhbd.ttc")


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


def load_font(path: Path, size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    if path.exists():
        return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def draw_popularity_chart(popularity_index: pd.DataFrame) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    df = popularity_index.sort_values("popularity_score_0_100", ascending=True).reset_index(
        drop=True
    )
    width, height = 1400, 850
    margin_left, margin_right = 240, 90
    margin_top, margin_bottom = 145, 105
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom
    bar_gap = 22
    bar_height = int((plot_height - bar_gap * (len(df) - 1)) / len(df))

    colors = {
        "mainstream": "#2F80ED",
        "middle": "#7B8794",
        "niche": "#D64545",
    }
    bg = "#F8FAFC"
    axis = "#243447"
    grid = "#D9E2EC"
    text = "#102A43"

    image = Image.new("RGB", (width, height), bg)
    draw = ImageDraw.Draw(image)
    font_title = load_font(FONT_BOLD, 42)
    font_subtitle = load_font(FONT_REGULAR, 24)
    font_label = load_font(FONT_BOLD, 25)
    font_small = load_font(FONT_REGULAR, 22)
    font_note = load_font(FONT_REGULAR, 19)

    draw.text((margin_left, 38), "Spotify 曲風熱門度指標", fill=text, font=font_title)
    draw.text(
        (margin_left, 92),
        "依 genre_popularity_mean 建立 0-100 分數；前 25% = 主流，後 25% = 小眾",
        fill="#52616B",
        font=font_subtitle,
    )

    for tick in range(0, 101, 20):
        x = margin_left + int(plot_width * tick / 100)
        draw.line((x, margin_top - 12, x, height - margin_bottom + 8), fill=grid, width=1)
        draw.text((x - 12, height - margin_bottom + 22), str(tick), fill="#52616B", font=font_note)

    draw.line(
        (margin_left, height - margin_bottom, width - margin_right, height - margin_bottom),
        fill=axis,
        width=2,
    )

    for i, row in df.iterrows():
        y = margin_top + i * (bar_height + bar_gap)
        score = float(row["popularity_score_0_100"])
        segment = row["genre_popularity_segment"]
        bar_width = int(plot_width * score / 100)
        color = colors.get(segment, "#7B8794")

        draw.text(
            (62, y + bar_height / 2 - 15),
            str(row["mapped_spotify_genre"]),
            fill=text,
            font=font_label,
        )
        draw.rounded_rectangle(
            (margin_left, y, margin_left + bar_width, y + bar_height),
            radius=10,
            fill=color,
        )

        value_label = (
            f"{score:.1f} 分 | 平均熱門度 {row['genre_popularity_mean']:.2f} | "
            f"{row['listener_type_zh']}"
        )
        label_x = margin_left + bar_width + 16
        if label_x > width - 430:
            label_x = margin_left + bar_width - 420
            label_fill = "white"
        else:
            label_fill = text
        draw.text((label_x, y + bar_height / 2 - 14), value_label, fill=label_fill, font=font_small)

    legend_y = height - 58
    legend_items = [
        ("主流型用戶", colors["mainstream"]),
        ("中間型用戶", colors["middle"]),
        ("小眾挖掘者", colors["niche"]),
    ]
    x = margin_left
    for label, color in legend_items:
        draw.rounded_rectangle((x, legend_y, x + 26, legend_y + 26), radius=5, fill=color)
        draw.text((x + 36, legend_y - 1), label, fill=text, font=font_note)
        x += 190

    image.save(POPULARITY_CHART_OUTPUT)


def main() -> None:
    genre_stats = pd.read_csv(GENRE_STATS_INPUT)
    popularity_index = build_popularity_index(genre_stats)
    popularity_index.to_csv(POPULARITY_INDEX_OUTPUT, index=False, encoding="utf-8-sig")
    draw_popularity_chart(popularity_index)
    print(f"Wrote {POPULARITY_INDEX_OUTPUT}")
    print(f"Wrote {POPULARITY_CHART_OUTPUT}")


if __name__ == "__main__":
    main()
