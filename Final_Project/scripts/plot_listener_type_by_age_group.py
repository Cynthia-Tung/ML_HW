from __future__ import annotations

from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = ROOT / "dataset"
FIGURE_DIR = ROOT / "figures"
USER_LISTENER_INPUT = DATASET_DIR / "spotify_user_listener_type_mapping.csv"
CHART_OUTPUT = FIGURE_DIR / "listener_type_by_age_group_bar_chart.png"
SUMMARY_OUTPUT = DATASET_DIR / "listener_type_by_age_group_summary.csv"
FONT_REGULAR = Path("C:/Windows/Fonts/msjh.ttc")
FONT_BOLD = Path("C:/Windows/Fonts/msjhbd.ttc")

AGE_GROUPS = ["10-19", "20-29", "30-39", "40-49", "50-59", "60+"]
LISTENER_TYPES = ["主流型用戶", "中間型用戶", "小眾挖掘者"]
COLORS = {
    "主流型用戶": "#2F80ED",
    "中間型用戶": "#7B8794",
    "小眾挖掘者": "#D64545",
}


def load_font(path: Path, size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    if path.exists():
        return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def build_summary(df: pd.DataFrame) -> pd.DataFrame:
    filtered = df[df["listener_type_zh"].isin(LISTENER_TYPES)].copy()
    counts = (
        filtered.groupby(["age_group", "listener_type_zh"], observed=False)
        .size()
        .unstack(fill_value=0)
        .reindex(index=AGE_GROUPS, columns=LISTENER_TYPES, fill_value=0)
    )
    percentages = counts.div(counts.sum(axis=1), axis=0).fillna(0) * 100

    summary = counts.reset_index()
    for listener_type in LISTENER_TYPES:
        summary[f"{listener_type}_percentage"] = percentages[listener_type].round(2).values
    summary["included_users"] = counts.sum(axis=1).values
    return summary


def draw_chart(summary: pd.DataFrame, unmatched_count: int) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    width, height = 1500, 900
    margin_left, margin_right = 120, 70
    margin_top, margin_bottom = 145, 155
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom

    bg = "#F8FAFC"
    text = "#102A43"
    muted = "#52616B"
    grid = "#D9E2EC"
    axis = "#243447"

    image = Image.new("RGB", (width, height), bg)
    draw = ImageDraw.Draw(image)
    font_title = load_font(FONT_BOLD, 42)
    font_subtitle = load_font(FONT_REGULAR, 23)
    font_axis = load_font(FONT_REGULAR, 21)
    font_tick = load_font(FONT_REGULAR, 19)
    font_label = load_font(FONT_BOLD, 19)
    font_legend = load_font(FONT_REGULAR, 21)
    font_note = load_font(FONT_REGULAR, 18)

    draw.text((margin_left, 36), "各年齡層中的主流 / 中間 / 小眾聽眾數量", fill=text, font=font_title)
    draw.text(
        (margin_left, 91),
        "依使用者 favorite_genre 對齊曲風熱門度指標；長條高度代表該年齡層的人數",
        fill=muted,
        font=font_subtitle,
    )

    max_count = int(summary[LISTENER_TYPES].to_numpy().max())
    y_max = ((max_count // 200) + 1) * 200
    if y_max < 1000:
        y_max = 1000

    for tick in range(0, y_max + 1, 200):
        y = margin_top + plot_height - int(plot_height * tick / y_max)
        draw.line((margin_left, y, width - margin_right, y), fill=grid, width=1)
        draw.text((margin_left - 78, y - 12), f"{tick:,}", fill=muted, font=font_tick)

    draw.line((margin_left, margin_top, margin_left, margin_top + plot_height), fill=axis, width=2)
    draw.line(
        (margin_left, margin_top + plot_height, width - margin_right, margin_top + plot_height),
        fill=axis,
        width=2,
    )

    group_width = plot_width / len(AGE_GROUPS)
    bar_width = min(48, int(group_width / 5))
    bar_gap = 12
    total_bar_width = bar_width * len(LISTENER_TYPES) + bar_gap * (len(LISTENER_TYPES) - 1)

    for group_index, row in summary.iterrows():
        group_center = margin_left + group_width * group_index + group_width / 2
        start_x = group_center - total_bar_width / 2
        for type_index, listener_type in enumerate(LISTENER_TYPES):
            count = int(row[listener_type])
            x0 = int(start_x + type_index * (bar_width + bar_gap))
            x1 = x0 + bar_width
            y1 = margin_top + plot_height
            y0 = y1 - int(plot_height * count / y_max)
            draw.rounded_rectangle((x0, y0, x1, y1), radius=7, fill=COLORS[listener_type])
            label = f"{count:,}"
            bbox = draw.textbbox((0, 0), label, font=font_label)
            label_x = x0 + (bar_width - (bbox[2] - bbox[0])) / 2
            draw.text((label_x, y0 - 26), label, fill=text, font=font_label)

        age_label = str(row["age_group"])
        bbox = draw.textbbox((0, 0), age_label, font=font_axis)
        draw.text(
            (group_center - (bbox[2] - bbox[0]) / 2, margin_top + plot_height + 24),
            age_label,
            fill=text,
            font=font_axis,
        )

    draw.text((24, margin_top + 5), "人數", fill=muted, font=font_axis)
    draw.text((width - 128, margin_top + plot_height + 24), "年齡層", fill=muted, font=font_axis)

    legend_y = height - 87
    legend_x = margin_left
    for listener_type in LISTENER_TYPES:
        draw.rounded_rectangle(
            (legend_x, legend_y, legend_x + 28, legend_y + 28),
            radius=5,
            fill=COLORS[listener_type],
        )
        draw.text((legend_x + 40, legend_y - 1), listener_type, fill=text, font=font_legend)
        legend_x += 190

    note = f"註：資料集2無對應曲風的使用者未納入此圖，未納入人數 {unmatched_count:,} 人。"
    draw.text((margin_left, height - 38), note, fill=muted, font=font_note)

    image.save(CHART_OUTPUT)


def main() -> None:
    df = pd.read_csv(USER_LISTENER_INPUT)
    summary = build_summary(df)
    unmatched_count = int((df["listener_type_zh"] == "資料集2無對應曲風").sum())

    summary.to_csv(SUMMARY_OUTPUT, index=False, encoding="utf-8-sig")
    draw_chart(summary, unmatched_count)

    print(f"Wrote {SUMMARY_OUTPUT}")
    print(f"Wrote {CHART_OUTPUT}")


if __name__ == "__main__":
    main()
