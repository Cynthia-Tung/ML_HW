# Spotify User Behavior Dataset preprocessing summary

- Input rows: 50,000
- Output analysis rows: 50,000
- Output model rows: 50,000
- Duplicate user_id rows removed: 0
- Missing values after preprocessing: 932,289
- User genres matched to spotify_songs.csv: 49.6%

## Created files

- `dataset\spotify_user_behavior_preprocessed.csv`: cleaned, enriched, analysis-friendly data.
- `dataset\spotify_user_behavior_model_ready.csv`: one-hot encoded and median-imputed model table.
- `dataset\spotify_genre_reference_stats.csv`: genre-level popularity/audio reference table.

## Important transformations

- Parsed `signup_date` and added `signup_year`, `signup_month`, `signup_quarter`, `tenure_days`, and `tenure_months`.
- Added `age_group` cohorts: 10-19, 20-29, 30-39, 40-49, 50-59, 60+.
- Converted Yes/No fields into `ad_interaction_flag` and `ad_conversion_flag`.
- Added subscription flags: `is_paid_subscription` and `is_active_subscription`.
- Mapped available user genres to `spotify_songs.csv` playlist genres and joined genre popularity/audio statistics.
- Added `genre_popularity_segment` using the top/bottom quartiles of genre mean popularity.
- Added `genre_audio_bubble_index`, where larger values mean the genre has a tighter audio-feature space.

## Unmatched user genres

- Bollywood: 4,246
- Classical: 4,272
- Country: 4,135
- Indie: 4,182
- Jazz: 4,134
- K-Pop: 4,228

## Genre reference overview

| mapped_spotify_genre | genre_track_count | genre_popularity_mean | genre_popularity_segment | genre_audio_bubble_index |
| --- | ---: | ---: | --- | ---: |
| pop | 5,507 | 47.74 | mainstream | 0.0016 |
| latin | 5,155 | 47.03 | mainstream | 0.0012 |
| rap | 5,746 | 43.22 | middle | 0.0010 |
| rock | 4,951 | 41.73 | middle | 0.0012 |
| r&b | 5,431 | 41.22 | niche | 0.0012 |
| edm | 6,043 | 34.83 | niche | 0.0041 |
