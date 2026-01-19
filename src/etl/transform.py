import pandas as pd

def transform_raw_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df.rename(
        columns={
            "publishedAt": "published_at",
            "view_count": "views"
        },
        inplace=True
    )

    df["published_at"] = pd.to_datetime(df["published_at"], errors="coerce").dt.tz_localize(None)
    df["trending_date"] = pd.to_datetime(df["trending_date"], errors="coerce").dt.tz_localize(None)

    df.dropna(
        subset=["video_id", "trending_date", "views", "likes", "comment_count"],
        inplace=True
    )

    df.drop_duplicates(
        subset=["video_id", "trending_date"],
        inplace=True
    )

    df.sort_values(["video_id", "trending_date"], inplace=True)

    return df
