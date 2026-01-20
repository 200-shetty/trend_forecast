import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path

from src.models.prophet_model import ProphetForecast, forecast_video_views
from src.models.train import RandomForestTrendPredictor
from src.models.ensemble import EnsembleForecaster
from src.etl.extract import extract_raw_data, get_available_countries
from src.etl.transform import transform_raw_data
from src.features.build_features import build_features, get_feature_descriptions
from src.analysis.feature_analysis import (
    analyze_feature_correlations,
    analyze_category_effects,
    analyze_temporal_patterns
)
from src.config import DATA_PROCESSED, MODELS_DIR, CATEGORY_NAMES

st.set_page_config(
    page_title="YouTube Trend Research",
    page_icon="📊",
    layout="wide"
)

# === SIDEBAR ===
st.sidebar.title("Research Dashboard")
st.sidebar.markdown("---")

# Country selection
available_countries = get_available_countries()
if not available_countries:
    available_countries = ["DE"]

country = st.sidebar.selectbox(
    "Select Country",
    available_countries,
    help="Choose the country dataset to analyze"
)

# Page selection
page = st.sidebar.radio(
    "Navigation",
    ["Overview", "Feature Analysis", "Forecast Explorer"]
)

st.sidebar.markdown("---")



# === DATA LOADING ===
@st.cache_data
def load_data(country_code: str):
    """Load and process data for a country."""
    raw_df = extract_raw_data(country_code)
    clean_df = transform_raw_data(raw_df)
    display_features = build_features(clean_df, for_ml=False)
    ml_features = build_features(clean_df, for_ml=True)

    video_info = raw_df[["video_id", "title", "channelTitle"]].drop_duplicates("video_id")
    return clean_df, display_features, ml_features, video_info


@st.cache_resource
def load_or_train_rf_model(_ml_features: pd.DataFrame):
    """Load trained RandomForest model, or train if not available."""
    model_path = MODELS_DIR / "random_forest_model.joblib"

    if model_path.exists():
        return RandomForestTrendPredictor.load(model_path)

    # Train the model if it doesn't exist
    model = RandomForestTrendPredictor()
    model.fit(_ml_features, use_cv=True)
    model.save()
    return model


# Load data
try:
    with st.spinner("Loading data..."):
        clean_df, display_features, ml_features, video_info = load_data(country)
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()

# Load or train model
try:
    with st.spinner("Loading model (training if needed)..."):
        rf_model = load_or_train_rf_model(ml_features)
except Exception as e:
    st.warning(f"Could not load/train model: {e}")
    rf_model = None


# === PAGE: OVERVIEW ===
if page == "Overview":
    st.title("YouTube Trend Forecasting Research")
    st.markdown("""
    This dashboard presents a comprehensive analysis of YouTube trending patterns,
    combining **machine learning** (RandomForest) and **time series forecasting** (Prophet)
    to understand what makes videos go viral.
    """)

    # Key metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Videos", f"{len(display_features):,}")
    with col2:
        st.metric("Avg Trend Duration", f"{display_features['trend_days'].mean():.1f} days")
    with col3:
        st.metric("Max Views (Early)", f"{display_features['early_views'].max():,.0f}")
    with col4:
        unique_channels = video_info["channelTitle"].nunique()
        st.metric("Unique Channels", f"{unique_channels:,}")

    st.markdown("---")

    # Top videos table
    st.subheader("Top Trending Videos")
    st.markdown("""
    Videos with the longest trending duration. Features are computed from the **first 3 days**
    of trending to predict total duration.
    """)

    top_videos = display_features.nlargest(10, "trend_days")
    top_videos = top_videos.merge(video_info, on="video_id", how="left")

    display_cols = ["title", "channelTitle", "early_views", "trend_days",
                   "virality_score", "engagement_rate"]
    available_cols = [c for c in display_cols if c in top_videos.columns]
    st.dataframe(
        top_videos[available_cols].style.format({
            "early_views": "{:,.0f}",
            "virality_score": "{:.1f}",
            "engagement_rate": "{:.4f}"
        }),
        width='stretch'
    )

    # Category distribution
    st.subheader("Category Distribution")

    if "categoryId" in display_features.columns:
        category_counts = display_features["categoryId"].value_counts()
        category_df = pd.DataFrame({
            "Category": category_counts.index.map(CATEGORY_NAMES).fillna("Unknown"),
            "Count": category_counts.values
        })
        st.bar_chart(category_df.set_index("Category"))


# === PAGE: FEATURE ANALYSIS ===
elif page == "Feature Analysis":
    st.title("Feature Analysis: What Drives Virality?")
    st.markdown("""
    **Research Question 3**: What factors drive YouTube virality?

    This analysis examines which features are most predictive of a video's trending duration.
    """)

    # Feature descriptions
    with st.expander("Feature Descriptions"):
        descriptions = get_feature_descriptions()
        for feature, desc in descriptions.items():
            st.markdown(f"- **{feature}**: {desc}")

    # Correlation analysis
    st.subheader("Feature Correlations with Trend Duration")

    correlations = analyze_feature_correlations(ml_features)

    # Define the specific features to show
    positive_features_to_show = ["view_growth_rate", "like_velocity", "view_velocity", "engagement_rate"]

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Positive Correlations** (more → longer trend)")
        # Filter to only show the specified positive features
        positive = correlations[
            (correlations["pearson_r"] > 0) &
            (correlations["feature"].isin(positive_features_to_show))
        ].sort_values("pearson_r", ascending=False)

        for _, row in positive.iterrows():
            st.markdown(f"- `{row['feature']}`: r = {row['pearson_r']:.3f}")

    with col2:
        st.markdown("**Negative Correlations** (more → shorter trend)")
        # Show top negative correlations (excluding the specified positive features)
        negative = correlations[
            (correlations["pearson_r"] < 0) &
            (~correlations["feature"].isin(positive_features_to_show))
        ].sort_values("pearson_r", ascending=True).head(5)

        if len(negative) > 0:
            for _, row in negative.iterrows():
                st.markdown(f"- `{row['feature']}`: r = {row['pearson_r']:.3f}")
        else:
            st.markdown("_No significant negative correlations found_")

    # Category effects
    st.subheader("Category Effects on Trend Duration")

    if "categoryId" in display_features.columns:
        category_effects = analyze_category_effects(display_features)

        st.bar_chart(
            category_effects.set_index("category_name")["mean"].sort_values(ascending=False)
        )

        st.markdown("""
        **Interpretation**: Categories with higher bars tend to produce videos that
        stay on trending longer. Effect size (Cohen's d) indicates practical significance.
        """)

    # Temporal patterns
    st.subheader("Temporal Patterns")

    temporal = analyze_temporal_patterns(display_features)

    if "day_of_week" in temporal:
        st.markdown("**Trend Duration by Publish Day**")
        dow_df = temporal["day_of_week"]
        st.bar_chart(dow_df.set_index("day_name")["mean"])

    if "optimal_hours" in temporal:
        st.info(f"Optimal publish hours (highest avg trend duration): {temporal['optimal_hours']}")

    # Feature importance (model is now always available)
    if rf_model is not None:
        st.subheader("RandomForest Feature Importance")

        importance = rf_model.get_feature_importance()
        top_10 = importance.head(10)

        st.bar_chart(top_10.set_index("feature")["importance"])

        st.markdown(f"""
        **Key Finding**: The top 3 features (`{top_10.iloc[0]['feature']}`,
        `{top_10.iloc[1]['feature']}`, `{top_10.iloc[2]['feature']}`)
        account for {top_10.head(3)['cumulative_importance'].iloc[-1]*100:.1f}% of predictive power.
        """)


# === PAGE: FORECAST EXPLORER ===
elif page == "Forecast Explorer":
    st.title("Video Forecast Explorer")
    st.markdown("""
    Select a video to see its **historical trending performance** and generate a **Prophet forecast**
    for future view counts.
    """)

    # Video selection
    video_counts = clean_df["video_id"].value_counts()
    eligible_videos = video_counts[video_counts >= 3].index.tolist()

    if not eligible_videos:
        st.warning("No videos with enough data points for forecasting.")
        st.stop()

    # Get video titles for selection
    video_options = video_info[video_info["video_id"].isin(eligible_videos)].copy()
    video_options = video_options.drop_duplicates(subset=["video_id"])
    video_options["display"] = video_options["title"].str[:50] + " - " + video_options["channelTitle"]

    selected_display = st.selectbox(
        "Select Video",
        video_options["display"].tolist()[:100],  # Limit for performance
        help="Videos with at least 3 trending days"
    )

    # Get video ID from selection
    selected_row = video_options[video_options["display"] == selected_display].iloc[0]
    video_id = selected_row["video_id"]
    video_title = selected_row["title"]
    channel = selected_row["channelTitle"]

    st.markdown(f"**Video**: {video_title}")
    st.markdown(f"**Channel**: {channel}")

    # Get video data for this video
    video_data = clean_df[clean_df["video_id"] == video_id][["trending_date", "views", "likes"]].copy()
    video_data = video_data.sort_values("trending_date")
    n_days = len(video_data)

    st.info(f"This video has **{n_days} days** of trending data")

    # === HISTORICAL DATA SECTION ===
    st.subheader("Historical Trending Performance")
    st.markdown("""
    This chart shows the **actual view and like counts** during the video's trending period.
    Prophet uses this historical data to learn the video's growth pattern and generate forecasts.
    """)

    # Better historical visualization with plotly
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    fig_hist = make_subplots(specs=[[{"secondary_y": True}]])

    fig_hist.add_trace(
        go.Scatter(
            x=video_data["trending_date"],
            y=video_data["views"],
            mode="lines+markers",
            name="Views",
            line=dict(color="blue", width=2)
        ),
        secondary_y=False
    )

    fig_hist.add_trace(
        go.Scatter(
            x=video_data["trending_date"],
            y=video_data["likes"],
            mode="lines+markers",
            name="Likes",
            line=dict(color="green", width=2)
        ),
        secondary_y=True
    )

    fig_hist.update_layout(
        title="Historical Views & Likes During Trending Period",
        hovermode="x unified"
    )
    fig_hist.update_xaxes(title_text="Date")
    fig_hist.update_yaxes(title_text="Views", secondary_y=False)
    fig_hist.update_yaxes(title_text="Likes", secondary_y=True)

    st.plotly_chart(fig_hist, width='stretch')

    # === FORECAST SECTION ===
    st.subheader("Generate View Forecast")

    # Forecast parameters
    forecast_days = st.slider("Forecast Horizon (days)", 3, 14, 7)

    # Run forecast
    if st.button("Generate Forecast", type="primary"):
        if n_days < 2:
            st.error(f"Need at least 2 data points to forecast. This video only has {n_days}.")
        else:
            with st.spinner("Fitting Prophet model..."):
                try:
                    result = forecast_video_views(clean_df, video_id, periods=forecast_days)
                    forecast = result["forecast"]

                    # Check if Prophet produced unrealistic results
                    min_pred = forecast["yhat"].min()
                    max_actual = video_data["views"].max()
                    has_negative = min_pred < 0
                    is_unrealistic = has_negative or (forecast["yhat"].max() > max_actual * 10)

                    if has_negative or n_days < 10:
                        st.markdown(f"""
                        """)

                    # Display metrics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Training Points", result["n_training_points"])
                    with col2:
                        mae_val = result['metrics']['MAE']
                        st.metric("MAE", f"{mae_val:,.0f}" if mae_val > 100 else f"{mae_val:.2f}")
                    with col3:
                        st.metric("R²", f"{result['metrics']['R2']:.3f}")

                    # Forecast chart
                    st.subheader("View Forecast")

                    forecast["ds"] = pd.to_datetime(forecast["ds"])
                    video_data_plot = video_data.copy()
                    video_data_plot["trending_date"] = pd.to_datetime(video_data_plot["trending_date"])

                    # Clip predictions to reasonable values (no negative views)
                    forecast["yhat_clipped"] = forecast["yhat"].clip(lower=0)
                    forecast["yhat_upper_clipped"] = forecast["yhat_upper"].clip(lower=0)
                    forecast["yhat_lower_clipped"] = forecast["yhat_lower"].clip(lower=0)

                    fig = go.Figure()

                    # 1. Upper confidence bound (invisible line for fill reference)
                    fig.add_trace(go.Scatter(
                        x=forecast["ds"],
                        y=forecast["yhat_upper_clipped"],
                        mode="lines",
                        line=dict(width=0),
                        showlegend=False,
                        name="Upper"
                    ))

                    # 2. Lower confidence bound with fill to previous trace
                    fig.add_trace(go.Scatter(
                        x=forecast["ds"],
                        y=forecast["yhat_lower_clipped"],
                        mode="lines",
                        line=dict(width=0),
                        fill="tonexty",
                        fillcolor="rgba(173, 216, 230, 0.5)",
                        name="80% Confidence Interval"
                    ))

                    # 3. Predicted line (on top of confidence interval)
                    fig.add_trace(go.Scatter(
                        x=forecast["ds"],
                        y=forecast["yhat_clipped"],
                        mode="lines+markers",
                        line=dict(color="blue", width=2),
                        marker=dict(size=5),
                        name="Prophet Forecast"
                    ))

                    # 4. Actual historical data (most visible - on top)
                    fig.add_trace(go.Scatter(
                        x=video_data_plot["trending_date"],
                        y=video_data_plot["views"],
                        mode="lines+markers",
                        name="Actual Views",
                        line=dict(color="red", width=3),
                        marker=dict(size=10, color="red")
                    ))

                    # Layout
                    fig.update_layout(
                        title="Prophet View Forecast with 80% Confidence Interval",
                        xaxis_title="Date",
                        yaxis_title="Views",
                        yaxis=dict(rangemode="tozero"),
                        hovermode="x unified",
                        legend=dict(
                            orientation="h",
                            yanchor="bottom",
                            y=1.02,
                            xanchor="left",
                            x=0
                        ),
                        showlegend=True
                    )

                    st.plotly_chart(fig, width='stretch')

                    st.markdown("""
                    - **Red line with dots**: Actual historical views (training data)
                    - **Blue line**: Prophet's forecast
                    - **Light blue shaded area**: 80% confidence interval (uncertainty range)
                    """)

                    if is_unrealistic:
                        st.error("""
                        **Note**: Prophet generated unrealistic predictions (negative or extreme values).
                        This demonstrates why **RandomForest is the better model** for this dataset -
                        it uses video-specific features rather than just time series patterns.
                        """)

                    # Download button
                    st.download_button(
                        "Download Forecast CSV",
                        forecast.to_csv(index=False),
                        file_name=f"{video_id}_forecast.csv",
                        mime="text/csv"
                    )

                except Exception as e:
                    st.error(f"Error generating forecast: {str(e)}")
                    st.info("Try selecting a different video with more data points.")


# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
</div>
""", unsafe_allow_html=True)
