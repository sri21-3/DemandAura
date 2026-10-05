import io
import os
import sys
import time
import logging
from datetime import datetime, timedelta

import boto3
import holidays
import numpy as np
import pandas as pd
import requests
from botocore.config import Config
from dotenv import load_dotenv
from google.cloud import bigquery
from pytrends.request import TrendReq
from sqlalchemy import create_engine, text, URL

# Setup Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# ------------------------------------------------------------------
# Environment & Database Connection Setup
# ------------------------------------------------------------------
load_dotenv(override=False)

def get_db_engine():
    db_host = os.getenv("DB_HOST")
    db_port = int(os.getenv("DB_PORT", 4000))
    db_user = os.getenv("DB_USERNAME")
    db_pass = os.getenv("DB_PASSWORD")
    db_name = os.getenv("DB_DATABASE")

    if not all([db_host, db_user, db_pass, db_name]):
        raise ValueError("Missing required database environment variables.")

    connection_url = URL.create(
        drivername="mysql+pymysql",
        username=db_user,
        password=db_pass,
        host=db_host,
        port=db_port,
        database=db_name,
    )

    connect_args = {}
    if db_host and "tidbcloud" in db_host:
        connect_args["ssl"] = {
            "ssl_verify_cert": True,
            "ssl_verify_identity": True,
            "ssl_ca": "/etc/ssl/certs/ca-certificates.crt"
        }

    return create_engine(connection_url, connect_args=connect_args)

# ------------------------------------------------------------------
# R2 Artifact Uploader (Independent Execution)
# ------------------------------------------------------------------
def upload_artifacts_to_r2(engine):
    logging.info("Starting independent Cloudflare R2 artifact export from TiDB...")
    
    query_full = text("""
    SELECT
        week_start, country_name, category, search_interest, search_velocity,
        search_acceleration, media_volume, tone_net_sentiment, tone_positive_score,
        tone_negative_score, tone_polarity, tone_activity_density, tone_self_group_density,
        source_diversity, demand_to_hype_ratio, inflation_rate, internet_penetration,
        gdp_per_capita, is_holiday, holiday_count
    FROM main
    ORDER BY country_name, category, week_start;
    """)

    with engine.connect() as connection:
        df_all = pd.read_sql(query_full, connection)

    if df_all.empty:
        logging.warning("Main database table is empty. Skipping R2 artifact export.")
        return

    df_all["week_start"] = pd.to_datetime(df_all["week_start"])
    grouped = df_all.groupby(["country_name", "category"])

    # Calculate Lags & Rolling Window Metrics
    for lag in [1, 2, 4]:
        df_all[f"search_interest_lag_{lag}"] = grouped["search_interest"].shift(lag)

    df_all["search_velocity_lag"] = grouped["search_velocity"].shift(1)
    df_all["search_velocity_lag1"] = grouped["search_velocity"].shift(1)
    df_all["search_velocity_lag2"] = grouped["search_velocity"].shift(2)
    df_all["search_acceleration_lag"] = grouped["search_acceleration"].shift(1)

    df_all["search_interest_roll_mean_4w"] = grouped["search_interest"].transform(lambda x: x.shift(1).rolling(4).mean())
    df_all["search_interest_roll_std_4w"] = grouped["search_interest"].transform(lambda x: x.shift(1).rolling(4).std())
    df_all["media_volume_roll_mean_4w"] = grouped["media_volume"].transform(lambda x: x.shift(1).rolling(4, min_periods=1).mean())
    df_all["log_media_volume"] = np.log1p(df_all["media_volume"])

    week_of_year = df_all["week_start"].dt.isocalendar().week
    df_all["sin_week"] = np.sin(2 * np.pi * week_of_year / 52)
    df_all["cos_week"] = np.cos(2 * np.pi * week_of_year / 52)

    # Compute Historical Market Segmentation Aggregates
    def calc_slope(group):
        group = group.sort_values("week_start")
        x = (group["week_start"] - group["week_start"].min()).dt.days // 7 + 1
        y = group["search_interest"]
        denom = np.sum((x - x.mean()) ** 2)
        return np.sum((x - x.mean()) * (y - y.mean())) / denom if denom != 0 else 0

    slopes = df_all.groupby(["country_name", "category"]).apply(calc_slope, include_groups=False).reset_index(name="search_interest_trend_slope")

    market_segmentation = df_all.groupby(["country_name", "category"]).agg(
        mean_search_interest=("search_interest", "mean"),
        mean_media_volume=("media_volume", "mean"),
        mean_demand_to_hype_ratio=("demand_to_hype_ratio", "mean"),
        mean_net_sentiment=("tone_net_sentiment", "mean"),
        mean_gdp_per_capita=("gdp_per_capita", "mean")
    ).reset_index().merge(slopes, on=["country_name", "category"])

    # Feature Datasets
    forecasting_cols = [
        'week_start', 'country_name', 'category', 'search_interest',
        'demand_to_hype_ratio', 'inflation_rate', 'gdp_per_capita',
        'holiday_count', 'search_interest_lag_1', 'search_interest_lag_2',
        'search_interest_lag_4', 'search_velocity_lag',
        'search_acceleration_lag', 'search_interest_roll_mean_4w',
        'search_interest_roll_std_4w', 'media_volume_roll_mean_4w', 'sin_week', 'cos_week'
    ]
    df_forecasting = df_all[forecasting_cols].copy()

    divergence_cols = [
        'country_name', 'category', 'search_velocity', 'search_acceleration',
        'tone_net_sentiment', 'tone_polarity', 'tone_activity_density',
        'source_diversity', 'inflation_rate', 'internet_penetration',
        'holiday_count', 'log_media_volume', 'search_velocity_lag1', 'search_velocity_lag2'
    ]
    df_divergence = df_all[divergence_cols].copy()

    latest_week = df_all["week_start"].max()
    df_final_incremental = df_all[df_all["week_start"] == latest_week].copy()

    # AWS/R2 Client Configuration
    R2_ACCOUNT_ID = os.getenv("R2_ACCOUNT_ID", "").strip()
    R2_ACCESS_KEY_ID = os.getenv("R2_ACCESS_KEY_ID", "").strip()
    R2_SECRET_ACCESS_KEY = os.getenv("R2_SECRET_ACCESS_KEY", "").strip()
    R2_BUCKET_NAME = os.getenv("R2_BUCKET_NAME", "").strip().strip("/")
    R2_ENDPOINT_URL = os.getenv("R2_ENDPOINT_URL", "").strip()

    endpoint_url = R2_ENDPOINT_URL if R2_ENDPOINT_URL else f"https://{R2_ACCOUNT_ID}.r2.cloudflarestorage.com"

    s3_client = boto3.client(
        "s3",
        endpoint_url=endpoint_url,
        aws_access_key_id=R2_ACCESS_KEY_ID,
        aws_secret_access_key=R2_SECRET_ACCESS_KEY,
        region_name="us-east-1",
        config=Config(signature_version="s3v4", s3={"addressing_style": "path"})
    )

    def upload_df(df_obj: pd.DataFrame, key_path: str):
        buf = io.StringIO()
        df_obj.to_csv(buf, index=False)
        s3_client.put_object(
            Bucket=R2_BUCKET_NAME,
            Key=key_path.lstrip("/"),
            Body=buf.getvalue().encode("utf-8"),
            ContentType="text/csv"
        )
        logging.info(f"✅ R2 Export Complete -> {R2_BUCKET_NAME}/{key_path}")

    upload_df(market_segmentation, "data/processed/market_segmentation.csv")
    upload_df(df_final_incremental, "data/processed/final_incremental_data.csv")
    upload_df(df_forecasting, "data/processed/Search_Intrest_forecasting_features.csv")
    upload_df(df_divergence, "data/processed/Demand_to_hype_Divergence_score_features.csv")

# ------------------------------------------------------------------
# Incremental Pipeline Main Routine
# ------------------------------------------------------------------
def run_pipeline():
    engine = get_db_engine()

    # 1. Fetch Target Extraction Range
    query_max_date = text("SELECT MAX(STR_TO_DATE(week_start, '%%Y-%%m-%%d')) AS last_date FROM main;")
    with engine.connect() as connection:
        last_date_df = pd.read_sql(query_max_date, connection)

    last_recorded_raw = last_date_df["last_date"].iloc[0]
    if pd.isna(last_recorded_raw):
        raise ValueError("Database table 'main' contains no recorded dates. Ingestion cannot proceed.")
    
    last_recorded_date = pd.to_datetime(last_recorded_raw).date()

    today = datetime.now().date()
    current_week_start = today - timedelta(days=(today.weekday() + 1) % 7)
    fetch_end_date = current_week_start - timedelta(days=7)

    if last_recorded_date >= fetch_end_date:
        logging.info(f"Data is up to date through {last_recorded_date}. Executing artifact refresh.")
        upload_artifacts_to_r2(engine)
        return

    NORMALIZATION_OVERLAP_WEEKS = 8
    fetch_start_date = last_recorded_date - timedelta(days=NORMALIZATION_OVERLAP_WEEKS * 7)

    timeframe_str = (
        f"{fetch_start_date.strftime('%Y-%m-%d')} "
        f"{fetch_end_date.strftime('%Y-%m-%d')}"
    )

    logging.info(f"Last recorded week : {last_recorded_date}")
    logging.info(f"Fetching timeframe  : {timeframe_str}")

    # 2. Google Trends Fetch
    COUNTRY_MAP = {
        "US": "United_States", "IN": "India", "FR": "France", "NG": "Nigeria",
        "GB": "United_Kingdom", "CA": "Canada", "AU": "Australia", "BR": "Brazil",
        "MX": "Mexico", "ZA": "South_Africa", "KE": "Kenya", "SG": "Singapore",
        "AE": "United_Arab_Emirates", "CN": "China"
    }

    EXPANDED_CATEGORY_BATCHES = {
        "Nutrition_Diets": [
            ["diet", "nutrition", "vegan", "vegetarian", "plant based"],
            ["keto", "paleo", "low carb", "intermittent fasting"],
            ["detox", "superfood", "organic", "weight loss"],
            ["supplement", "protein powder", "whey", "creatine"],
            ["vitamin", "minerals", "probiotics", "functional food"],
        ],
        "Fitness_Wearables": [
            ["fitness", "exercise", "workout", "training", "gym"],
            ["yoga", "pilates", "aerobics", "hiit"],
            ["crossfit", "cardio", "running", "cycling"],
            ["wearable", "smartwatch", "fitness tracker", "garmin"],
            ["fitbit", "apple watch", "heart rate monitor", "step counter"],
        ],
        "Fashion_Beauty": [
            ["fashion", "clothing", "apparel", "style", "designer"],
            ["luxury fashion", "fast fashion", "streetwear", "athleisure"],
            ["skincare", "makeup", "cosmetics", "moisturizer"],
            ["anti aging", "haircare", "shampoo", "fragrance"],
            ["perfume", "beauty treatment", "sneakers", "jewelry"],
        ],
    }

    pytrends = TrendReq(hl="en-US", tz=360, timeout=(10, 30), retries=3, backoff_factor=2)
    
    total_expected_batches = len(COUNTRY_MAP) * sum(len(b) for b in EXPANDED_CATEGORY_BATCHES.values())
    failed_batches_count = 0
    all_new_records = []

    for geo_code, country_name in COUNTRY_MAP.items():
        for cat_name, batches in EXPANDED_CATEGORY_BATCHES.items():
            category_batch_dfs = []
            for batch_idx, keyword_batch in enumerate(batches):
                success = False
                for attempt in range(1, 4):
                    try:
                        pytrends.build_payload(keyword_batch, geo=geo_code, timeframe=timeframe_str)
                        df_raw = pytrends.interest_over_time()

                        if not df_raw.empty:
                            if "isPartial" in df_raw.columns:
                                df_raw = df_raw[~df_raw["isPartial"]].drop(columns=["isPartial"])

                            batch_cols = [kw for kw in keyword_batch if kw in df_raw.columns]
                            if batch_cols:
                                # Sunday-aligned resampling matching GDELT dates
                                df_weekly = df_raw[batch_cols].resample("W-SUN", closed="left", label="left").mean().round(2)
                                score_col = f"batch_{batch_idx}_score"
                                df_weekly[score_col] = df_weekly[batch_cols].mean(axis=1)
                                category_batch_dfs.append(df_weekly[[score_col]])
                                success = True
                                break
                    except Exception as e:
                        logging.warning(f"Attempt {attempt} failed for {country_name} - {cat_name} batch {batch_idx}: {e}")
                        time.sleep(2 * attempt)

                if not success:
                    failed_batches_count += 1

            if category_batch_dfs:
                df_cat_merged = pd.concat(category_batch_dfs, axis=1)
                df_cat_merged["search_interest"] = df_cat_merged.mean(axis=1)
                max_val = df_cat_merged["search_interest"].max()
                if max_val > 0:
                    df_cat_merged["search_interest"] = (df_cat_merged["search_interest"] / max_val * 100).round(2)

                df_cat_merged = df_cat_merged.reset_index().rename(columns={"date": "week_start"})
                df_cat_merged["country_code"] = geo_code
                df_cat_merged["country_name"] = country_name
                df_cat_merged["category"] = cat_name
                all_new_records.append(df_cat_merged[["week_start", "country_code", "country_name", "category", "search_interest"]])

    if failed_batches_count / max(total_expected_batches, 1) > 0.10:
        raise RuntimeError(f"Pipeline aborted: Exceeded 10% failure threshold ({failed_batches_count}/{total_expected_batches} failed batches).")

    if not all_new_records:
        raise RuntimeError("No Google Trends records extracted.")

    df_incremental = pd.concat(all_new_records, ignore_index=True)
    df_incremental["week_start"] = pd.to_datetime(df_incremental["week_start"])

    # 3. Trends Coverage Assertion
    expected_series_count = len(COUNTRY_MAP) * len(EXPANDED_CATEGORY_BATCHES) # 42
    actual_series_count = df_incremental[["country_name", "category"]].drop_duplicates().shape[0]
    if actual_series_count < expected_series_count:
        raise RuntimeError(f"Google Trends coverage failure: Extracted {actual_series_count}/{expected_series_count} required series.")

    # 4. Read Full Baseline History
    query_baseline = text("""
    SELECT
        week_start, country_name, category, search_interest, media_volume,
        tone_net_sentiment, tone_positive_score, tone_negative_score,
        tone_polarity, tone_activity_density, tone_self_group_density,
        source_diversity, inflation_rate, internet_penetration, gdp_per_capita
    FROM main;
    """)
    with engine.connect() as connection:
        df_baseline = pd.read_sql(query_baseline, connection)

    df_baseline["week_start"] = pd.to_datetime(df_baseline["week_start"])
    df_baseline["search_interest"] = pd.to_numeric(df_baseline["search_interest"], errors="coerce")

    # 5. Overlap Count & Robust Dual-Level Normalization
    df_overlap = pd.merge(
        df_baseline[["week_start", "country_name", "category", "search_interest"]].rename(columns={"search_interest": "baseline_interest"}),
        df_incremental[["week_start", "country_name", "category", "search_interest"]].rename(columns={"search_interest": "new_interest"}),
        on=["week_start", "country_name", "category"],
        how="inner"
    )

    valid_overlap = df_overlap[(df_overlap["baseline_interest"] > 0) & (df_overlap["new_interest"] > 0)].copy()
    valid_overlap["Weekly_Ratio"] = valid_overlap["baseline_interest"] / valid_overlap["new_interest"]

    df_ratios = valid_overlap.groupby(["country_name", "category"], as_index=False).agg(
        Ratio=("Weekly_Ratio", "median"), Overlap_Count=("Weekly_Ratio", "count")
    )
    df_ratios_valid = df_ratios[df_ratios["Overlap_Count"] >= 3].copy()

    df_country_ratios = valid_overlap.groupby("country_name", as_index=False).agg(
        Country_Ratio=("Weekly_Ratio", "median")
    )

    df_scaled = pd.merge(df_incremental, df_ratios_valid[["country_name", "category", "Ratio"]], on=["country_name", "category"], how="left")
    df_scaled = pd.merge(df_scaled, df_country_ratios, on="country_name", how="left")
    
    # Dual-fallback strategy
    df_scaled["Ratio"] = df_scaled["Ratio"].fillna(df_scaled["Country_Ratio"]).fillna(1.0)
    df_scaled["search_interest"] = (df_scaled["search_interest"] * df_scaled["Ratio"]).round(2)
    df_incremental_normalized = df_scaled.drop(columns=["Ratio", "Country_Ratio", "country_code"])

    # Combine Preserving Baseline History First
    df_full = pd.concat([df_baseline, df_incremental_normalized], ignore_index=True)
    df_full["week_start"] = pd.to_datetime(df_full["week_start"])
    df_full = df_full.sort_values("week_start", kind="stable").drop_duplicates(
        subset=["country_name", "category", "week_start"], keep="first"
    ).reset_index(drop=True)

    # 6. GDELT Query (Scoped to New Weeks Only)
    client = bigquery.Client(project="gdelt-506803")
    
    gdelt_start_date = last_recorded_date + timedelta(days=7)
    gdelt_end_date = fetch_end_date + timedelta(days=6)

    gdelt_start_str = gdelt_start_date.strftime("%Y-%m-%d")
    gdelt_end_str = gdelt_end_date.strftime("%Y-%m-%d")

    bq_query = """
    SELECT
      DATE_TRUNC(PARSE_DATE('%Y%m%d', SUBSTR(CAST(DATE AS STRING), 1, 8)), WEEK(SUNDAY)) AS week_start,
      CASE
        WHEN V2Locations LIKE '%#US#%' THEN 'United_States'
        WHEN V2Locations LIKE '%#IN#%' THEN 'India'
        WHEN V2Locations LIKE '%#FR#%' THEN 'France'
        WHEN V2Locations LIKE '%#NI#%' THEN 'Nigeria'
        WHEN V2Locations LIKE '%#UK#%' THEN 'United_Kingdom'
        WHEN V2Locations LIKE '%#CA#%' THEN 'Canada'
        WHEN V2Locations LIKE '%#AS#%' THEN 'Australia'
        WHEN V2Locations LIKE '%#BR#%' THEN 'Brazil'
        WHEN V2Locations LIKE '%#MX#%' THEN 'Mexico'
        WHEN V2Locations LIKE '%#SF#%' THEN 'South_Africa'
        WHEN V2Locations LIKE '%#KE#%' THEN 'Kenya'
        WHEN V2Locations LIKE '%#SN#%' THEN 'Singapore'
        WHEN V2Locations LIKE '%#AE#%' THEN 'United_Arab_Emirates'
        WHEN V2Locations LIKE '%#CH#%' THEN 'China'
      END AS country_name,
      CASE
        WHEN REGEXP_CONTAINS(LOWER(Themes), r'diet|nutrition|vegan|vegetarian|plant-based|keto|paleo|low-carb|low-fat|gluten-free|sugar-free|fasting|detox|organic|supplement|protein|vitamin|weight loss') THEN 'Nutrition_Diets'
        WHEN REGEXP_CONTAINS(LOWER(Themes), r'fitness|exercise|workout|training|gym|yoga|pilates|crossfit|running|wearable|smartwatch|tracker|garmin|fitbit') THEN 'Fitness_Wearables'
        WHEN REGEXP_CONTAINS(LOWER(Themes), r'fashion|clothing|apparel|style|designer|luxury|streetwear|athleisure|cosmetic|skincare|makeup|fragrance|spa') THEN 'Fashion_Beauty'
        ELSE 'Other'
      END AS category,
      COUNT(*) AS media_volume_new,
      ROUND(AVG(SAFE_CAST(SPLIT(V2Tone, ',')[SAFE_OFFSET(0)] AS FLOAT64)), 2) AS tone_net_sentiment_new,
      ROUND(AVG(SAFE_CAST(SPLIT(V2Tone, ',')[SAFE_OFFSET(1)] AS FLOAT64)), 2) AS tone_positive_score_new,
      ROUND(AVG(SAFE_CAST(SPLIT(V2Tone, ',')[SAFE_OFFSET(2)] AS FLOAT64)), 2) AS tone_negative_score_new,
      ROUND(AVG(SAFE_CAST(SPLIT(V2Tone, ',')[SAFE_OFFSET(3)] AS FLOAT64)), 2) AS tone_polarity_new,
      ROUND(AVG(SAFE_CAST(SPLIT(V2Tone, ',')[SAFE_OFFSET(4)] AS FLOAT64)), 2) AS tone_activity_density_new,
      ROUND(AVG(SAFE_CAST(SPLIT(V2Tone, ',')[SAFE_OFFSET(5)] AS FLOAT64)), 2) AS tone_self_group_density_new,
      COUNT(DISTINCT SourceCommonName) AS source_diversity_new
    FROM `gdelt-bq.gdeltv2.gkg_partitioned`
    WHERE _PARTITIONTIME >= TIMESTAMP(@start_date)
      AND _PARTITIONTIME <= TIMESTAMP(@end_date)
      AND REGEXP_CONTAINS(LOWER(Themes), r'diet|nutrition|fitness|fashion')
      AND V2Tone IS NOT NULL
    GROUP BY week_start, country_name, category
    HAVING week_start IS NOT NULL AND country_name IS NOT NULL AND category != 'Other';
    """

    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("start_date", "STRING", gdelt_start_str),
            bigquery.ScalarQueryParameter("end_date", "STRING", gdelt_end_str),
        ],
        maximum_bytes_billed=10 * 1024**3 # 10 GB limit safety guard
    )

    df_gdelt = client.query(bq_query, job_config=job_config).to_dataframe()
    df_gdelt["week_start"] = pd.to_datetime(df_gdelt["week_start"])

    # Merge GDELT exclusively into target incremental rows
    df_full = pd.merge(df_full, df_gdelt, on=["week_start", "country_name", "category"], how="left")

    for col in ["media_volume", "tone_net_sentiment", "tone_positive_score", "tone_negative_score", "tone_polarity", "tone_activity_density", "tone_self_group_density", "source_diversity"]:
        new_col = f"{col}_new"
        if new_col in df_full.columns:
            df_full[col] = df_full[new_col].combine_first(df_full[col])
            df_full = df_full.drop(columns=[new_col])

    df_full["media_volume"] = df_full["media_volume"].fillna(0)
    df_full["source_diversity"] = df_full["source_diversity"].fillna(0)

    # 7. Group-Aware Features & Imputation
    df_full = df_full.sort_values(["country_name", "category", "week_start"]).reset_index(drop=True)
    grouped = df_full.groupby(["country_name", "category"])

    df_full["search_velocity"] = grouped["search_interest"].diff()
    df_full["search_acceleration"] = grouped["search_velocity"].diff()

    # Group-Aware Forward Fill for Sentiment
    sentiment_cols = [
        "tone_net_sentiment", "tone_positive_score", "tone_negative_score",
        "tone_polarity", "tone_activity_density", "tone_self_group_density"
    ]
    for col in sentiment_cols:
        df_full[col] = grouped[col].transform(lambda s: s.ffill())

    # 8. Holidays & World Bank Macro Fetch
    HOLIDAY_COUNTRY_MAP = {
        "United_States": "US", "India": "IN", "France": "FR", "Nigeria": "NG",
        "United_Kingdom": "GB", "Canada": "CA", "Australia": "AU", "Brazil": "BR",
        "Mexico": "MX", "South_Africa": "ZA", "Kenya": "KE", "Singapore": "SG",
        "United_Arab_Emirates": "AE", "China": "CN"
    }

    years_to_cover = df_full["week_start"].dt.year.unique()
    holiday_dict = {c: holidays.country_holidays(iso, years=years_to_cover) for c, iso in HOLIDAY_COUNTRY_MAP.items()}

    def count_holidays(row):
        hols = holiday_dict.get(row["country_name"], {})
        start = row["week_start"].date()
        return sum(1 for i in range(7) if (start + timedelta(days=i)) in hols)

    df_full["holiday_count"] = df_full.apply(count_holidays, axis=1)
    df_full["is_holiday"] = (df_full["holiday_count"] > 0).astype(int)

    # HTTPS World Bank Macro API Call
    COUNTRY_ISO_MAP = {
        "United_States": "USA", "India": "IND", "France": "FRA", "Nigeria": "NGA",
        "United_Kingdom": "GBR", "Canada": "CAN", "Australia": "AUS", "Brazil": "BRA",
        "Mexico": "MEX", "South_Africa": "ZAF", "Kenya": "KEN", "Singapore": "SGP",
        "United_Arab_Emirates": "ARE", "China": "CHN"
    }
    
    wb_records = []
    country_codes_str = ";".join(COUNTRY_ISO_MAP.values())
    for col_name, code in [("inflation_rate", "FP.CPI.TOTL.ZG"), ("internet_penetration", "IT.NET.USER.ZS"), ("gdp_per_capita", "NY.GDP.PCAP.CD")]:
        url = f"https://api.worldbank.org/v2/country/{country_codes_str}/indicator/{code}?date=2015:2026&format=json&per_page=1000"
        try:
            res = requests.get(url, timeout=15)
            if res.status_code == 200 and len(res.json()) > 1:
                for entry in res.json()[1]:
                    if entry.get("value") is not None:
                        wb_records.append({
                            "ISO3": entry["countryiso3code"],
                            "Year": int(entry["date"]),
                            "Metric": col_name,
                            "Value": float(entry["value"])
                        })
        except Exception as e:
            logging.warning(f"World Bank API fetch failed for {col_name}: {e}")

    if wb_records:
        df_wb = pd.DataFrame(wb_records)
        iso_to_c = {v: k for k, v in COUNTRY_ISO_MAP.items()}
        df_wb["country_name"] = df_wb["ISO3"].map(iso_to_c)
        df_macro = df_wb.sort_values("Year").groupby(["country_name", "Metric"])["Value"].last().unstack().reset_index()
        df_full = pd.merge(df_full.drop(columns=[c for c in df_macro.columns if c != "country_name"], errors="ignore"), df_macro, on="country_name", how="left")

    # Macro Group Forward Fill Fallback
    for m_col in ["inflation_rate", "internet_penetration", "gdp_per_capita"]:
        if m_col in df_full.columns:
            df_full[m_col] = grouped[m_col].transform(lambda s: s.ffill())

    df_full["demand_to_hype_ratio"] = df_full["search_interest"] / (df_full["media_volume"] + 1)

    # 9. Extract Incremental Weeks and Write to TiDB
    col_final = [
        'week_start', 'country_name', 'category', 'search_interest',
        'search_velocity', 'search_acceleration', 'media_volume',
        'tone_net_sentiment', 'tone_positive_score', 'tone_negative_score',
        'tone_polarity', 'tone_activity_density', 'tone_self_group_density',
        'source_diversity', 'demand_to_hype_ratio', 'inflation_rate',
        'internet_penetration', 'gdp_per_capita', 'is_holiday', 'holiday_count'
    ]

    target_week_df = df_full[df_full["week_start"] > pd.to_datetime(last_recorded_date)][col_final].copy()
    target_week_df["week_start"] = target_week_df["week_start"].dt.strftime("%Y-%m-%d")

    if target_week_df.empty:
        logging.info("No target incremental rows prepared.")
        return

    logging.info(f"Appending {len(target_week_df)} rows to TiDB 'main' table...")
    try:
        target_week_df.to_sql(name="main", con=engine, if_exists="append", index=False, chunksize=1000)
        logging.info("✅ Incremental append committed successfully to database.")
    except Exception as db_err:
        logging.error(f"Failed to append incremental rows to TiDB: {db_err}")
        raise db_err

    # 10. Trigger Artifact Export
    upload_artifacts_to_r2(engine)

if __name__ == "__main__":
    try:
        run_pipeline()
    except Exception as e:
        logging.error(f"Pipeline failure: {e}")
        sys.exit(1)