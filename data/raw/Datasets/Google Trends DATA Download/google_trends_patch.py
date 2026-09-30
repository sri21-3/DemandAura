import time
import pandas as pd
from pytrends.request import TrendReq

pytrends = TrendReq(hl="en-US", tz=360, retries=5, backoff_factor=3)

EXPANDED_CATEGORY_BATCHES = {
    "Nutrition_Diets": [
        ["diet", "nutrition", "vegan", "vegetarian", "plant based"],
        ["diet", "keto", "paleo", "low carb", "intermittent fasting"],
        ["diet", "detox", "superfood", "organic", "weight loss"],
        ["diet", "supplement", "protein powder", "whey", "creatine"],
        ["diet", "vitamin", "minerals", "probiotics", "functional food"],
    ],
    "Fitness_Wearables": [
        ["fitness", "exercise", "workout", "training", "gym"],
        ["fitness", "yoga", "pilates", "aerobics", "hiit"],
        ["fitness", "crossfit", "cardio", "running", "cycling"],
        ["fitness", "wearable", "smartwatch", "fitness tracker", "garmin"],
        ["fitness", "fitbit", "apple watch", "heart rate monitor", "step counter"],
    ],
    "Fashion_Beauty": [
        ["fashion", "clothing", "apparel", "style", "designer"],
        ["fashion", "luxury fashion", "fast fashion", "streetwear", "athleisure"],
        ["fashion", "skincare", "makeup", "cosmetics", "moisturizer"],
        ["fashion", "anti aging", "haircare", "shampoo", "fragrance"],
        ["fashion", "perfume", "beauty treatment", "sneakers", "jewelry"],
    ],
}

COUNTRY_MAP = {
    "US": "United_States",
    "IN": "India",
    "FR": "France",
    "NG": "Nigeria",
    "GB": "United_Kingdom",
    "CA": "Canada",
    "AU": "Australia",
    "BR": "Brazil",
    "MX": "Mexico",
    "ZA": "South_Africa",
    "KE": "Kenya",
    "SG": "Singapore",
    "AE": "United_Arab_Emirates",
    "CN": "China",
}

all_final_records = []

print("🚀 Starting Resilient Extraction with 8-Week Buffer Window...")

for geo_code, country_name in COUNTRY_MAP.items():
    for cat_name, batches in EXPANDED_CATEGORY_BATCHES.items():
        print(f"Fetching -> Country: {country_name:<20} | Category: {cat_name}")
        
        category_batch_dfs = []

        for batch_idx, keyword_batch in enumerate(batches):
            success = False
            attempts = 0

            while not success and attempts < 4:
                try:
                    attempts += 1
                    if attempts > 1:
                        time.sleep(15 * attempts)
                        pytrends = TrendReq(hl="en-US", tz=360, retries=5, backoff_factor=3)

                    # Fetching from 2022-09-01 to allow up to 8 weeks of clean lag calculation
                    pytrends.build_payload(
                        keyword_batch, geo=geo_code, timeframe="2022-09-01 2026-08-28"
                    )
                    df_raw = pytrends.interest_over_time()

                    if not df_raw.empty:
                        if "isPartial" in df_raw.columns:
                            df_raw = df_raw.drop(columns=["isPartial"])
                        
                        batch_cols = [kw for kw in keyword_batch if kw in df_raw.columns]
                        df_raw[f"batch_{batch_idx}_score"] = df_raw[batch_cols].mean(axis=1)
                        category_batch_dfs.append(df_raw[[f"batch_{batch_idx}_score"]])
                        success = True

                    time.sleep(6)

                except Exception as e:
                    print(f"  └─ ⚠️ Batch {batch_idx+1} attempt {attempts} failed: {e}")
                    time.sleep(20)

        if category_batch_dfs:
            df_cat_merged = pd.concat(category_batch_dfs, axis=1)
            df_cat_merged["Search_Interest"] = df_cat_merged.mean(axis=1)

            max_val = df_cat_merged["Search_Interest"].max()
            if max_val > 0:
                df_cat_merged["Search_Interest"] = (df_cat_merged["Search_Interest"] / max_val * 100).round(2)

            df_cat_merged = df_cat_merged.reset_index().rename(columns={"date": "Week_Start"})
            df_cat_merged["Country_Code"] = geo_code
            df_cat_merged["Country_Name"] = country_name
            df_cat_merged["Category"] = cat_name

            all_final_records.append(
                df_cat_merged[["Week_Start", "Country_Code", "Country_Name", "Category", "Search_Interest"]]
            )

if all_final_records:
    df_trends_full = pd.concat(all_final_records, ignore_index=True)
    df_trends_full["Week_Start"] = pd.to_datetime(df_trends_full["Week_Start"])
    
    df_trends_full = df_trends_full.sort_values(["Country_Name", "Category", "Week_Start"])

    print("\n⚙️ Calculating Velocity, Acceleration, and Up-to-8-Week Lags...")

    group_cols = ["Country_Name", "Category"]

    # 1. Momentum Metrics
    df_trends_full["Search_Velocity"] = df_trends_full.groupby(group_cols)["Search_Interest"].diff(1)
    df_trends_full["Search_Acceleration"] = df_trends_full.groupby(group_cols)["Search_Velocity"].diff(1)

    # 2. Search Interest Lags (1 to 8 weeks)
    for i in range(1, 9):
        df_trends_full[f"Search_Interest_Lag{i}"] = df_trends_full.groupby(group_cols)["Search_Interest"].shift(i)

    # 3. Search Velocity Lags (1 to 4 weeks)
    for i in range(1, 5):
        df_trends_full[f"Search_Velocity_Lag{i}"] = df_trends_full.groupby(group_cols)["Search_Velocity"].shift(i)

    # 4. Search Acceleration Lags (1 to 2 weeks)
    for i in range(1, 3):
        df_trends_full[f"Search_Acceleration_Lag{i}"] = df_trends_full.groupby(group_cols)["Search_Acceleration"].shift(i)

    # Drop only the initial 8-week buffer (Sept & Oct 2022) used strictly for lookback calculation, keeping Nov 2022 onward.
    
# 5. Baseline Interest (Rolling 8-week mean, lagged by 1 to prevent leakage)
    df_trends_full["Baseline_Interest_8w"] = (
        df_trends_full.groupby(group_cols)["Search_Interest"]
        .shift(1)
        .transform(lambda x: x.rolling(window=8, min_periods=8).mean())
    )

    df_trends_full = df_trends_full[df_trends_full["Week_Start"] >= "2022-11-01"]

    df_trends_full["Week_Start"] = df_trends_full["Week_Start"].dt.strftime("%Y-%m-%d")

    output_filename = "google_trends_8week_lags_clean_2022_2026.csv"
    df_trends_full.to_csv(output_filename, index=False)

    print("\n" + "="*65)
    print("✅ SUCCESS! 8-Week Lags Engineered and Data Fully Preserved.")
    print(f"Total Rows Extracted : {len(df_trends_full):,}")
    print(f"Saved To             : {output_filename}")
    print("="*65)
