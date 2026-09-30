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

print("🚀 Starting Raw Data Extraction (No Lags)...")

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
    df_raw_full = pd.concat(all_final_records, ignore_index=True)
    df_raw_full["Week_Start"] = pd.to_datetime(df_raw_full["Week_Start"]).dt.strftime("%Y-%m-%d")
    
    raw_filename = "google_trends_raw_2022_2026.csv"
    df_raw_full.to_csv(raw_filename, index=False)
    
    print("\n" + "="*65)
    print(f"✅ RAW DATA SAVED: {raw_filename} ({len(df_raw_full):,} rows)")
    print("="*65)