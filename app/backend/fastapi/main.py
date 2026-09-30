import io
import os
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

import boto3
import joblib
import numpy as np
import pandas as pd
from botocore.config import Config
from botocore.exceptions import BotoCoreError, ClientError
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# ------------------------------------------------------------------
# 1. Environment & Cloudflare R2 Configuration
# ------------------------------------------------------------------
load_dotenv(override=True)

R2_ACCOUNT_ID = os.getenv("R2_ACCOUNT_ID", "").strip()
R2_ACCESS_KEY_ID = os.getenv("R2_ACCESS_KEY_ID", "").strip()
R2_SECRET_ACCESS_KEY = os.getenv("R2_SECRET_ACCESS_KEY", "").strip()
R2_BUCKET_NAME = os.getenv("R2_BUCKET_NAME", "").strip().strip("/")

required_values = {
    "R2_ACCOUNT_ID": R2_ACCOUNT_ID,
    "R2_ACCESS_KEY_ID": R2_ACCESS_KEY_ID,
    "R2_SECRET_ACCESS_KEY": R2_SECRET_ACCESS_KEY,
    "R2_BUCKET_NAME": R2_BUCKET_NAME,
}

missing = [name for name, value in required_values.items() if not value]
if missing:
    raise ValueError(f"❌ Missing R2 environment variables: {', '.join(missing)}")

s3_client = boto3.client(
    "s3",
    endpoint_url=f"https://{R2_ACCOUNT_ID}.r2.cloudflarestorage.com",
    aws_access_key_id=R2_ACCESS_KEY_ID,
    aws_secret_access_key=R2_SECRET_ACCESS_KEY,
    region_name="auto",
    config=Config(signature_version="s3v4"),
)


# ------------------------------------------------------------------
# 2. Object Keys in R2
# ------------------------------------------------------------------
R2_XGB_MODEL_KEY = "pickles/xgb_divergence_model.pkl"
R2_OHE_ENCODER_KEY = "pickles/onehot_encoder.pkl"
R2_REG_SCALER_KEY = "pickles/scaler.pkl"
R2_LGB_FORECAST_MODEL_KEY = "pickles/lightgbm_search_interest_forecast_model.pkl"
R2_LGB_FORECAST_META_KEY = "pickles/lightgbm_search_interest_forecast_model_metadata.pkl"

R2_DIVERGENCE_CSV_KEY = "data/processed/Demand_to_hype_Divergence_score_features.csv"
R2_CLUSTER_CSV_KEY = "data/processed/market_segmentation.csv"
R2_FORECAST_CSV_KEY = "data/processed/Search_Intrest_forecasting_features.csv"
R2_4_W_CLUSTER_CSV_KEY = "data/processed/4_weeks_market_segmentation.csv"

R2_CLUSTER_MD_KEY = "reports/market_segmentation.md"
R2_4W_CLUSTER_MD_KEY = "reports/4_weeks_market_segmentation.md"


# ------------------------------------------------------------------
# 3. Helper Functions for Cloudflare R2 Loading
# ------------------------------------------------------------------
def load_pkl_from_r2(r2_key: str) -> Any:
    """Fetch and deserialize a joblib pickle object directly from R2 into memory."""
    clean_key = r2_key.lstrip("/")
    try:
        print(f"⏳ Fetching pickle '{clean_key}' from R2...", end="", flush=True)
        response = s3_client.get_object(Bucket=R2_BUCKET_NAME, Key=clean_key)
        buffer = io.BytesIO(response["Body"].read())
        obj = joblib.load(buffer)
        print(" ✅ Loaded!")
        return obj
    except (BotoCoreError, ClientError) as e:
        print(f"\n❌ Cloudflare R2 Client Error loading pickle '{clean_key}': {e}")
        raise RuntimeError(f"Failed to fetch '{clean_key}' from storage.") from e
    except Exception as e:
        print(f"\n❌ Deserialization error for pickle '{clean_key}': {e}")
        raise RuntimeError(f"Failed to deserialize object '{clean_key}'.") from e


def load_csv_from_r2(r2_key: str, **pandas_kwargs) -> pd.DataFrame:
    """Fetch and load a CSV object directly into a pandas DataFrame from R2."""
    clean_key = r2_key.lstrip("/")
    try:
        print(f"⏳ Fetching CSV '{clean_key}' from R2...", end="", flush=True)
        response = s3_client.get_object(Bucket=R2_BUCKET_NAME, Key=clean_key)
        buffer = io.BytesIO(response["Body"].read())
        df = pd.read_csv(buffer, **pandas_kwargs)
        print(f" ✅ Loaded! Shape: {df.shape}")
        return df
    except (BotoCoreError, ClientError) as e:
        print(f"\n❌ Cloudflare R2 Client Error loading CSV '{clean_key}': {e}")
        raise RuntimeError(f"Failed to fetch CSV '{clean_key}' from storage.") from e
    except Exception as e:
        print(f"\n❌ Parsing error for CSV '{clean_key}': {e}")
        raise RuntimeError(f"Failed to parse CSV '{clean_key}'.") from e


def load_text_from_r2(r2_key: str) -> str:
    """Fetch raw markdown/text files from Cloudflare R2."""
    clean_key = r2_key.lstrip("/")
    try:
        print(f"⏳ Fetching text/markdown '{clean_key}' from R2...", end="", flush=True)
        response = s3_client.get_object(Bucket=R2_BUCKET_NAME, Key=clean_key)
        content = response["Body"].read().decode("utf-8")
        print(" ✅ Loaded!")
        return content
    except Exception as e:
        print(f"\n❌ Error loading markdown '{clean_key}' from R2: {e}")
        return ""


# Global in-memory memory caches
ml_artifacts: Dict[str, Any] = {}
ml_dataframes: Dict[str, pd.DataFrame] = {}
ml_reports: Dict[str, str] = {}


# ------------------------------------------------------------------
# 4. Consolidated FastAPI Lifespan Handler
# ------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Handles complete startup model/dataset loading and shutdown cleanup.
    """
    print("\n🚀 Starting up Demand Planning & Forecasting API...")
    print("--------------------------------------------------")

    # 1. Load ML Models and Artifacts
    ml_artifacts["xgb_model"] = load_pkl_from_r2(R2_XGB_MODEL_KEY)
    ml_artifacts["onehot_encoder"] = load_pkl_from_r2(R2_OHE_ENCODER_KEY)
    ml_artifacts["reg_scaler"] = load_pkl_from_r2(R2_REG_SCALER_KEY)
    ml_artifacts["lgb_forecast_model"] = load_pkl_from_r2(R2_LGB_FORECAST_MODEL_KEY)
    ml_artifacts["lgb_forecast_metadata"] = load_pkl_from_r2(R2_LGB_FORECAST_META_KEY)

    # 2. Load CSV Datasets
    ml_dataframes["df_divergence"] = load_csv_from_r2(R2_DIVERGENCE_CSV_KEY)
    ml_dataframes["df_cluster"] = load_csv_from_r2(R2_CLUSTER_CSV_KEY)
    ml_dataframes["df_cluster_4w"] = load_csv_from_r2(R2_4_W_CLUSTER_CSV_KEY)
    ml_dataframes["df_forecast"] = load_csv_from_r2(R2_FORECAST_CSV_KEY)

    # 3. Load Markdown Cluster Reports
    ml_reports["cluster_md"] = load_text_from_r2(R2_CLUSTER_MD_KEY)
    ml_reports["cluster_4w_md"] = load_text_from_r2(R2_4W_CLUSTER_MD_KEY)

    print("--------------------------------------------------")
    print("✅ All artifacts successfully cached in memory!\n")

    yield

    # Release memory on shutdown
    ml_artifacts.clear()
    ml_dataframes.clear()
    ml_reports.clear()
    print("🧹 Memory cleared on application shutdown.")


# ------------------------------------------------------------------
# 5. App Initialization
# ------------------------------------------------------------------
app = FastAPI(
    title="Demand Planning & Forecasting API",
    description="API for market divergence predictions, segmenting markets, and forecasting multi-week Search Interest based on Cloudflare R2 objects.",
    version="4.0.0",
    lifespan=lifespan,
)

@app.get("/", tags=["Root"])
def read_root():
    return {
        "message": "Welcome to DemandAura API — Sense the Future of Global Consumer Demand, in Fashion & Beauty, Fitness & Wearables, Nutrition & Diets ",
        "status": "online",
        "documentation": "/docs",
        "health_check": "/health",
    }

@app.get("/health", tags=["Health Check"])
def health_check():
    """Verify that all models, dataframes, and reports are present in RAM."""
    return {
        "status": "healthy",
        "loaded_artifacts": list(ml_artifacts.keys()),
        "loaded_dataframes": list(ml_dataframes.keys()),
        "loaded_reports": list(ml_reports.keys()),
    }


# ------------------------------------------------------------------
# 6. Schemas & Feature Constants
# ------------------------------------------------------------------
class DivergenceRequest(BaseModel):
    country_name: str = Field(
        ...,
        description="Target country name (e.g., 'United_States', 'India')",
        json_schema_extra={"examples": ["United_States"]},
    )
    category: str = Field(
        ...,
        description="Product or market category (e.g., 'Fashion_Beauty')",
        json_schema_extra={"examples": ["Fashion_Beauty"]},
    )


class DivergenceResponse(BaseModel):
    country_name: str
    category: str
    record_date: str
    predicted_divergence_score: float
    status: str = "success"


NUMERICAL_COLS_TO_SCALE = [
    "search_velocity",
    "search_acceleration",
    "tone_net_sentiment",
    "tone_polarity",
    "tone_activity_density",
    "source_diversity",
    "inflation_rate",
    "internet_penetration",
    "holiday_count",
    "search_velocity_lag1",
    "search_velocity_lag2",
]

EXPECTED_MODEL_FEATURES = [
    "search_velocity",
    "search_acceleration",
    "tone_net_sentiment",
    "tone_polarity",
    "tone_activity_density",
    "source_diversity",
    "inflation_rate",
    "internet_penetration",
    "holiday_count",
    "log_media_volume",
    "search_velocity_lag1",
    "search_velocity_lag2",
    "country_name_Australia",
    "country_name_Brazil",
    "country_name_Canada",
    "country_name_China",
    "country_name_France",
    "country_name_India",
    "country_name_Kenya",
    "country_name_Mexico",
    "country_name_Nigeria",
    "country_name_Singapore",
    "country_name_South_Africa",
    "country_name_United_Arab_Emirates",
    "country_name_United_Kingdom",
    "country_name_United_States",
    "category_Fashion_Beauty",
    "category_Fitness_Wearables",
    "category_Nutrition_Diets",
]


class ForecastRequest(BaseModel):
    country_name: str = Field(
        ...,
        description="Target country name (e.g., 'United_States', 'India')",
        json_schema_extra={"examples": ["United_States"]},
    )
    category: str = Field(
        ...,
        description="Product or market category (e.g., 'Fashion_Beauty')",
        json_schema_extra={"examples": ["Fashion_Beauty"]},
    )


class WeeklyForecastPoint(BaseModel):
    week_date: str
    predicted_search_interest: float


class ForecastResponse(BaseModel):
    country_name: str
    category: str
    target_variable: str
    forecast_points: List[WeeklyForecastPoint]
    status: str = "success"


EXPECTED_FORECAST_FEATURES = [
    "country_name",
    "category",
    "demand_to_hype_ratio",
    "inflation_rate",
    "gdp_per_capita",
    "holiday_count",
    "search_interest_lag_1",
    "search_interest_lag_2",
    "search_interest_lag_4",
    "search_velocity_lag",
    "search_acceleration_lag",
    "search_interest_roll_mean_4w",
    "search_interest_roll_std_4w",
    "media_volume_roll_mean_4w",
    "sin_week",
    "cos_week",
]


class TableDataResponse(BaseModel):
    total_records: int
    columns: List[str]
    data: List[Dict[str, Any]]


class MarkdownReportResponse(BaseModel):
    report_title: str
    content: str


# ------------------------------------------------------------------
# 7. Prediction & Forecasting Endpoints
# ------------------------------------------------------------------
@app.post(
    "/predict/divergence",
    response_model=DivergenceResponse,
    tags=["Predictions"],
    summary="Predict Demand vs. Hype Divergence Score",
)
def predict_divergence(payload: DivergenceRequest):
    df_divergence = ml_dataframes.get("df_divergence")
    xgb_model = ml_artifacts.get("xgb_model")
    scaler = ml_artifacts.get("reg_scaler")
    encoder = ml_artifacts.get("onehot_encoder")

    if (
        df_divergence is None
        or xgb_model is None
        or scaler is None
        or encoder is None
    ):
        raise HTTPException(
            status_code=500,
            detail="Required ML artifacts or datasets are not loaded in memory.",
        )

    filtered_df = df_divergence[
        (
            df_divergence["country_name"].astype(str).str.lower()
            == payload.country_name.lower()
        )
        & (
            df_divergence["category"].astype(str).str.lower()
            == payload.category.lower()
        )
    ].copy()

    if filtered_df.empty:
        raise HTTPException(
            status_code=404,
            detail=f"No baseline data found for country_name='{payload.country_name}' and category='{payload.category}'.",
        )

    date_col = next(
        (
            col
            for col in ["date", "timestamp", "dt"]
            if col in filtered_df.columns
        ),
        None,
    )
    if date_col:
        filtered_df[date_col] = pd.to_datetime(filtered_df[date_col])
        recent_record = filtered_df.sort_values(
            by=date_col, ascending=False
        ).iloc[0]
        record_date_str = str(recent_record[date_col].date())
    else:
        recent_record = filtered_df.iloc[-1]
        record_date_str = "latest_available"

    try:
        num_features = recent_record[NUMERICAL_COLS_TO_SCALE].values.reshape(
            1, -1
        )
        scaled_num_features = scaler.transform(num_features)
        scaled_num_df = pd.DataFrame(
            scaled_num_features, columns=NUMERICAL_COLS_TO_SCALE
        )
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error scaling numerical features: {str(e)}"
        )

    try:
        cat_df = pd.DataFrame(
            [
                {
                    "country_name": payload.country_name,
                    "category": payload.category,
                }
            ]
        )
        encoded_cat_features = encoder.transform(cat_df)

        if hasattr(encoded_cat_features, "toarray"):
            encoded_cat_features = encoded_cat_features.toarray()

        encoded_cat_cols = encoder.get_feature_names_out(
            ["country_name", "category"]
        )
        encoded_cat_df = pd.DataFrame(
            encoded_cat_features, columns=encoded_cat_cols
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error one-hot encoding categories: {str(e)}",
        )

    unscaled_cols = [
        col
        for col in EXPECTED_MODEL_FEATURES
        if col not in NUMERICAL_COLS_TO_SCALE and col not in encoded_cat_cols
    ]
    unscaled_dict = {}
    for col in unscaled_cols:
        if col in recent_record:
            unscaled_dict[col] = recent_record[col]
        else:
            unscaled_dict[col] = 0.0

    unscaled_df = pd.DataFrame([unscaled_dict])
    full_feature_df = pd.concat(
        [scaled_num_df, encoded_cat_df, unscaled_df], axis=1
    )

    try:
        model_input = full_feature_df[EXPECTED_MODEL_FEATURES].values
    except KeyError as e:
        raise HTTPException(
            status_code=500,
            detail=f"Feature alignment mismatch. Missing column: {str(e)}",
        )

    try:
        prediction = xgb_model.predict(model_input)[0]
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"XGBoost model prediction failed: {str(e)}"
        )

    return DivergenceResponse(
        country_name=payload.country_name,
        category=payload.category,
        record_date=record_date_str,
        predicted_divergence_score=float(prediction),
    )


@app.post(
    "/forecast/search-interest",
    response_model=ForecastResponse,
    tags=["Predictions"],
    summary="Forecast Multi-Week Search Interest using LightGBM",
)
@app.post(
    "/forecast/search-interest",
    response_model=ForecastResponse,
    tags=["Predictions"],
    summary="Forecast Multi-Week Search Interest using LightGBM",
)
def forecast_search_interest(payload: ForecastRequest):
    df_forecast = ml_dataframes.get("df_forecast")
    lgb_model = ml_artifacts.get("lgb_forecast_model")
    metadata = ml_artifacts.get("lgb_forecast_metadata")

    if df_forecast is None or lgb_model is None or metadata is None:
        raise HTTPException(
            status_code=500,
            detail="Forecast model or features dataset not loaded in memory.",
        )

    # 1. Filter dataset for target entity
    filtered_df = df_forecast[
        (df_forecast["country_name"].astype(str).str.lower() == payload.country_name.lower())
        & (df_forecast["category"].astype(str).str.lower() == payload.category.lower())
    ].copy()

    if filtered_df.empty:
        raise HTTPException(
            status_code=404,
            detail=f"No forecasting baseline features found for country_name='{payload.country_name}' and category='{payload.category}'.",
        )

    # 2. Extract latest date baseline
    date_col = next(
        (col for col in ["date", "timestamp", "dt", "week_start"] if col in filtered_df.columns),
        None,
    )
    
    if date_col:
        filtered_df[date_col] = pd.to_datetime(filtered_df[date_col])
        filtered_df = filtered_df.sort_values(by=date_col, ascending=True)
        latest_date = filtered_df[date_col].max()
    else:
        latest_date = pd.Timestamp.now()

    # Get latest row as starting feature state
    last_record = filtered_df.iloc[-1].copy()

    feature_list = metadata.get("features", EXPECTED_FORECAST_FEATURES)
    cat_cols = metadata.get("categorical_features", ["country_name", "category"])
    cat_categories = metadata.get("categorical_categories", {})

    # 3. Recursive Forecasting over 4 Future Horizon Steps
    forecast_points = []
    current_features = last_record.copy()

    for step in range(1, 5):
        # Generate future step date
        future_date = latest_date + pd.Timedelta(weeks=step)
        
        # Construct single-row feature DataFrame
        input_row = pd.DataFrame([current_features[feature_list]])

        # Align categorical variables for LightGBM
        for col in cat_cols:
            if col in input_row.columns:
                if col in cat_categories:
                    input_row[col] = pd.Categorical(input_row[col], categories=cat_categories[col])
                else:
                    input_row[col] = input_row[col].astype("category")

        # Predict future value
        try:
            pred_val = float(lgb_model.predict(input_row)[0])
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"LightGBM inference failed at step {step}: {str(e)}",
            )

        # Update cyclical week signals if present in dataset
        if "sin_week" in current_features and "cos_week" in current_features:
            week_num = future_date.isocalendar().week
            current_features["sin_week"] = np.sin(2 * np.pi * week_num / 52.0)
            current_features["cos_week"] = np.cos(2 * np.pi * week_num / 52.0)

        # Update lag features iteratively for step + 1
        if "search_interest_lag_4" in current_features and "search_interest_lag_2" in current_features:
            current_features["search_interest_lag_4"] = current_features["search_interest_lag_2"]
        if "search_interest_lag_2" in current_features and "search_interest_lag_1" in current_features:
            current_features["search_interest_lag_2"] = current_features["search_interest_lag_1"]
        if "search_interest_lag_1" in current_features:
            current_features["search_interest_lag_1"] = pred_val

        date_str = str(future_date.date()) if date_col else f"Week_{step}"

        forecast_points.append(
            WeeklyForecastPoint(
                week_date=date_str,
                predicted_search_interest=round(pred_val, 4),
            )
        )

    target_name = metadata.get("target", "search_interest")

    return ForecastResponse(
        country_name=payload.country_name,
        category=payload.category,
        target_variable=target_name if isinstance(target_name, str) else "search_interest",
        forecast_points=forecast_points,
    )

# ------------------------------------------------------------------
# 8. Clustering Data & Report Endpoints
# ------------------------------------------------------------------
@app.get(
    "/clusters/market-segmentation",
    response_model=TableDataResponse,
    tags=["Clustering & Segmentation"],
    summary="Get Market Segmentation tabular data",
)
def get_market_segmentation(
    country_name: Optional[str] = None, category: Optional[str] = None
):
    df = ml_dataframes.get("df_cluster")
    if df is None:
        raise HTTPException(
            status_code=500, detail="Market segmentation dataset not loaded."
        )

    filtered_df = df.copy()

    if country_name:
        filtered_df = filtered_df[
            filtered_df["country_name"].astype(str).str.lower()
            == country_name.lower()
        ]
    if category:
        filtered_df = filtered_df[
            filtered_df["category"].astype(str).str.lower() == category.lower()
        ]

    filtered_df = filtered_df.replace(
        {np.nan: None, np.inf: None, -np.inf: None}
    )

    return TableDataResponse(
        total_records=len(filtered_df),
        columns=list(filtered_df.columns),
        data=filtered_df.to_dict(orient="records"),
    )


@app.get(
    "/clusters/4-weeks-segmentation",
    response_model=TableDataResponse,
    tags=["Clustering & Segmentation"],
    summary="Get 4-Week Market Segmentation tabular data",
)
def get_4w_market_segmentation(
    country_name: Optional[str] = None, category: Optional[str] = None
):
    df = ml_dataframes.get("df_cluster_4w")
    if df is None:
        raise HTTPException(
            status_code=500, detail="4-Week Market segmentation dataset not loaded."
        )

    filtered_df = df.copy()

    if country_name:
        filtered_df = filtered_df[
            filtered_df["country_name"].astype(str).str.lower()
            == country_name.lower()
        ]
    if category:
        filtered_df = filtered_df[
            filtered_df["category"].astype(str).str.lower() == category.lower()
        ]

    filtered_df = filtered_df.replace(
        {np.nan: None, np.inf: None, -np.inf: None}
    )

    return TableDataResponse(
        total_records=len(filtered_df),
        columns=list(filtered_df.columns),
        data=filtered_df.to_dict(orient="records"),
    )


@app.get(
    "/reports/market-segmentation",
    response_model=MarkdownReportResponse,
    tags=["Reports"],
    summary="Get Markdown report view for Market Segmentation",
)
def get_market_segmentation_report():
    report_content = ml_reports.get("cluster_md", "")
    if not report_content:
        raise HTTPException(
            status_code=404, detail="Market segmentation report not found."
        )

    return MarkdownReportResponse(
        report_title="Overall Market Segmentation Report",
        content=report_content,
    )


@app.get(
    "/reports/4-weeks-segmentation",
    response_model=MarkdownReportResponse,
    tags=["Reports"],
    summary="Get Markdown report view for 4-Week Market Segmentation",
)
def get_4w_market_segmentation_report():
    report_content = ml_reports.get("cluster_4w_md", "")
    if not report_content:
        raise HTTPException(
            status_code=404, detail="4-Week Market segmentation report not found."
        )

    return MarkdownReportResponse(
        report_title="4-Week Market Segmentation Report",
        content=report_content,
    )


# ------------------------------------------------------------------
# 9. Server Execution
# ------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)