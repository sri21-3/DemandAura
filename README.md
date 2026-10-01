# DemandAura  
#### 🌐 Macro-Level Demand Intelligence & Market Foresight Platform  

DemandAura is a **global market intelligence and demand forecasting platform** designed to analyze consumer intent versus media noise across **14 countries** and **3 core lifestyle categories**.  

Powered by **advanced machine learning models** and **real-time data pipelines**, DemandAura helps brands, strategists, and supply chain teams make **data-driven decisions** on expansion, ad spend, and inventory allocation.

---

## 🎯 Consumer Sectors
- 👗 Fashion & Beauty  
- 🏃 Fitness & Wearables  
- 🥗 Nutrition & Diets  

---

## 🌍 Countries Covered
Australia · Brazil · Canada · China · France · India · Kenya · Mexico · Nigeria · Singapore · South Africa · UAE · UK · USA  

---

## 📊 Market Categories & Data Keywords  

### 🥗 Nutrition & Diets  
- Diet, nutrition, vegan, vegetarian, plant based  
- Keto, paleo, low carb, intermittent fasting  
- Detox, superfood, organic, weight loss  
- Supplements, protein powder, whey, creatine  
- Vitamins, minerals, probiotics, functional food  

### 🏃 Fitness & Wearables  
- Fitness, exercise, workout, training, gym  
- Yoga, pilates, aerobics, HIIT  
- CrossFit, cardio, running, cycling  
- Wearables, smartwatch, fitness tracker, Garmin  
- Fitbit, Apple Watch, heart rate monitor, step counter  

### 👗 Fashion & Beauty  
- Fashion, clothing, apparel, style, designer  
- Luxury fashion, fast fashion, streetwear, athleisure  
- Skincare, makeup, cosmetics, moisturizer  
- Anti-aging, haircare, shampoo, fragrance  
- Perfume, beauty treatment, sneakers, jewelry  

---

## 🤖 Machine Learning Models  

| Model | Purpose | Business Impact |
|-------|---------|-----------------|
| **XGBoost Regression** (Demand vs. Hype Divergence) | Evaluates balance between search interest & media coverage | Prevents wasted ad spend, identifies underserved demand |
| **K-Means Clustering (3+ Years)** | Long-term strategic segmentation | Guides expansion, pricing, and global playbook |
| **K-Means Clustering (4 Weeks)** | Short-term momentum radar | Detects rapid trend breakouts & cooling categories |
| **LightGBM Time-Series Forecast** | Predicts 4-week search interest | Enables proactive inventory allocation |

---

## 🔄 Automated Workflow Diagram  


Automated Workflow Diagram

+-----------------------------------------------------------------------+
|                       GITHUB ACTIONS SCHEDULE                         |
|                    (Every Monday @ 3:00 AM IST)                       |
+-----------------------------------++----------------------------------+
                                    ||
                                    \/
+-----------------------------------------------------------------------+
|                       DATA INGESTION PIPELINE                         |
|                   (`incremental_fetch_data.ipynb`)                    |
|                                                                       |
|   +-----------------------+               +-----------------------+   |
|   |     Google Trends     |               |      GDELT GKG        |   |
|   |  (Search Intent Data) |               |  (Global News/Hype)   |   |
|   +-----------+-----------+               +-----------+-----------+   |
|               |                                       |               |
|               +-------------------+-------------------+               |
|                                   |                                   |
|                                   \/                                  |
|            +---------------------------------------------+            |
|            | Normalization & Feature Engineering Engine  |            |
|            +----------------------+----------------------+            |
+-----------------------------------|-----------------------------------+
                                    |
                                    \/
+-----------------------------------------------------------------------+
|                         STORAGE & PERSISTENCE                         |
|                                                                       |
|   +---------------------------------+  +--------------------------+   |
|   |      Cloudflare R2 Bucket       |  |      TiDB Database       |   |
|   |  (Model Features & Artifacts)   |  | (1-Week Incremental Data)|   |
|   +---------------------------------+  +--------------------------+   |
+-----------------------------------|-----------------------------------+
                                    |
                                    \/
+-----------------------------------------------------------------------+
|                          MACHINE LEARNING INFRA                       |
|                                                                       |
|  [XGBoost Divergence]   [K-Means 3Yr]   [K-Means 4Wk]   [LightGBM]    |
+-----------------------------------|-----------------------------------+
                                    |
                                    \/
+-----------------------------------------------------------------------+
|                          FASTAPI BACKEND SERVICE                      |
|                  (Exposes Analytical REST Endpoints)                  |
+-----------------------------------|-----------------------------------+
                                    |
                                    \/
+-----------------------------------------------------------------------+
|                          FRONTEND WEB APP                             |
|               (`DemandAura-Web-app` deployed on AI Studio)            |
+-----------------------------------------------------------------------+


---

## ⚙️ Pipeline Execution Details  
- **Trigger:** GitHub Actions (Every Monday, 3:00 AM IST)  
- **Ingestion:** Google Trends + GDELT GKG  
- **Normalization:** Rescales indices, rolling aggregations  
- **Storage:** Cloudflare R2 + TiDB  
- **Separation:** Strict temporal isolation between training & prediction  

---

## 🛠 Tech Stack  
- **Backend:** Python, FastAPI  
- **Pipeline:** Jupyter Notebooks, GitHub Actions  
- **ML:** XGBoost, LightGBM, scikit-learn, pandas, NumPy  
- **Databases:** TiDB, Cloudflare R2  
- **Frontend:** DemandAura-Web-app (AI Studio)  

---

## 🚦 Getting Started  

### Prerequisites  
- Python 3.9+  
- Cloudflare R2 & TiDB access keys  

### Installation  
```bash
git clone https://github.com/your-username/DemandAura.git
cd DemandAura

python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate

pip install -r requirements.txt
```


Environment Configuration:
Create a .env file in the root directory:

TIDB_HOST=<your-tidb-host>
TIDB_USER=<your-tidb-user>
TIDB_PASSWORD=<your-tidb-password>
R2_ACCOUNT_ID=<your-r2-account-id>
R2_ACCESS_KEY_ID=<your-r2-access-key>
R2_SECRET_ACCESS_KEY=<your-r2-secret-key>


Run the FastAPI Server locally:

uvicorn app.main:app --reload --port 8000


Access API documentation at http://localhost:8000/docs.

© DemandAura Platform
