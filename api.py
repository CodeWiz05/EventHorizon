from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
import joblib
import os
import numpy as np
import pandas as pd
import datetime
import subprocess
import yfinance as yf
import asyncio
from scipy.spatial import distance
from src.data_engineering import MarketDataEngineer

app = FastAPI(title="Regime Intelligence Engine API", version="1.0")

# Enable CORS for the React Frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
model_cache = {}

async def run_training_script():
    print("🚀 MLOPS: Starting background retraining job...")
    process = await asyncio.create_subprocess_exec(
        "python", "main.py",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE
    )
    
    stdout, stderr = await process.communicate()
    
    if process.returncode == 0:
        print("✅ MLOPS: Retraining complete. Artifacts updated.")
    else:
        print(f"❌ MLOPS: Retraining failed.\n{stderr.decode()}")

def get_latest_artifact(ticker: str):
    safe_ticker = ticker.replace("^", "").replace("=", "_")
    file_path = f"api_models/{safe_ticker}_artifact.joblib"
    
    # Check if file exists
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail=f"No model found for {ticker}")
    
    # Get last modified time
    mtime = os.path.getmtime(file_path)
    
    # If not in cache or file was updated, reload it
    if safe_ticker not in model_cache or model_cache[safe_ticker]['mtime'] < mtime:
        print(f"🔄 MLOPS: Detected new artifact for {ticker}. Hot-reloading...")
        model_cache[safe_ticker] = {
            'data': joblib.load(file_path),
            'mtime': mtime
        }
    
    return model_cache[safe_ticker]['data']

def load_artifact(ticker: str):
    safe_ticker = ticker.replace("^", "").replace("=", "_")
    path = f"api_models/{safe_ticker}_artifact.joblib"
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail=f"Model artifact not found for {ticker}")
    return joblib.load(path)

@app.get("/")
def read_root():
    return {"status": "Regime Intelligence Engine is Online"}

@app.get("/predict/{ticker}")
def get_prediction(ticker: str):
    try:
        # 1. Load the frozen artifact
        artifact = get_latest_artifact(ticker)
        
        # 2. Fetch live data (Last 365 days ensures enough padding)
        end_date = datetime.date.today().strftime("%Y-%m-%d")
        start_date = (datetime.date.today() - datetime.timedelta(days=365)).strftime("%Y-%m-%d")
        
        engineer = MarketDataEngineer(ticker=ticker, start_date=start_date, end_date=end_date)
        raw_df = engineer.run_pipeline()
        
        if raw_df.empty:
            raise HTTPException(status_code=500, detail="Yahoo Finance returned empty data.")
            
        # 3. Extract and Scale ALL initial features (Scaler expects the superset)
        initial_features = engineer.feature_names
        X_raw_all = raw_df[initial_features].values
        scaler = artifact["scaler"]
        X_scaled_all = scaler.transform(X_raw_all)
        
        # 4. Mahalanobis Anomaly Detection (Detector expects the superset)
        detector = artifact["anomaly_detector"]
        latest_X_all = X_scaled_all[-1]
        dist = distance.mahalanobis(latest_X_all, detector["mean"], detector["inv_cov"])
        is_shock = (dist ** 2) > detector["threshold"]
        
        # 5. Slice the FINAL features specifically for the HMM
        final_features = artifact["features"]
        feature_indices = [initial_features.index(f) for f in final_features]
        X_scaled_final = X_scaled_all[:, feature_indices]
        
        # 6. Causal HMM Forward Pass
        hmm = artifact["hmm_model"]
        shock_label = artifact["shock_label"]
        current_probs = np.zeros(shock_label + 1)
        
        if is_shock:
            current_probs[shock_label] = 1.0
            current_state = shock_label
        else:
            # Detect historical anomalies in the live window to clean the sequence
            is_outlier_array = np.array([(distance.mahalanobis(row, detector["mean"], detector["inv_cov"]) ** 2) > detector["threshold"] for row in X_scaled_all])
            X_clean = X_scaled_final[~is_outlier_array]
            
            if len(X_clean) > 0:
                filtering_probs = hmm.predict_filtering_proba(X_clean)
                current_probs[:shock_label] = filtering_probs[-1]
                current_state = int(np.argmax(current_probs[:shock_label]))
            else:
                current_probs[shock_label] = 1.0
                current_state = shock_label
                
        # 7. T+1 Forecasting (P_{t+1} = P_t * TPM)
        forecast_probs = np.zeros(shock_label + 1)
        if is_shock:
            # SHOCK DECAY DYNAMICS: 100% certainty is unscientific.
            # We assign a high persistence prior, but allow probability mass to diffuse 
            # into adjacent high-volatility states (Bear/Chop) for T+1.
            forecast_probs[shock_label] = 0.85  # 85% chance tail-risk persists tomorrow
            forecast_probs[0] = 0.12            # 12% chance it digests into standard Bear
            forecast_probs[1] = 0.03            # 3% chance it stabilizes to Chop
            forecast_probs[2] = 0.00            # 0% chance of immediate V-shape Bull recovery
        else:
            tpm = hmm.transmat_
            forecast_probs[:shock_label] = np.dot(current_probs[:shock_label], tpm)
            
        # 8. Map to Optimal Exposure Policy
        optimal_weights = artifact["policy_weights"]
        target_exposure = sum(forecast_probs[i] * optimal_weights[i] for i in range(shock_label + 1))
        
        # 9. Generate Human-Readable Signal
        if is_shock:
            signal = "CRITICAL: Tail-Risk Event Detected. Maximize Hedging."
        elif target_exposure >= 0.8:
            signal = "Risk-On: Maintain Growth Exposure."
        elif target_exposure <= 0.3:
            signal = "Risk-Off: High Volatility/Deteriorating Regime. Reduce Exposure."
        else:
            signal = "Neutral: Choppy Regime. Maintain Partial/Hedged Exposure."

        # Format output
        return {
            "metadata": {
                "ticker": ticker,
                "asset_class": artifact["asset_class"],
                "latest_date": str(raw_df.index[-1].date()),
                "latest_close": round(raw_df['Close'].iloc[-1], 2)
            },
            "regime_intelligence": {
                "is_black_swan": bool(is_shock),
                "current_dominant_state": int(current_state),
                "current_probabilities": {f"Regime {i}": round(p, 4) for i, p in enumerate(current_probs)},
                "forecast_t1_probabilities": {f"Regime {i}": round(p, 4) for i, p in enumerate(forecast_probs)}
            },
            "policy_action": {
                "recommended_exposure": round(float(target_exposure), 2),
                "signal": signal
            }
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/screener")
def get_global_screener():
    # The 6 assets we trained models for
    target_assets = ["^GSPC", "^NSEI", "BTC-USD", "EURUSD=X", "GC=F", "TLT"]
    screener_data = []

    for ticker in target_assets:
        try:
            # We reuse your existing prediction logic to get the live state
            data = get_prediction(ticker)
            
            screener_data.append({
                "ticker": data["metadata"]["ticker"],
                "asset_class": data["metadata"]["asset_class"],
                "latest_close": data["metadata"]["latest_close"],
                "current_state": data["regime_intelligence"]["current_dominant_state"],
                "is_black_swan": data["regime_intelligence"]["is_black_swan"],
                "target_exposure": data["policy_action"]["recommended_exposure"]
            })
        except Exception as e:
            print(f"Skipping {ticker} for screener due to error: {e}")
            continue

    return {"heatmap": screener_data}

@app.get("/analytics/{ticker}")
def get_analytics(ticker: str):
    """
    Reads the real backtest results exported by main.py and serves them to React.
    """
    # 1. Match the 'safe_ticker' logic from main.py (^GSPC -> GSPC, EURUSD=X -> EURUSD_X)
    safe_ticker = ticker.replace("^", "").replace("=", "_")
    
    # Define paths
    metrics_path = f"backtests/metrics_{safe_ticker}.csv"
    curve_path = f"backtests/equity_curve_{safe_ticker}.csv"
    
    # 2. Safety Check: Does the backtest even exist?
    if not os.path.exists(metrics_path) or not os.path.exists(curve_path):
        raise HTTPException(
            status_code=404, 
            detail=f"Backtest results for {ticker} not found. Please run main.py first."
        )
    
    try:
        # 3. Read Metrics (The top cards)
        metrics_df = pd.read_csv(metrics_path)
        # Convert the first row of the dataframe into a clean dictionary
        metrics_dict = metrics_df.iloc[0].to_dict()
        
        # 4. Read Equity Curve (The chart)
        curve_df = pd.read_csv(curve_path)
        # Convert the entire dataframe into a list of dictionaries for Recharts
        # Format: [{"date": "2020-01", "strategy": 10000, "benchmark": 10000}, ...]
        equity_curve_list = curve_df.to_dict(orient="records")
        
        return {
            "metrics": metrics_dict,
            "equity_curve": equity_curve_list
        }

    except Exception as e:
        print(f"❌ ANALYTICS ERROR: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error while parsing backtest files.")
    
@app.post("/admin/retrain")
async def trigger_retrain(background_tasks: BackgroundTasks):
    background_tasks.add_task(run_training_script)
    return {"status": "Training started in background. Artifacts will hot-reload upon completion."}