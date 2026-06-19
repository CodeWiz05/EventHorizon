import pandas as pd
import numpy as np
import joblib
import os
import logging
import warnings

from src.data_engineering import MarketDataEngineer, FeatureFusion
from src.hmm_engine import RegimeHMM
from src.risk_analytics import RiskAnalyzer
from src.backtester import PolicyOptimizer, RegimePolicyEngine

# NEW MODULAR IMPORTS
from src.diagnostics import print_hmm_structural_diagnostics, block_bootstrap_pvalue
from src.feature_ops import rank_ic_predictive_power
from src.model_ops import run_seed_stability_test, run_causal_walk_forward

warnings.filterwarnings("ignore")
logging.getLogger("hmmlearn").setLevel(logging.CRITICAL)

def main():
    datasets = ["^NSEI", "^GSPC", "BTC-USD", "EURUSD=X", "GC=F", "TLT"]
    all_results = []
    
    os.makedirs("backtests", exist_ok=True)
    os.makedirs("api_models", exist_ok=True)

    for dataset in datasets:
        safe_ticker = dataset.replace("^", "").replace("=", "_")
        print(f"\n{'='*85}\nPROCESSING DATASET: {dataset}\n{'='*85}")
        
        # ==============================================================================
        # 1. DATA PIPELINE
        # ==============================================================================
        engineer = MarketDataEngineer(ticker=dataset, start_date="2015-01-01", end_date="2024-12-31")
        try:
            raw_df = engineer.run_pipeline()
            if raw_df.empty: 
                print(f"FAILED: yfinance returned empty data for {dataset}. Skipping.")
                continue
        except Exception as e:
            print(f"Pipeline failed for {dataset} : {e}"); continue
            
        # NEW FIX: Calculate the T+21 Forward Return column BEFORE splitting the data
        horizon = 21
        raw_df[f'T{horizon}_Return'] = (raw_df['Close'].shift(-horizon) / raw_df['Close']) - 1
            
        train_raw = raw_df.loc[:'2022-12-31'].copy()
        test_raw = raw_df.loc['2023-01-01':].copy()
        
        fusion = FeatureFusion(feature_cols=engineer.feature_names)
        train_fused = fusion.fit_transform(train_raw)
        test_fused = fusion.transform(test_raw)
        
        # ==============================================================================
        # 2. FEATURE SELECTION & OUTLIER REMOVAL
        # ==============================================================================
        temp_engine = RegimeHMM(max_states=3)
        train_outliers = temp_engine.fit_detect_anomalies(train_fused[engineer.feature_names].values)
        
        max_features = 5 if engineer.context['behavior'] == 'pro-cyclical' else 6
        final_features = rank_ic_predictive_power(train_raw, engineer.feature_names, train_outliers, horizon=21, top_k=max_features)
        
        if len(final_features) == 0: 
            print(f"FAILED: No features passed the IC threshold for {dataset}")
            continue

        print(f"\n[FEATURE LOCK] {dataset}: Selected Top {len(final_features)} Features")
        
        # ==============================================================================
        # 3. CORE HMM TRAINING & DIAGNOSTICS
        # ==============================================================================
        X_train, X_test = train_fused[final_features].values, test_fused[final_features].values
        returns_train, returns_test = train_raw['Log_Return'].values, test_raw['Log_Return'].values
        
        try: 
            optimal_engine = run_seed_stability_test(X_train[~train_outliers], returns_train[~train_outliers], max_states=3)
        except Exception as e:
            print(f"Model Optimization Failed: {e}"); continue
            
        dynamic_shock_label = optimal_engine.optimal_model.n_components
        print_hmm_structural_diagnostics(optimal_engine.optimal_model, dynamic_shock_label, dataset)

        # Build Training Regime DataFrame
        train_regimes = np.empty(len(X_train), dtype=int)
        train_regimes[train_outliers] = dynamic_shock_label
        train_regimes[~train_outliers] = optimal_engine.optimal_model.predict(X_train[~train_outliers])

        train_clean_probs = optimal_engine.optimal_model.predict_filtering_proba(X_train[~train_outliers])
        train_probs = np.zeros((len(X_train), dynamic_shock_label + 1))
        train_probs[~train_outliers, :dynamic_shock_label] = train_clean_probs
        train_probs[train_outliers, dynamic_shock_label] = 1.0 

        train_regime_df = train_raw.copy()
        train_regime_df['Regime'] = train_regimes
        for i in range(dynamic_shock_label + 1): 
            train_regime_df[f'Prob_Regime_{i}'] = train_probs[:, i]

        # ==============================================================================
        # 4. CAUSAL WALK-FORWARD VALIDATION (OOS)
        # ==============================================================================
        test_outliers = temp_engine.transform_anomalies(test_fused[engineer.feature_names].values)
        train_means = optimal_engine.optimal_model.means_
        
        test_regimes_causal, causal_probs = run_causal_walk_forward(
            X_train, X_test, train_outliers, test_outliers, dynamic_shock_label, 
            dynamic_shock_label, returns_train, returns_test, train_means
        )
        
        test_regime_df = test_raw.copy()
        test_regime_df['Regime'] = test_regimes_causal 
        for i in range(dynamic_shock_label + 1): 
            test_regime_df[f'Prob_Regime_{i}'] = causal_probs[:, i]

        # ==============================================================================
        # 5. RISK POLICY & BACKTESTING (WITH TRANSACTION COSTS)
        # ==============================================================================
        risk_profile = RiskAnalyzer(regime_df=train_regime_df, raw_features_df=train_raw, horizon=21).compute_metrics()
        optimizer = PolicyOptimizer(train_df=train_regime_df, risk_profile=risk_profile, shock_label=dynamic_shock_label, asset_class=engineer.asset_class)
        optimal_weights = optimizer.calibrate_weights()

        engine = RegimePolicyEngine(test_df=test_regime_df, optimal_weights=optimal_weights, tpm=optimal_engine.optimal_model.transmat_)
        test_results_df = engine.generate_signals()

        # APPLY INSTITUTIONAL TRANSACTION COSTS
        fee_map = {
            "CRYPTO": 0.0015,                      # 15 bps (Spread + Exchange fee)
            "EQUITY": 0.0005,                      # 5 bps (Slippage)
            "COMMODITY_PRECIOUS_METAL": 0.0003,    # 3 bps (Futures execution)
            "FIXED_INCOME": 0.0002,                # 2 bps (Highly liquid Treasury markets)
            "FX": 0.0001                           # 1 bps (Interbank spread)
        }
        bps_fee = fee_map.get(engineer.asset_class, 0.001)
        
        exposure_change = test_results_df['Target_Exposure'].diff().abs().fillna(0)
        t_costs = exposure_change * bps_fee
        
        strat_returns = (test_results_df['Target_Exposure'].shift(1) * test_results_df['Log_Return']) - t_costs
        bh_returns = test_results_df['Log_Return']

        aligned_df = pd.DataFrame({'Strat': strat_returns, 'BH': bh_returns}).dropna()

        # Calculate Performance Metrics
        strat_ann_ret = aligned_df['Strat'].mean() * 252
        bh_ann_ret = aligned_df['BH'].mean() * 252
        oos_strat_sharpe = strat_ann_ret / (aligned_df['Strat'].std() * np.sqrt(252))
        oos_bh_sharpe = bh_ann_ret / (aligned_df['BH'].std() * np.sqrt(252))
        
        strat_cum = (1 + aligned_df['Strat']).cumprod()
        bh_cum = (1 + aligned_df['BH']).cumprod()
        
        strat_running_max = strat_cum.cummax()
        bh_running_max = bh_cum.cummax()
        
        strat_dd = ((strat_cum - strat_running_max) / strat_running_max).min()
        bh_dd = ((bh_cum - bh_running_max) / bh_running_max).min()

        strat_calmar = strat_ann_ret / abs(strat_dd) if strat_dd != 0 else 0
        bh_calmar = bh_ann_ret / abs(bh_dd) if bh_dd != 0 else 0

        p_val_boot = block_bootstrap_pvalue(aligned_df['Strat'], aligned_df['BH'], block_size=10, n_bootstraps=5000)

        all_results.append({
            "Asset": dataset, 
            "Strat_Sharpe": f"{oos_strat_sharpe:.2f}",
            "B&H_Sharpe": f"{oos_bh_sharpe:.2f}",
            "Strat_MaxDD": f"{strat_dd*100:.1f}%",
            "B&H_MaxDD": f"{bh_dd*100:.1f}%",
            "Strat_Calmar": f"{strat_calmar:.2f}",
            "B&H_Calmar": f"{bh_calmar:.2f}"
        })

        # ==============================================================================
        # 6. EXPORT ARTIFACTS
        # ==============================================================================
        strat_dd_curve = ((strat_cum - strat_running_max) / strat_running_max) * 100
        bh_dd_curve = ((bh_cum - bh_running_max) / bh_running_max) * 100

        export_df = pd.DataFrame({
            "date": aligned_df.index.strftime('%Y-%m'),
            "strategy": strat_cum * 10000,
            "benchmark": bh_cum * 10000,
            "strategy_dd": strat_dd_curve,
            "benchmark_dd": bh_dd_curve
        })
        
        metrics_df = pd.DataFrame([{
            "sharpe": f"{oos_strat_sharpe:.2f}",
            "benchmark_sharpe": f"{oos_bh_sharpe:.2f}",
            "max_drawdown": f"{strat_dd*100:.1f}%",
            "benchmark_drawdown": f"{bh_dd*100:.1f}%",
            "cagr": f"{strat_ann_ret*100:.1f}%",
            "benchmark_cagr": f"{bh_ann_ret*100:.1f}%",
            "calmar": f"{strat_calmar:.2f}",
            "benchmark_calmar": f"{bh_calmar:.2f}"
        }])

        export_df.to_csv(f"backtests/equity_curve_{safe_ticker}.csv", index=False)
        metrics_df.to_csv(f"backtests/metrics_{safe_ticker}.csv", index=False)
            
        deployment_artifact = {
            "asset_class": engineer.asset_class,
            "features": final_features,
            "scaler": fusion.scaler,
            "anomaly_detector": {
                "inv_cov": temp_engine.inv_cov_matrix,
                "mean": temp_engine.mean_distr,
                "threshold": temp_engine.threshold
            },
            "hmm_model": optimal_engine.optimal_model,
            "shock_label": dynamic_shock_label,
            "policy_weights": optimal_weights
        }
        
        export_path = f"api_models/{safe_ticker}_artifact.joblib"
        joblib.dump(deployment_artifact, export_path)
        logging.info(f"Serialized production artifact to {export_path}")

    # Final Summary Table
    if all_results:
        print("\n" + "="*105)
        print("INSTITUTIONAL REGIME POLICY OOS VALIDATION (RISK-ADJUSTED METRICS)")
        print("="*105)
        print(pd.DataFrame(all_results).to_string(index=False))
        print("="*105)

if __name__ == "__main__":
    main()