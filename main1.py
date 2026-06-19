import numpy as np
import pandas as pd
import logging
import warnings
from scipy import stats
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from src.data_engineering import MarketDataEngineer, FeatureFusion
from src.hmm_engine import RegimeHMM
from src.risk_analytics import RiskAnalyzer
from src.backtester import PolicyOptimizer, RegimePolicyEngine
from src.student_t_hmm import StudentTHMM
import joblib
import os

warnings.filterwarnings("ignore")
logging.getLogger("hmmlearn").setLevel(logging.CRITICAL)

def print_hmm_structural_diagnostics(model, n_states, dataset_name):
    print("\n" + "="*85)
    print(f"HMM STRUCTURAL DIAGNOSTICS: {dataset_name} (THE 'STEP 1' GATE)")
    print("="*85)
    
    tpm = model.transmat_
    print("TRANSITION PROBABILITY MATRIX (A):")
    df_tpm = pd.DataFrame(tpm, 
                          index=[f"From Reg {i}" for i in range(n_states)], 
                          columns=[f"To Reg {i}" for i in range(n_states)])
    print(df_tpm.round(4).to_string())
    
    print("\nSELF-TRANSITION PROBABILITIES (Stickiness):")
    self_transitions = np.diag(tpm)
    for i, p in enumerate(self_transitions):
        status = "✅ PASS (Highly Stable)" if p > 0.85 else ("⚠️ ACCEPTABLE" if p > 0.70 else "❌ FAIL (Flickering Noise)")
        implied_duration = 1.0 / (1.0 - p) if p < 1.0 else np.inf
        print(f"Regime {i}: {p:.4f} | Implied Duration: ~{implied_duration:.1f} days | {status}")
        
    avg_persistence = np.mean(self_transitions)
    sys_status = "✅ ROBUST" if avg_persistence > 0.80 else "❌ UNSTABLE"
    print(f"\nAverage System Persistence: {avg_persistence:.4f} [{sys_status}]")
    print("="*85 + "\n")

def block_bootstrap_pvalue(strat_rets, bh_rets, block_size=10, n_bootstraps=5000):
    diff = (strat_rets - bh_rets).values
    n = len(diff)
    if n < block_size: return 1.0
    
    obs_diff = np.mean(diff)
    centered_diff = diff - obs_diff
    
    boot_means = np.zeros(n_bootstraps)
    num_blocks = (n // block_size) + 1
    
    for i in range(n_bootstraps):
        start_indices = np.random.randint(0, n - block_size + 1, size=num_blocks)
        boot_sample = np.concatenate([centered_diff[idx : idx + block_size] for idx in start_indices])[:n]
        boot_means[i] = np.mean(boot_sample)
        
    p_val = np.mean(np.abs(boot_means) >= np.abs(obs_diff))
    return p_val

def run_seed_stability_test(X_clean: np.ndarray, returns_clean: np.ndarray, max_states: int = 3) -> RegimeHMM:
    seeds = [42, 100, 2026, 7, 99]
    best_overall_bic = np.inf
    best_engine = None
    bic_scores = []
    for seed in seeds:
        engine = RegimeHMM(max_states=max_states, random_state=seed)
        engine.optimize_model(X_clean)
        if engine.optimal_model is not None:
            states = engine.optimal_model.predict(X_clean)
            if np.min(np.bincount(states, minlength=engine.optimal_states)) < 50: continue
            
            log_likelihood = engine.optimal_model.score(X_clean)
            n_params = engine._calculate_free_parameters(engine.optimal_states, X_clean.shape[1])
            bic = -2 * log_likelihood + n_params * np.log(len(X_clean))
            bic_scores.append(bic)
            if bic < best_overall_bic:
                best_overall_bic = bic
                best_engine = engine

    if not bic_scores:
        raise RuntimeError("CRITICAL FAILURE: No seed produced a valid model with >= 50 occurrences per regime.")
    return best_engine

def rank_ic_predictive_power(train_df: pd.DataFrame, features: list, outliers: np.ndarray, horizon: int, top_k: int):
    t_returns = train_df[f'T{horizon}_Return'].values
    X_clean = train_df[features].values[~outliers]
    y_clean = t_returns[~outliers]
    valid_idx = ~np.isnan(y_clean)
    X_valid, y_valid = X_clean[valid_idx], y_clean[valid_idx]
    
    purged_idx = []
    last = -np.inf
    for i in range(len(y_valid)):
        if i - last >= horizon:
            purged_idx.append(i)
            last = i
            
    X_purged, y_purged = X_valid[purged_idx], y_valid[purged_idx]
    
    correlations = []
    for i in range(X_purged.shape[1]):
        corr, p_val = stats.spearmanr(X_purged[:, i], y_purged)
        correlations.append((features[i], corr, p_val))
        
    ranking = pd.DataFrame(correlations, columns=['Feature', 'IC', 'P_Value'])
    ranking['Abs_IC'] = ranking['IC'].abs()
    
    selected_features = ranking[ranking['Abs_IC'] > 0.04].sort_values(by='Abs_IC', ascending=False).head(top_k)['Feature'].tolist()
    return selected_features

def run_causal_walk_forward(X_train, X_test, train_outliers, test_outliers, global_n_states, shock_label, returns_train, returns_test, train_means):
    test_len = len(X_test)
    step_size = 252 
    causal_states = np.empty(test_len, dtype=int)
    causal_states.fill(-1)
    causal_probs = np.zeros((test_len, global_n_states + 1))
    
    X_expanding, outliers_expanding, returns_expanding = X_train.copy(), train_outliers.copy(), returns_train.copy()
    
    for start_idx in range(0, test_len, step_size):
        end_idx = min(start_idx + step_size, test_len)
        X_clean_exp = X_expanding[~outliers_expanding]
        
        model = StudentTHMM(n_components=global_n_states, df=5.0, n_iter=1000, random_state=42, tol=1e-3)
        model.fit(X_clean_exp)
        
        dist_matrix = cdist(model.means_, train_means, metric='euclidean')
        new_indices, original_indices = linear_sum_assignment(dist_matrix)
        state_mapping = {new_idx: orig_idx for new_idx, orig_idx in zip(new_indices, original_indices)}
        
        chunk, chunk_outliers = X_test[start_idx:end_idx], test_outliers[start_idx:end_idx]
        chunk_clean = chunk[~chunk_outliers]
        
        if len(chunk_clean) > 0:
            # STRICT CAUSAL FIX: Feed continuous timeline to prevent Markov Amnesia
            full_X = np.vstack([X_clean_exp, chunk_clean])
            full_probs_raw = model.predict_filtering_proba(full_X)
            
            # Extract only the test portion
            raw_probs_unaligned = full_probs_raw[-len(chunk_clean):]
            
            aligned_probs = np.zeros_like(raw_probs_unaligned)
            for new_idx, orig_idx in state_mapping.items():
                aligned_probs[:, orig_idx] = raw_probs_unaligned[:, new_idx]
                
            causal_probs[start_idx:end_idx, :global_n_states][~chunk_outliers] = aligned_probs
            causal_probs[start_idx:end_idx, shock_label][chunk_outliers] = 1.0
            
            # STRICT CAUSAL FIX: States must be argmax of filtering probs, NOT Viterbi predict()
            causal_states[start_idx:end_idx][~chunk_outliers] = np.argmax(aligned_probs, axis=1)
            causal_states[start_idx:end_idx][chunk_outliers] = shock_label
            
        X_expanding = np.vstack([X_expanding, chunk])
        outliers_expanding = np.concatenate([outliers_expanding, chunk_outliers])
        returns_expanding = np.concatenate([returns_expanding, returns_test[start_idx:end_idx]])
        
    return causal_states, causal_probs

def main():
    datasets = ["^NSEI", "^GSPC", "BTC-USD", "EURUSD=X", "GC=F", "TLT"]
    horizons = [21] 
    all_results = []

    for dataset in datasets:
        safe_ticker = dataset.replace("^", "").replace("=", "_")
        print("\n" + "="*85)
        print(f"PROCESSING DATASET: {dataset}")
        print("="*85)
        
        engineer = MarketDataEngineer(ticker=dataset, start_date="2015-01-01", end_date="2024-12-31")
        try: 
            raw_df = engineer.run_pipeline()
            
            # STRICT DEFENSIVE CHECK: Did yfinance fail silently?
            if raw_df.empty:
                print(f"FAILED: yfinance returned an empty dataset for {dataset}. Skipping.")
                continue
                
            initial_features = engineer.feature_names
        except Exception as e: 
            print(f"Pipeline failed for {dataset} : {e}")
            continue
        raw_df.index = pd.to_datetime(raw_df.index).tz_localize(None)
        
        for h in horizons:
            raw_df[f'T{h}_Return'] = (raw_df['Close'].shift(-h) / raw_df['Close']) - 1
        train_raw, test_raw = raw_df.loc[:'2022-12-31'].copy(), raw_df.loc['2023-01-01':].copy()
        
        fusion = FeatureFusion(feature_cols=initial_features)
        train_fused, test_fused = fusion.fit_transform(train_raw), fusion.transform(test_raw)
        
        temp_engine = RegimeHMM(max_states=3)
        train_outliers = temp_engine.fit_detect_anomalies(train_fused[initial_features].values)
        
        max_features = 5 if engineer.context['behavior'] == 'pro-cyclical' else 6
        final_features = rank_ic_predictive_power(train_raw, initial_features, train_outliers, horizon=21, top_k=max_features)
        if len(final_features) == 0: 
            print(f"FAILED: No features passed the IC threshold for {dataset}")
            continue

        print(f"\n[FEATURE LOCK] {dataset} ({engineer.context['behavior']}): Selected Top {len(final_features)} Features")
        print(f"Features: {final_features}\n")

        X_train, X_test = train_fused[final_features].values, test_fused[final_features].values
        returns_train, returns_test = train_raw['Log_Return'].values, test_raw['Log_Return'].values
        
        try: optimal_engine = run_seed_stability_test(X_train[~train_outliers], returns_train[~train_outliers], max_states=3)
        except: continue
            
        dynamic_shock_label = optimal_engine.optimal_model.n_components
        print_hmm_structural_diagnostics(optimal_engine.optimal_model, dynamic_shock_label, dataset)
        padded_transmat = np.eye(dynamic_shock_label + 1)
        padded_transmat[:dynamic_shock_label, :dynamic_shock_label] = optimal_engine.optimal_model.transmat_
        
        train_regimes = np.empty(len(X_train), dtype=int)
        train_regimes[train_outliers] = dynamic_shock_label
        train_regimes[~train_outliers] = optimal_engine.optimal_model.predict(X_train[~train_outliers])

        train_regime_df = train_raw.copy()
        train_regime_df['Regime'] = train_regimes

        test_outliers = temp_engine.transform_anomalies(test_fused[initial_features].values)
        
        train_means = optimal_engine.optimal_model.means_
        test_regimes_causal, causal_probs = run_causal_walk_forward(X_train, X_test, train_outliers, test_outliers, dynamic_shock_label, dynamic_shock_label, returns_train, returns_test, train_means)
        
        test_regime_df = test_raw.copy()
        test_regime_df['Regime'] = test_regimes_causal 
        for i in range(dynamic_shock_label + 1): 
            test_regime_df[f'Prob_Regime_{i}'] = causal_probs[:, i]

        train_clean_probs = optimal_engine.optimal_model.predict_filtering_proba(X_train[~train_outliers])
        train_probs = np.zeros((len(X_train), dynamic_shock_label + 1))
        train_probs[~train_outliers, :dynamic_shock_label] = train_clean_probs
        train_probs[train_outliers, dynamic_shock_label] = 1.0 

        for i in range(dynamic_shock_label + 1): 
            train_regime_df[f'Prob_Regime_{i}'] = train_probs[:, i]

        risk_profile = RiskAnalyzer(regime_df=train_regime_df, raw_features_df=train_raw, horizon=21).compute_metrics()
        
        # Replace this old line:
        # optimizer = PolicyOptimizer(train_df=train_regime_df, risk_profile=risk_profile, shock_label=dynamic_shock_label, context=engineer.context)

        # With this updated line:
        optimizer = PolicyOptimizer(train_df=train_regime_df, risk_profile=risk_profile, shock_label=dynamic_shock_label, asset_class=engineer.asset_class)
        optimal_weights = optimizer.calibrate_weights()

        engine = RegimePolicyEngine(test_df=test_regime_df, optimal_weights=optimal_weights)
        test_results_df = engine.generate_signals()

        # APPLY TRANSACTION COSTS (10 bps / 0.1% fee on exposure delta)
        transaction_fee_bps = 0.001 
        exposure_change = test_results_df['Target_Exposure'].diff().abs().fillna(0)
        t_costs = exposure_change * transaction_fee_bps
        
        # Gross returns minus transaction costs = Net Returns
        strat_returns = (test_results_df['Target_Exposure'].shift(1) * test_results_df['Log_Return']) - t_costs
        bh_returns = test_results_df['Log_Return']

        aligned_df = pd.DataFrame({'Strat': strat_returns, 'BH': bh_returns}).dropna()

        # Calculate OOS Sharpe Ratios
        strat_ann_ret = aligned_df['Strat'].mean() * 252
        bh_ann_ret = aligned_df['BH'].mean() * 252
        oos_strat_sharpe = strat_ann_ret / (aligned_df['Strat'].std() * np.sqrt(252))
        oos_bh_sharpe = bh_ann_ret / (aligned_df['BH'].std() * np.sqrt(252))
        
        # Calculate Max Drawdown
        strat_cum = (1 + aligned_df['Strat']).cumprod()
        bh_cum = (1 + aligned_df['BH']).cumprod()
        strat_dd = ((strat_cum - strat_cum.cummax()) / strat_cum.cummax()).min()
        bh_dd = ((bh_cum - bh_cum.cummax()) / bh_cum.cummax()).min()

        # Calculate Calmar Ratio (Return / Max Drawdown)
        strat_calmar = strat_ann_ret / abs(strat_dd) if strat_dd != 0 else 0
        bh_calmar = bh_ann_ret / abs(bh_dd) if bh_dd != 0 else 0

        # Keep Bootstrap to show if return drag is statistically significant
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

        # NEW: Export Backtest Data for the Dashboard
        if not os.path.exists("backtests"):
            os.makedirs("backtests")
            
        # 1. Save the Equity & Drawdown Curves (Strategy vs Benchmark)
        strat_cum = (1 + aligned_df['Strat']).cumprod()
        bh_cum = (1 + aligned_df['BH']).cumprod()
        
        # Calculate running maximums to compute real-time drawdown
        strat_running_max = strat_cum.cummax()
        bh_running_max = bh_cum.cummax()
        
        # Calculate underwater drawdown curves (as percentages)
        strat_dd_curve = ((strat_cum - strat_running_max) / strat_running_max) * 100
        bh_dd_curve = ((bh_cum - bh_running_max) / bh_running_max) * 100

        # Create export dataframe with both equity values and drawdown percentages
        export_df = pd.DataFrame({
            "date": aligned_df.index.strftime('%Y-%m'),
            "strategy": strat_cum * 10000,
            "benchmark": bh_cum * 10000,
            "strategy_dd": strat_dd_curve,
            "benchmark_dd": bh_dd_curve
        })
        
        # 2. Save the Metrics as a small JSON or CSV
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

        # Save files
        export_df.to_csv(f"backtests/equity_curve_{safe_ticker}.csv", index=False)
        metrics_df.to_csv(f"backtests/metrics_{safe_ticker}.csv", index=False)

        # ==============================================================================
        # MLOPS: SERIALIZE DEPLOYMENT ARTIFACT
        # ==============================================================================
        if not os.path.exists("api_models"):
            os.makedirs("api_models")
            
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

    if all_results:
        print("\n" + "="*105)
        print("INSTITUTIONAL REGIME POLICY OOS VALIDATION (RISK-ADJUSTED METRICS)")
        print("="*105)
        print(pd.DataFrame(all_results).to_string(index=False))
        print("="*105)

if __name__ == "__main__":
    main()