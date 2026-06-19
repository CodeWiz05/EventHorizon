import pandas as pd
import numpy as np
import logging

class PolicyOptimizer:
    def __init__(self, train_df: pd.DataFrame, risk_profile: pd.DataFrame, shock_label: int, asset_class: str):
        self.train_df = train_df.copy()
        self.risk_profile = risk_profile
        self.shock_label = shock_label
        self.asset_class = asset_class

    def calibrate_weights(self) -> dict:
        logging.info(f"Routing to Asset-Specific Strategy Sleeve for {self.asset_class}...")
        
        # 1. Asset-Specific Constraints & Temperature (Sharpness)
        if self.asset_class == "CRYPTO":
            sharpness = 4.0      # Highly aggressive (Winner-takes-all) to avoid chop
            is_safe_haven = False
            max_exposure = 1.0
        elif self.asset_class == "FX":
            sharpness = 1.5      # Smooth blending for mean-reverting markets
            is_safe_haven = False
            max_exposure = 0.5
        elif self.asset_class == "COMMODITY_PRECIOUS_METAL":
            sharpness = 2.5      
            is_safe_haven = True
            max_exposure = 1.0
        elif self.asset_class == "FIXED_INCOME":
            sharpness = 2.0      
            is_safe_haven = True
            max_exposure = 0.8
        else: # EQUITY
            sharpness = 3.0      # Strong conviction, but not binary
            is_safe_haven = False
            max_exposure = 1.0

        # 2. Dimensionally Consistent, Duration-Weighted Scoring
        scores = {}
        for regime, row in self.risk_profile.iterrows():
            # Fix 1: Dimensional Consistency (Sharpe scaled by absolute tail risk)
            base_score = row['Sharpe'] / (1.0 + abs(row['Max_Drawdown']) + abs(row['CVaR_95']))
            
            # Fix 2: Duration Weighting (Log scale penalizes short noise regimes)
            duration_multiplier = np.log(max(row['Mean_Duration_Days'], 2))
            
            raw_score = base_score * duration_multiplier
            
            # Fix 3: Soften Shock State (No hardcoded 1.0 or 0.0)
            if regime == self.shock_label:
                if is_safe_haven:
                    raw_score = abs(raw_score) * 2.0  # Boost appeal for safe havens
                else:
                    raw_score = -abs(raw_score) * 2.0 # Deeply penalize for risk assets
            
            scores[regime] = raw_score

        # 3. Temperature-Scaled Softmax Allocation
        regimes = list(scores.keys())
        score_values = np.array([scores[r] for r in regimes])
        
        # Normalize scores (Z-score) so sharpness acts uniformly across different assets
        if len(score_values) > 1 and np.std(score_values) > 0:
            norm_scores = (score_values - np.mean(score_values)) / np.std(score_values)
        else:
            norm_scores = score_values
            
        softmax_probs = np.exp(norm_scores * sharpness) / np.sum(np.exp(norm_scores * sharpness))
        
        # Scale to max allowable exposure
        final_weights = {regimes[i]: round(float(softmax_probs[i] * max_exposure), 3) for i in range(len(regimes))}
        
        # Ensure matrix completeness
        for i in range(self.shock_label + 1):
            if i not in final_weights:
                final_weights[i] = 0.0

        logging.info(f"[{self.asset_class}] Locked Softmax Weights: {final_weights}")
        return final_weights


class RegimePolicyEngine:
    def __init__(self, test_df: pd.DataFrame, optimal_weights: dict, tpm: np.ndarray = None):
        self.df = test_df.copy()
        self.optimal_weights = optimal_weights
        self.tpm = tpm  # Transition Probability Matrix

    def generate_signals(self) -> pd.DataFrame:
        # Strictly sort columns to guarantee Prob_Regime_0, 1, 2, 3 order
        prob_cols = sorted([col for col in self.df.columns if col.startswith('Prob_Regime_')], 
                           key=lambda x: int(x.split('_')[-1]))
        current_probs = self.df[prob_cols].fillna(0).values
        
        expected_probs = current_probs.copy()
        
        # =====================================================================
        # Fix 4: TRANSITION AWARENESS (Forward Expectation Weighting)
        # =====================================================================
        if self.tpm is not None:
            n_hmm_states = self.tpm.shape[0]
            # Matrix multiplication strictly on native HMM states (Dimensions 0 to 2)
            # The Shock state (Dimension 3) bypasses the TPM and retains its raw probability
            expected_probs[:, :n_hmm_states] = np.dot(current_probs[:, :n_hmm_states], self.tpm)
        else:
            logging.warning("TPM not provided to Policy Engine. Forward expectation disabled.")
        
        exposure = np.zeros(len(self.df))
        for regime, weight in self.optimal_weights.items():
            # Map the weight to the exact probability index
            if regime < expected_probs.shape[1]:
                exposure += expected_probs[:, regime] * weight

        # =====================================================================
        # CONTINUOUS STRUCTURAL TREND FILTER
        # =====================================================================
        ma_100 = self.df['Close'].rolling(window=100, min_periods=1).mean()
        trend_strength = (self.df['Close'] / ma_100) - 1.0
        
        # Gentle 50% modifier bounded to a max 10% penalty/boost
        trend_multiplier = 1.0 + 0.5 * np.clip(trend_strength.fillna(0), -0.20, 0.20)
        exposure = exposure * trend_multiplier
        
        self.df['Raw_Exposure'] = np.clip(exposure, 0.0, 1.0)
        
        # Execution Smoothing
        self.df['Target_Exposure'] = self.df['Raw_Exposure'].ewm(alpha=0.65, adjust=False).mean()
        
        return self.df