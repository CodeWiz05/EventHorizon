import pandas as pd
import numpy as np
import logging

class RiskAnalyzer:
    def __init__(self, regime_df: pd.DataFrame, raw_features_df: pd.DataFrame, horizon: int):
        self.df = regime_df.copy()
        self.horizon = horizon

    def compute_metrics(self) -> pd.DataFrame:
        regimes = sorted(self.df['Regime'].dropna().unique())
        stats = []

        blocks = self.df['Regime'].diff().ne(0).cumsum().rename('Block')
        durations = self.df.groupby(blocks).size()
        regime_mapped = self.df.groupby(blocks)['Regime'].first()
        dur_df = pd.DataFrame({'Regime': regime_mapped, 'Duration': durations})
        mean_durations = dur_df.groupby('Regime')['Duration'].mean()

        for r in regimes:
            mask = self.df['Regime'] == r
            r_rets = self.df.loc[mask, 'Log_Return']

            ann_ret = r_rets.mean() * 252
            ann_vol = r_rets.std() * np.sqrt(252)
            sharpe = ann_ret / ann_vol if ann_vol > 0 else 0

            cum_ret = (1 + r_rets).cumprod()
            max_dd = ((cum_ret - cum_ret.cummax()) / cum_ret.cummax()).min() if len(cum_ret) > 0 else 0.0

            mean_dur = mean_durations.get(r, 0)
            
            # --- NEW: REGIME SPECIFIC VaR and CVaR ---
            # 5th percentile daily loss
            var_95 = np.percentile(r_rets, 5) if len(r_rets) > 0 else 0.0
            # Expected loss given that we breached VaR (Conditional VaR / Expected Shortfall)
            cvar_95 = r_rets[r_rets <= var_95].mean() if len(r_rets[r_rets <= var_95]) > 0 else var_95

            stats.append({
                'Regime': r,
                'Ann_Return': ann_ret,
                'Ann_Vol': ann_vol,
                'Sharpe': sharpe,
                'Max_Drawdown': max_dd,
                'VaR_95': var_95,       # Translated to daily risk
                'CVaR_95': cvar_95,     # Translated to tail risk
                'Mean_Duration_Days': mean_dur
            })

        profile = pd.DataFrame(stats).set_index('Regime')

        profile['Label'] = 'Transition / Choppy'
        profile.loc[profile['Ann_Return'].idxmax(), 'Label'] = 'Bull Regime'
        profile.loc[profile['Ann_Return'].idxmin(), 'Label'] = 'Bear Regime'

        print("\n" + "="*85)
        print(f"REGIME INTERPRETABILITY PROFILE [Target: T+{self.horizon}]")
        print("="*85)
        print(profile.round(4).to_string())
        print("="*85 + "\n")

        return profile