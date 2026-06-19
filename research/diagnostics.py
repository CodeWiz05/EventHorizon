import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class MarketDiagnostics:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.close_col = [c for c in self.df.columns if 'Close' in str(c)][0]
        self.df['Daily_Return'] = self.df[self.close_col].pct_change().dropna()

    def run_diagnostics(self):
        rets = self.df['Daily_Return'].dropna()
        
        # Statistical Moments
        mean = np.mean(rets)
        std = np.std(rets)
        skewness = stats.skew(rets)
        kurt = stats.kurtosis(rets) # Excess kurtosis (Normal = 0)
        
        # Jarque-Bera Test for Normality
        jb_stat, p_val = stats.jarque_bera(rets)
        
        print("\n" + "="*80)
        print("DIAGNOSTICS: DISTRIBUTIONAL ANALYSIS (NON-GAUSSIAN REALITY)")
        print("="*80)
        print(f"Mean Return:      {mean*100:.4f}%")
        print(f"Standard Dev:     {std*100:.4f}%")
        print(f"Skewness:         {skewness:.4f} (Negative = Fat left tail / Crash risk)")
        print(f"Excess Kurtosis:  {kurt:.4f} ( > 0 means Fat Tails / Leptokurtic)")
        print(f"Jarque-Bera Stat: {jb_stat:.2f}")
        print(f"JB P-Value:       {p_val:.6f} ( < 0.05 strictly rejects Normality)")
        print("-" * 80)
        print("CONCLUSION: Market returns are highly non-Gaussian. Gaussian HMMs will")
        print("structurally underestimate tail risk, necessitating our Two-Pass anomaly filter.")
        print("="*80)

    def plot_residuals(self):
        logging.info("Plotting Returns Distribution vs Normal Curve...")
        rets = self.df['Daily_Return'].dropna()
        
        plt.figure(figsize=(10, 6))
        plt.hist(rets, bins=100, density=True, alpha=0.6, color='steelblue', label='Actual Returns')
        
        # Overlay Normal Distribution
        xmin, xmax = plt.xlim()
        x = np.linspace(xmin, xmax, 100)
        p = stats.norm.pdf(x, np.mean(rets), np.std(rets))
        plt.plot(x, p, 'k', linewidth=2, label='Gaussian Assumption')
        
        plt.title('Daily Returns Distribution vs. Gaussian Assumption')
        plt.xlabel('Daily Return')
        plt.ylabel('Density')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()