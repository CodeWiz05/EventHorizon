import matplotlib.pyplot as plt
import pandas as pd

def plot_regimes_and_performance(df: pd.DataFrame, title: str = "Out-of-Sample Performance"):
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [2, 1]}, sharex=True)
    
    # Calculate Cumulative Returns
    hmm_cum = (1 + df['HMM_Return'].fillna(0)).cumprod()
    bh_cum = (1 + df['BuyHold_Return'].fillna(0)).cumprod()
    
    ax1.plot(df.index, hmm_cum, label='HMM Strategy', color='green', linewidth=1.5)
    ax1.plot(df.index, bh_cum, label='Buy & Hold', color='gray', alpha=0.6, linewidth=1.5)
    ax1.set_title(title, fontsize=14, fontweight='bold')
    ax1.set_ylabel('Cumulative Return')
    ax1.legend(loc='upper left')
    
    # Shade regimes on the bottom chart (Price)
    ax2.plot(df.index, df.iloc[:, 0], color='black', linewidth=1) # Close price
    
    regime_colors = {0: 'lightgreen', 1: 'lightcoral', 2: 'khaki', 3: 'lightblue', 4: 'red'}
    for i in range(len(df) - 1):
        regime = df['Regime'].iloc[i]
        ax2.axvspan(df.index[i], df.index[i+1], color=regime_colors.get(regime, 'grey'), alpha=0.3)
        
    ax2.set_ylabel('Asset Price')
    plt.tight_layout()
    plt.show()