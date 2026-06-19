import numpy as np
import pandas as pd
from scipy import stats
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