import pandas as pd
import numpy as np

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
