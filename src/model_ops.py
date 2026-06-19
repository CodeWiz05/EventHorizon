import numpy as np
import pandas as pd
from src.hmm_engine import RegimeHMM
from src.student_t_hmm import StudentTHMM
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

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
        
        dist_matrix = cdist(model.means_, train_means, metric='cosine')
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