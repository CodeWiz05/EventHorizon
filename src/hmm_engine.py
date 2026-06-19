import numpy as np
import pandas as pd
from scipy.spatial import distance
from scipy.stats import chi2
from src.student_t_hmm import StudentTHMM
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class RegimeHMM:
    def __init__(self, max_states: int = 4, random_state: int = 42):
        self.max_states = max_states
        self.random_state = random_state
        self.optimal_model = None
        self.optimal_states = None
        self.shock_state_label = max_states 

    def _calculate_free_parameters(self, n_components: int, n_features: int) -> int:
        transitions = n_components * (n_components - 1)
        means = n_components * n_features
        covariances = n_components * n_features 
        init_probs = n_components - 1
        return int(transitions + means + covariances + init_probs)

    def fit_detect_anomalies(self, X_train: np.ndarray) -> np.ndarray:
        logging.info("Fitting Mahalanobis parameters on Train Data...")
        cov_matrix = np.cov(X_train, rowvar=False)
        
        # Covariance Regularization to prevent singular matrices
        cov_matrix += np.eye(cov_matrix.shape[0]) * 1e-4
        
        self.inv_cov_matrix = np.linalg.inv(cov_matrix)
        self.mean_distr = np.mean(X_train, axis=0)
        
        self.threshold = chi2.ppf(0.99, df=X_train.shape[1])
        distances = np.array([distance.mahalanobis(row, self.mean_distr, self.inv_cov_matrix) for row in X_train])
        is_outlier = (distances ** 2) > self.threshold
        logging.info(f"Train Pass 1: Isolated {np.sum(is_outlier)} tail-risk events.")
        return is_outlier

    def transform_anomalies(self, X_test: np.ndarray) -> np.ndarray:
        distances = np.array([distance.mahalanobis(row, self.mean_distr, self.inv_cov_matrix) for row in X_test])
        is_outlier = (distances ** 2) > self.threshold
        return is_outlier

    def optimize_model(self, X_clean: np.ndarray):
        logging.info("Pass 2: Optimizing HMM (Forcing Convergence)...")
        
        print(f"Input shape to HMM: {X_clean.shape}")
        assert X_clean.shape[1] > 0, "No features passed to HMM"
        
        n_samples, n_features = X_clean.shape
        best_bic = np.inf

        for n_states in range(2, self.max_states + 1):
            try:
                # Strictly enforced parameters
                model = StudentTHMM(n_components=n_states, df=5.0, 
                                    n_iter=1000, random_state=self.random_state, tol=1e-3)
                model.fit(X_clean)
                
                if not model.monitor_.converged:
                    logging.warning(f"{n_states}-State Model did NOT converge. Discarding.")
                    continue
                
                log_likelihood = model.score(X_clean)
                n_params = self._calculate_free_parameters(n_states, n_features)
                bic = -2 * log_likelihood + n_params * np.log(n_samples)
                
                if bic < best_bic:
                    best_bic = bic
                    self.optimal_model = model
                    self.optimal_states = n_states
            except Exception as e:
                logging.warning(f"Model failed for {n_states} states: {e}")

        if self.optimal_model is None:
            raise RuntimeError("CRITICAL FAILURE: No HMM architectures converged.")
            
        logging.info(f"Optimal states chosen: {self.optimal_states}")

    def run_engine(self, fused_df: pd.DataFrame) -> pd.DataFrame:
        feature_cols = [col for col in fused_df.columns if col != 'Close']
        X = fused_df[feature_cols].values
        
        print(f"Input shape to HMM: {X.shape}")
        assert X.shape[1] > 0, "No features passed to HMM"
        
        is_outlier = self.detect_anomalies(X)
        X_clean = X[~is_outlier]
        
        self.optimize_model(X_clean)
        clean_states = self.optimal_model.predict(X_clean)
        
        regime_array = np.empty(len(X), dtype=int)
        regime_array[is_outlier] = self.shock_state_label
        regime_array[~is_outlier] = clean_states
        
        regime_df = fused_df.copy()
        regime_df['Regime'] = regime_array
        return regime_df