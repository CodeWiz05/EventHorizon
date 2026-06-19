import numpy as np
from scipy.special import gammaln, logsumexp
from hmmlearn.base import BaseHMM
from sklearn.cluster import KMeans
import logging

class StudentTHMM(BaseHMM):
    """
    Custom Hidden Markov Model with multivariate Student-t emissions.
    Designed explicitly for leptokurtic financial returns.
    """
    def __init__(self, n_components=3, df=5.0, lambda_reg=1e-4, n_iter=100, tol=1e-4, random_state=None):
        super().__init__(n_components=n_components, n_iter=n_iter, tol=tol, 
                         random_state=random_state, params="stmc", init_params="stmc")
        self.df_ = df
        self.lambda_reg = lambda_reg  # Covariance Regularization Parameter
        self.means_ = None
        self.covars_ = None 
        
    def _init(self, X, lengths=None):
        super()._init(X, lengths=lengths)
        
        # Explicitly initialize KMeans with n_init to ensure stable starts
        kmeans = KMeans(n_clusters=self.n_components, n_init=10, random_state=self.random_state)
        kmeans.fit(X)
        self.means_ = kmeans.cluster_centers_
        
        self.covars_ = np.ones((self.n_components, self.n_features)) * np.var(X, axis=0)

    def _check(self):
        super()._check()
        self.means_ = np.asarray(self.means_)
        self.covars_ = np.asarray(self.covars_)

    def _get_n_fit_scalars_per_param(self):
        nc = self.n_components
        nf = self.n_features
        return {
            "s": nc - 1,
            "t": nc * (nc - 1),
            "m": nc * nf,
            "c": nc * nf,
        }

    def _compute_log_likelihood(self, X):
        n_samples = X.shape[0]
        log_lik = np.zeros((n_samples, self.n_components))
        
        nu = self.df_
        d = self.n_features
        
        log_c = gammaln((nu + d) / 2.0) - gammaln(nu / 2.0) - (d / 2.0) * np.log(nu * np.pi)
        
        for i in range(self.n_components):
            diff = X - self.means_[i]
            # Diagonal covariance calculation
            D2 = np.sum((diff ** 2) / self.covars_[i], axis=1)
            log_det_cov = np.sum(np.log(self.covars_[i]))
            log_lik[:, i] = log_c - 0.5 * log_det_cov - ((nu + d) / 2.0) * np.log(1.0 + D2 / nu)
            
        return log_lik
        
    def _initialize_sufficient_statistics(self):
        stats = super()._initialize_sufficient_statistics()
        stats['post'] = np.zeros(self.n_components)
        stats['obs'] = np.zeros((self.n_components, self.n_features))
        stats['obs_sq'] = np.zeros((self.n_components, self.n_features))
        stats['obs_weight'] = np.zeros(self.n_components)
        return stats
        
    def _accumulate_sufficient_statistics(self, stats, obs, framelogprob, posters, fwdlattice, bwdlattice):
        super()._accumulate_sufficient_statistics(stats, obs, framelogprob, posters, fwdlattice, bwdlattice)
        
        nu = self.df_
        d = self.n_features
        
        for i in range(self.n_components):
            diff = obs - self.means_[i]
            D2 = np.sum((diff ** 2) / self.covars_[i], axis=1)
            
            u_it = (nu + d) / (nu + D2)
            omega_it = posters[:, i] * u_it
            
            stats['post'][i] += posters[:, i].sum()
            stats['obs_weight'][i] += omega_it.sum()
            stats['obs'][i] += np.dot(omega_it, obs)
            stats['obs_sq'][i] += np.dot(omega_it, obs**2)
            
    def _do_mstep(self, stats):
        super()._do_mstep(stats)
        
        for i in range(self.n_components):
            if stats['obs_weight'][i] > 1e-6:
                self.means_[i] = stats['obs'][i] / stats['obs_weight'][i]
                
                raw_cov = (stats['obs_sq'][i] / stats['obs_weight'][i]) - (self.means_[i] ** 2)
                
                # FIX: Add lambda_reg to prevent matrix collapse and stabilize BIC variance
                self.covars_[i] = np.maximum(raw_cov, 1e-4) + self.lambda_reg

    def predict_filtering_proba(self, X):
        """
        CRITICAL ALGORITHMIC FIX: 
        Calculates strictly causal, real-time filtering probabilities P(Z_t | X_1...X_t).
        Bypasses the backward-pass smoothing entirely to eliminate lookahead bias.
        """
        framelogprob = self._compute_log_likelihood(X)
        n_samples, n_components = framelogprob.shape
        log_prob_matrix = np.zeros((n_samples, n_components))
        
        # Safe log of transmat to handle structural zeros
        log_transmat = np.log(np.maximum(self.transmat_, 1e-12))
        
        # Base case (t=0)
        startprob = np.maximum(self.startprob_, 1e-12) if hasattr(self, 'startprob_') else np.ones(n_components)/n_components
        log_prob_matrix[0] = np.log(startprob) + framelogprob[0]
        log_prob_matrix[0] -= logsumexp(log_prob_matrix[0]) # Normalize
        
        # Recursive Forward Pass
        for t in range(1, n_samples):
            log_prior = logsumexp(log_prob_matrix[t-1][:, np.newaxis] + log_transmat, axis=0)
            log_posterior = log_prior + framelogprob[t]
            log_prob_matrix[t] = log_posterior - logsumexp(log_posterior) # Normalize
            
        return np.exp(log_prob_matrix)

    def _generate_sample_from_state(self, state, random_state=None):
        # Implementation required by BaseHMM inheritance, though unused in projection
        return np.zeros(self.n_features)