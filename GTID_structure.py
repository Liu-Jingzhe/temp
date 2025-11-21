import torch
import numpy as np
import pickle


import numpy as np
from sklearn.decomposition import PCA

LAYER_NUM = 3
CODE_NUM = 64


def _pairwise_sq_dists(A, B):
    """Pairwise squared Euclidean distances between rows of A and B."""
    A2 = np.sum(A**2, axis=1, keepdims=True)      # [m, 1]
    B2 = np.sum(B**2, axis=1, keepdims=True).T    # [1, n]
    return A2 + B2 - 2 * (A @ B.T)                # [m, n]

def _rbf_kernel(A, B, sigma):
    """Gaussian (RBF) kernel matrix between rows of A and B."""
    d2 = _pairwise_sq_dists(A, B)
    return np.exp(-d2 / (2.0 * sigma**2))

def _median_heuristic_sigma(Z):
    """Median pairwise distance heuristic for RBF bandwidth."""
    if Z.shape[0] < 2:
        return 1.0
    d2 = _pairwise_sq_dists(Z, Z)
    iu = np.triu_indices_from(d2, k=1)
    dist = np.sqrt(np.maximum(d2[iu], 0.0))
    med = np.median(dist)
    return float(med if med > 0 else 1.0)

def variance_normalized_mmd(X, Y, sigma=None, unbiased=False, return_sigma=True):
    """
    Compute (variance-normalized) MMD^2 between two 7-D groups.

    Args:
        X: np.ndarray of shape [m, 7]
        Y: np.ndarray of shape [n, 7]
        sigma: RBF bandwidth; if None, use median heuristic on stacked data
        unbiased: if True, use unbiased MMD^2 estimator; else biased estimator
        return_sigma: include the sigma used in the output dict

    Returns:
        dict with:
          - 'nMMD2': variance-normalized MMD^2
          - 'MMD2' : raw MMD^2
          - 'sigma': bandwidth used (if return_sigma=True)
    """
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    assert X.ndim == 2 and Y.ndim == 2 and X.shape[1] == 7 and Y.shape[1] == 7, \
        "X, Y must be [num_samples, 7]"

    # Bandwidth
    if sigma is None:
        sigma = _median_heuristic_sigma(np.vstack([X, Y]))

    # Kernel blocks
    Kxx = _rbf_kernel(X, X, sigma)
    Kyy = _rbf_kernel(Y, Y, sigma)
    Kxy = _rbf_kernel(X, Y, sigma)

    m, n = len(X), len(Y)

    # MMD^2
    if unbiased:
        mmd_xx = (np.sum(Kxx) - np.sum(np.diag(Kxx))) / (m * (m - 1)) if m > 1 else 0.0
        mmd_yy = (np.sum(Kyy) - np.sum(np.diag(Kyy))) / (n * (n - 1)) if n > 1 else 0.0
        mmd_xy = (2.0 * np.sum(Kxy)) / (m * n) if (m > 0 and n > 0) else 0.0
        MMD2 = mmd_xx + mmd_yy - mmd_xy
    else:
        MMD2 = Kxx.mean() + Kyy.mean() - 2.0 * Kxy.mean()

    # Variance-normalization denominator (for RBF, diag ≈ 1; compute explicitly)
    Exx = np.mean(np.diag(Kxx)) if m > 0 else 0.0
    Eyy = np.mean(np.diag(Kyy)) if n > 0 else 0.0
    denom = Exx + Eyy  # ~ 2.0 for RBF

    nMMD2 = np.nan if denom <= 1e-12 else MMD2 / denom

    out = {"nMMD2": float(nMMD2), "MMD2": float(MMD2)}
    if return_sigma:
        out["sigma"] = float(sigma)
    return out


with open('/GFT/token_info/structure/structure_ar_ci_arxiv.pkl', 'rb') as f:
    data_1 = pickle.load(f)
with open('/GFT/token_info/structure/structure_ar_ci_pubmed.pkl', 'rb') as f:
    data_2 = pickle.load(f)

for layer in range(LAYER_NUM):
    for code in range(CODE_NUM):
        feature_1 = data_1[layer][code]
        feature_2 = data_2[layer][code]
        if len(feature_1)!=0 and len(feature_2)!=0:
            mmd_results = variance_normalized_mmd(feature_1, feature_2, sigma=None, unbiased=False, return_sigma=True)
            print(f'Layer {layer}, Code {code}, nMMD2: {mmd_results["nMMD2"]}, MMD2: {mmd_results["MMD2"]}, sigma: {mmd_results["sigma"]}')
