import torch
import numpy as np
import pickle


import numpy as np
from sklearn.decomposition import PCA

LAYER_NUM = 3
CODE_NUM = 64


def _pairwise_sq_dists(A, B):
    """Compute pairwise squared Euclidean distances between rows of A and B."""
    # A: [m, d], B: [n, d]
    A2 = np.sum(A**2, axis=1, keepdims=True)       # [m, 1]
    B2 = np.sum(B**2, axis=1, keepdims=True).T     # [1, n]
    return A2 + B2 - 2 * (A @ B.T)                 # [m, n]

def _rbf_kernel(A, B, sigma):
    """Gaussian (RBF) kernel matrix between rows of A and B."""
    d2 = _pairwise_sq_dists(A, B)
    return np.exp(-d2 / (2.0 * sigma**2))

def _median_heuristic_sigma(Z):
    """Median pairwise distance heuristic for RBF bandwidth."""
    # Z: [N, d]
    if Z.shape[0] < 2:
        return 1.0
    d2 = _pairwise_sq_dists(Z, Z)
    # take upper triangle without diagonal
    iu = np.triu_indices_from(d2, k=1)
    dist = np.sqrt(np.maximum(d2[iu], 0.0))
    med = np.median(dist)
    return float(med if med > 0 else 1.0)

def variance_normalized_mmd_with_pca(X, Y, pca_dim=7, sigma=None, unbiased=False, return_sigma=True):
    """
    Compute (variance-normalized) MMD^2 between two groups after PCA to `pca_dim`.

    Args:
        X: np.ndarray of shape [m, 384]
        Y: np.ndarray of shape [n, 384]
        pca_dim: target PCA dimension (default 7)
        sigma: RBF bandwidth; if None, uses median heuristic on combined PCA features
        unbiased: if True, use the unbiased MMD^2 estimator; else uses the biased estimator
        return_sigma: if True, include the sigma used in the outputs

    Returns:
        results: dict with keys:
            - 'nMMD2': variance-normalized MMD^2
            - 'MMD2': raw MMD^2
            - 'sigma': bandwidth used (if return_sigma)
    """
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    assert X.ndim == 2 and Y.ndim == 2, "X, Y must be [num_samples, feature_dim]"
    assert X.shape[1] == 384 and Y.shape[1] == 384, "Expect 384-d inputs per your spec"

    # 1) PCA on combined data
    Z = np.vstack([X, Y])
    pca = PCA(n_components=pca_dim, svd_solver='auto', whiten=False, random_state=0)
    Zp = pca.fit_transform(Z)
    Xp = Zp[:len(X)]
    Yp = Zp[len(X):]

    # 2) Choose sigma if not provided (median heuristic on combined PCA features)
    if sigma is None:
        sigma = _median_heuristic_sigma(Zp)

    # 3) Kernel blocks
    Kxx = _rbf_kernel(Xp, Xp, sigma)
    Kyy = _rbf_kernel(Yp, Yp, sigma)
    Kxy = _rbf_kernel(Xp, Yp, sigma)

    m, n = len(Xp), len(Yp)

    # 4) MMD^2 (biased or unbiased)
    if unbiased:
        # Unbiased: exclude diagonals for Kxx, Kyy
        if m > 1:
            mmd_xx = (np.sum(Kxx) - np.sum(np.diag(Kxx))) / (m * (m - 1))
        else:
            mmd_xx = 0.0
        if n > 1:
            mmd_yy = (np.sum(Kyy) - np.sum(np.diag(Kyy))) / (n * (n - 1))
        else:
            mmd_yy = 0.0
        mmd_xy = (2.0 * np.sum(Kxy)) / (m * n)
        MMD2 = mmd_xx + mmd_yy - mmd_xy
    else:
        # Biased: simple means (includes diagonals)
        MMD2 = Kxx.mean() + Kyy.mean() - 2.0 * Kxy.mean()

    # 5) Variance-normalization:
    # For RBF kernels, k(x,x) = 1, so E[k(x,x)] ≈ 1 and E[k(y,y)] ≈ 1.
    # We compute them explicitly (robust if kernel changes).
    Exx = np.mean(np.diag(Kxx)) if m > 0 else 0.0
    Eyy = np.mean(np.diag(Kyy)) if n > 0 else 0.0
    denom = Exx + Eyy  # ≈ 2.0 for RBF

    # Guard against degenerate denom
    if denom <= 1e-12:
        nMMD2 = np.nan
    else:
        nMMD2 = MMD2 / denom

    out = {"nMMD2": float(nMMD2), "MMD2": float(MMD2)}
    if return_sigma:
        out["sigma"] = float(sigma)
    return out



# --------- utilities ----------
def zscore_combine(X, Y, eps=1e-12):
    Z = np.vstack([X, Y])
    mu = Z.mean(axis=0, keepdims=True)
    sd = Z.std(axis=0, keepdims=True)
    sd = np.where(sd < eps, 1.0, sd)
    return (X - mu) / sd, (Y - mu) / sd

def pairwise_sq_dists(A, B):
    A2 = np.sum(A**2, axis=1, keepdims=True)
    B2 = np.sum(B**2, axis=1, keepdims=True).T
    return A2 + B2 - 2 * (A @ B.T)

def median_distance(Z):
    if Z.shape[0] < 2:
        return 1.0
    d2 = pairwise_sq_dists(Z, Z)
    iu = np.triu_indices_from(d2, k=1)
    return float(np.median(np.sqrt(np.maximum(d2[iu], 0.0))))

def pick_sigma_dim_norm(Z, c=1.0):
    n, d = Z.shape
    med = median_distance(Z)
    return c * med / np.sqrt(d if d > 0 else 1)

def rbf_kernel(A, B, sigma):
    d2 = pairwise_sq_dists(A, B)
    return np.exp(-d2 / (2.0 * sigma**2))

def mmd2_blocks(Kxx, Kyy, Kxy, unbiased=False):
    m, n = Kxx.shape[0], Kyy.shape[0]
    if unbiased:
        xx = (np.sum(Kxx) - np.sum(np.diag(Kxx))) / (m * (m - 1)) if m > 1 else 0.0
        yy = (np.sum(Kyy) - np.sum(np.diag(Kyy))) / (n * (n - 1)) if n > 1 else 0.0
        xy = (2.0 * np.sum(Kxy)) / (m * n) if (m > 0 and n > 0) else 0.0
        return xx + yy - xy
    else:
        return Kxx.mean() + Kyy.mean() - 2.0 * Kxy.mean()

def nmmd2_for_space(X, Y, c=1.0, unbiased=False, match_sizes=True, seed=0):
    rng = np.random.default_rng(seed)
    Xz, Yz = zscore_combine(X, Y)

    if match_sizes:
        m, n = len(Xz), len(Yz)
        k = min(m, n)
        Xz = Xz[rng.choice(m, size=k, replace=False)]
        Yz = Yz[rng.choice(n, size=k, replace=False)]

    Z = np.vstack([Xz, Yz])
    sigma = pick_sigma_dim_norm(Z, c=c)

    Kxx = rbf_kernel(Xz, Xz, sigma)
    Kyy = rbf_kernel(Yz, Yz, sigma)
    Kxy = rbf_kernel(Xz, Yz, sigma)

    MMD2 = mmd2_blocks(Kxx, Kyy, Kxy, unbiased=unbiased)

    # Variance-normalization (for RBF, diag ≈ 1, but compute explicitly)
    Exx = float(np.mean(np.diag(Kxx))) if len(Xz) else 0.0
    Eyy = float(np.mean(np.diag(Kyy))) if len(Yz) else 0.0
    denom = Exx + Eyy
    nMMD2 = np.nan if denom <= 1e-12 else MMD2 / denom

    return {"nMMD2": float(nMMD2), "MMD2": float(MMD2), "sigma": float(sigma)}




    
with open('/GFT/token_info/structure/structure_ar_ci_arxiv.pkl', 'rb') as f:
    data_1 = pickle.load(f)
with open('/GFT/token_info/structure/structure_ar_ci_pubmed.pkl', 'rb') as f:
    data_2 = pickle.load(f)

for layer in range(LAYER_NUM):
    for code in range(CODE_NUM):
        feature_1 = data_1[layer][code]
        feature_2 = data_2[layer][code]
        if len(feature_1)!=0 and len(feature_2)!=0:
            mmd_results = nmmd2_for_space(feature_1, feature_2)
            print(f'Layer {layer}, Code {code}, nMMD2: {mmd_results["nMMD2"]}, MMD2: {mmd_results["MMD2"]}, sigma: {mmd_results["sigma"]}')
