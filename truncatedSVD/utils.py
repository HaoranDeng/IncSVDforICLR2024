from numpy.linalg import svd
import matplotlib.pyplot as plt
import pickle
import os
from os import mkdir
from os.path import normpath, exists, join
import numpy as np
from sklearn.metrics import precision_score, recall_score
import scipy.sparse

def compute_scores(y_true, y_pred):
    p_score = precision_score(y_true, y_pred)
    r_score = recall_score(y_true, y_pred)
    return p_score, r_score

def predict(y_scores, threholds):
    return (y_scores >= threholds) * 1

def p_r_curve(y_true, y_scores):
    thresholds = sorted(np.unique(y_scores))
    precisions, recalls = [], []
    for thre in thresholds:
        y_pred = predict(y_scores, thre)
        r = compute_scores(y_true, y_pred)
        precisions.append(r[0])
        recalls.append(r[1])

    last_ind = np.searchsorted(recalls[::-1], recalls[0]) + 1
    precisions = precisions[-last_ind:]
    recalls = recalls[-last_ind:]
    thresholds = thresholds[-last_ind:]
    precisions.append(1)
    recalls.append(0)
    return precisions, recalls, thresholds

def check_and_create_dir(dirname):
    """Check if directory exists. If it does not exist, create it.
    
    Parameters
    ----------
    dirname : str
        Name of directory
    """
    if not exists(normpath(dirname)):
        mkdir(normpath(dirname))

        
def get_truncated_svd(A, k):
    """Get truncated SVD using a deterministic method.
    
    Parameters
    ----------
    A : ndarray of shape (m, n)
        Real matrix
    
    k : int 
        Rank
    """
    # print(type(A))
    if isinstance(A, scipy.sparse._csr.csr_matrix):
        pass
        u, s, vh = scipy.sparse.linalg.svds(A, k)
    else:
        u, s, vh = svd(A, full_matrices=False)
    return u[:, :k], s[:k], vh[:k, :]


def save_result(filename, key, value):
    import os
    import pickle
    if not os.path.exists(filename):
        res = {}
        with open(filename, "wb") as f:
            pickle.dump(res, f)
    
    with open(filename, "rb") as f:
        res = pickle.load(f)
        res[key] = value
    with open(filename, "wb") as f:
        pickle.dump(res, f)
    
def init_figure(title, xlabel, ylabel, yscale="log", figsize=(4, 3), fontsize="medium"):
    """Initialize matplotlib figure
    
    Parameters
    ----------
    title : str
        Figure title
    
    xlabel : str
        x-axis label
    
    ylabel : str
        y-axis label
    
    yscale : str, default="log"
        y-axis scale
    
    figsize : tuple, default=(4, 3)
        Figure size
    
    Returns
    -------
    fig : Figure
        matploltib Figure object
        
    ax : axes
        Matplotlib Axes object
    """
    # Initialize figure
    fig, ax = plt.subplots(figsize=figsize)
    ax.grid(True, which="both", linewidth=1, linestyle="--", color="k", alpha=0.1)
    ax.tick_params(
        which="both", direction="in", bottom=True, top=True, left=True, right=True
    )

    # Label figure
    ax.set_title(title, fontsize=fontsize)
    ax.set_xlabel(xlabel, fontsize=fontsize)
    ax.set_ylabel(ylabel, fontsize=fontsize)
    ax.set_yscale(yscale)
    ax.tick_params(axis='both', which='major', labelsize=fontsize)

    return fig, ax