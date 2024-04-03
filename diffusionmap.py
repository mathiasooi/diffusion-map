import numpy as np
from scipy.linalg import eigh
import plotly.graph_objs as go
from scipy.spatial import distance_matrix
from plotly.subplots import make_subplots

from sklearn.datasets import make_swiss_roll

def diffusion_map(data, n_evecs, t=1, alpha=1):
    """
    Find diffusion map using the first n_evecs dimensions in diffusion space

    t     - time step
    alpha - normalization parameter for guassian kernel function
    """
    
    # calculate pairwise euclidean distances
    M = distance_matrix(data, data)

    # apply guassian kernel
    # transition probability from a->b is proportional to exp(-dist^2/alpha)
    K = np.exp(-M**2 / alpha)

    # markov matrix
    r = np.sum(K, axis=0)
    Di = np.diag(1/r)
    P = np.matmul(Di, K)
    P = np.linalg.matrix_power(P, t)

    D_right = np.diag((r)**0.5)
    D_left = np.diag((r)**-0.5)
    P_prime = np.matmul(D_right, np.matmul(P,D_left))


    # eigendecomposition
    eigen_values, eigen_vectors = eigh(P_prime)

    # sort eigenvectors by largest -> smallest
    idx = eigen_values.argsort()[::-1]
    eigen_values = eigen_values[idx]
    eigen_vectors = eigen_vectors[:,idx]

    # diffusion cords
    diffusion_coordinates = np.matmul(D_left, eigen_vectors)

    d_map = diffusion_coordinates[:, 1 :n_evecs+1]
    return d_map

noises = [0, 0.3, 0.5, 0.7]

fig = make_subplots(
    rows=2, cols=4,
    subplot_titles=([f"noise = {i}" for i in noises]),
    specs=[[{"type": "Scatter3d"}, {"type": "Scatter3d"}, {"type": "Scatter3d"}, {"type": "Scatter3d"}],
           [{"type": "Scatter"}, {"type": "Scatter"}, {"type": "Scatter"}, {"type": "Scatter"}]]
                    )
for i, noise in enumerate(noises):
    print("Processing noise =", noise)
    swiss_roll, _ = make_swiss_roll(n_samples=1500, noise=noise)
    d_map = diffusion_map(swiss_roll, 2, t=1, alpha=0.9)
    fig.add_trace(go.Scatter3d(x=swiss_roll[:, 0], y=swiss_roll[:, 1], z=swiss_roll[:, 2], mode="markers", marker=dict(
        size=4,color=d_map[:, 0],opacity=1,colorscale="Viridis"
    )), row=1, col=i+1)
    fig.add_trace(go.Scatter(x=d_map[:, 0], y=d_map[:, 1], mode="markers", marker=dict(
        size=4,color=d_map[:, 0],opacity=1,colorscale="Viridis"
    )), row=2, col=i+1)

fig.show()