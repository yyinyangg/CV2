import math
import random

import gco
import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import csr_matrix

np.random.seed(seed=2022)


def mrf_denoising_nllh(x, y, sigma_noise):
    """Elementwise negative log likelihood.

      Args:
        x: candidate denoised image
        y: noisy image
        sigma_noise: noise level for Gaussian noise

      Returns:
        A `nd.array` with dtype `float32/float64`.
    """
    assert np.shape(x) == np.shape(y)
    H, W = np.shape(x)
    nllh = np.zeros(H * W, dtype=np.float32)
    for i in range(H):
        for j in range(W):
            nllh[i * W + j] = -0.5 / sigma_noise ** 2 * (x[i, j] - y[i, j]) ** 2
    assert (nllh.dtype in [np.float32, np.float64])
    return nllh


def edges4connected(height, width):
    """Construct edges for 4-connected neighborhood MRF.
    The output representation is such that output[i] specifies two indices
    of connected nodes in an MRF stored with row-major ordering.

      Args:
        height, width: size of the MRF.

      Returns:
        A `nd.array` with dtype `int32/int64` of size |E| x 2.
    """

    edge = []
    for i in range(height):
        for j in range(width - 1):
            index = i * width + j
            edge.append([index, index + 1])

    for i in range(height - 1):
        for j in range(width):
            index = i * width + j
            edge.append([index, index + width])

    edges = np.array(edge)
    assert (edges.shape[0] == 2 * (height * width) - (height + width) and edges.shape[1] == 2)
    assert (edges.dtype in [np.int32, np.int64])
    return edges


def my_sigma():
    return 8


def my_lmbda():
    return 8


def alpha_expansion(noisy, init, edges, candidate_pixel_values, s, lmbda):
    """ Run alpha-expansion algorithm.

      Args:
        noisy: Given noisy grayscale image.
        init: Image for denoising initilisation
        edges: Given neighboor of MRF.
        candidate_pixel_values: Set of labels to consider
        s: sigma for likelihood estimation
        lmbda: Regularization parameter for Potts model.

      Runs through the set of candidates and iteratively expands a label.
      If there have been recorded changes, re-run through the complete set of candidates.
      Stops, if there are no changes in the labelling.

      Returns:
        A `nd.array` of type `int32`. Assigned labels minimizing the costs.
    """

    H, W = np.shape(noisy)
    unary = np.zeros((2, H * W))
    flatten_random_Init = init.flatten()
    pairwise = np.zeros([H * W, H * W])
    for i in range(H):
        for j in range(W):
            if (j < W - 1):
                if (init[i, j] != init[i, j + 1]):
                    row_idx, column_idx = edges[i * (W - 1) + j]
                    pairwise[row_idx, column_idx] = lmbda
            if (i < H - 1):
                if (init[i, j] != init[i + 1, j]):
                    row_idx, column_idx = edges[H * (W - 1) + i * W + j]
                    pairwise[row_idx, column_idx] = lmbda
    sparse_pairwise = csr_matrix(pairwise)
    for i in range(3):
        for alpha in candidate_pixel_values:
            alpha_array = np.full((H, W), alpha)
            print('run with alpha =', alpha)
            unary[0] = mrf_denoising_nllh(init, noisy, s)
            unary[1] = mrf_denoising_nllh(alpha_array,noisy, s)
            result = gco.graphcut(unary, sparse_pairwise)
            mask = result == 0
            flatten_random_Init[mask] = alpha
            init = flatten_random_Init.reshape((H, W))
    denoised = flatten_random_Init.reshape((H, W))
    assert (np.equal(denoised.shape, init.shape).all())
    assert (denoised.dtype == init.dtype)
    return denoised


def compute_psnr(img1, img2):
    """Computes PSNR b/w img1 and img2"""
    assert np.shape(img2) == np.shape(img1)
    H, W = np.shape(img1)
    MSE = 0
    for i in range(H):
        for j in range(W):
            MSE += 1 / (H * W) * pow(img1[i, j] - img2[i, j], 2)
    v_max = max(np.max(img1), np.max(img2))
    psnr = 10 * math.log10(v_max ** 2 / MSE)
    return psnr


def show_images(i0, i1):
    """
    Visualize estimate and ground truth in one Figure.
    Only show the area for valid gt values (>0).
    """

    # Crop images to valid ground truth area
    row, col = np.nonzero(i0)
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(i0, "gray", interpolation='nearest')
    plt.subplot(1, 2, 2)
    plt.imshow(i1, "gray", interpolation='nearest')
    plt.show()


# Example usage in main()
# Feel free to experiment with your code in this function
# but make sure your final submission can execute this code
if __name__ == '__main__':
    # Read images
    noisy = ((255 * plt.imread('data/la-noisy.png')).squeeze().astype(np.int32)).astype(np.float32)
    gt = (255 * plt.imread('data/la.png')).astype(np.int32)

    lmbda = my_lmbda()
    s = my_sigma()

    # Create 4 connected edge neighborhood
    edges = edges4connected(noisy.shape[0], noisy.shape[1])

    # Candidate search range
    labels = np.arange(0, 255)

    # Graph cuts with random initialization
    random_init = np.random.randint(low=0, high=255, size=noisy.shape)
    print('test')
    estimated = alpha_expansion(noisy, random_init, edges, labels, s, lmbda)
    show_images(noisy, estimated)
    psnr_before = compute_psnr(noisy, gt)
    psnr_after = compute_psnr(estimated, gt)
    print(psnr_before, psnr_after)