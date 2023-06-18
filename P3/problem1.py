import math
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
    x = np.array(x,dtype='float64')
    y = np.array(y, dtype='float64')
    nllh = (x-y)**2/2/sigma_noise**2
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
    edges = np.zeros((2 * (height*width) - (height+width),2),dtype='int64')
    rows = height * (width-1)
    columns = width * (height-1)
    edges[0:rows,:] = [0,1]
    edges[0:rows,:] += np.arange(0,rows).reshape(-1,1)
    edges[rows:,:] = [0,width]
    edges[rows:,:] += np.arange(0,columns).reshape(-1,1)

    assert (edges.shape[0] == 2 * (height*width) - (height+width) and edges.shape[1] == 2)
    assert (edges.dtype in [np.int32, np.int64])
    return edges

def my_sigma():
    return 10

def my_lmbda():
    return 50

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
    source = np.array(noisy, dtype='float64')
    denoised = np.copy(init)
    N = source.shape[0] * source.shape[1]
    unary = np.zeros((2,N))
    pairwise = np.zeros((N,N))

    round = 1
    flag = True
    maxIter = 5
    while(flag):
        print(f" start {round} th Round")
        round +=1
        new_denoised = np.copy(denoised)
        for v in candidate_pixel_values:
            target = np.full(source.shape,v)
            unary[0,:] = mrf_denoising_nllh(new_denoised, source, s).flatten()
            unary[1, :] = mrf_denoising_nllh(new_denoised, target, s).flatten()

            new_denoised = new_denoised.flatten()
            for e in edges.tolist():
                if new_denoised[e[0]] != new_denoised[e[1]]:
                    pairwise[e[0],e[1]] = lmbda
            labels = gco.graphcut(unary, csr_matrix(pairwise))
            new_denoised[labels==0] = source.flatten()[labels==0]
            new_denoised[labels==1] = v
            new_denoised = new_denoised.reshape(source.shape)
        if not (np.count_nonzero(new_denoised!=denoised)>0):
            flag = False
        denoised = new_denoised

        if round > maxIter:
            break

    assert (np.equal(denoised.shape, init.shape).all())
    assert (denoised.dtype == init.dtype)
    return denoised

def compute_psnr(img1, img2):
    """Computes PSNR b/w img1 and img2"""
    H,W = img1.shape
    v_MAX = 255
    MSE =  np.sum((img1 - img2)**2) / (H*W)
    psnr = 10 * np.log10(v_MAX**2/MSE)
    return psnr

def show_images(i0, i1):
    """
    Visualize estimate and ground truth in one Figure.
    Only show the area for valid gt values (>0).
    """

    # Crop images to valid ground truth area
    row, col = np.nonzero(i0)
    plt.figure()
    plt.subplot(1,2,1)
    plt.imshow(i0, "gray", interpolation='nearest')
    plt.subplot(1,2,2)
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
    estimated = alpha_expansion(noisy, random_init, edges, labels, s, lmbda)
    show_images(noisy, estimated)
    psnr_before = compute_psnr(noisy, gt)
    psnr_after = compute_psnr(estimated, gt)
    print(psnr_before, psnr_after)
