import numpy as np

from scipy import interpolate  # Use this for interpolation
from scipy import signal  # Feel free to use convolutions, if needed
from scipy import ndimage
from scipy import optimize  # For gradient-based optimisation
from PIL import Image  # For loading images
import matplotlib.pyplot as plt

# for experiments with different initialisation
from problem1 import random_disparity
from problem1 import constant_disparity

np.random.seed(seed=2023)


def rgb2gray(rgb):
    """Converting RGB image to greyscale.
    The same as in Assignment 1 (no graded here).

    Args:
        rgb: numpy array of shape (H, W, 3)

    Returns:
        gray: numpy array of shape (H, W)

    """
    H, W, channel = rgb.shape
    gray = np.zeros((H, W))
    for i in range(H):
        for j in range(W):
            gray[i, j] = (rgb[i, j, 0] * 1913 + rgb[i, j, 1] * 4815 + rgb[i, j, 2] * 3272) / 10000
    return gray


def load_data(i0_path, i1_path, gt_path):
    """Loading the data.
    The same as in Assignment 1 (not graded here).

    Args:
        i0_path: path to the first image
        i1_path: path to the second image
        gt_path: path to the disparity image

    Returns:
        i_0: numpy array of shape (H, W)
        i_1: numpy array of shape (H, W)
        g_t: numpy array of shape (H, W)
    """
    i_0 = np.array(Image.open(i0_path), dtype='float64')
    i_1 = np.array(Image.open(i1_path), dtype='float64')
    g_t = np.array(Image.open(gt_path), dtype='float64')

    i_0 = i_0 / i_0.max()
    i_1 = i_1 / i_1.max()


    return i_0, i_1, g_t


def log_gaussian(x,  mu, sigma):
    """Calcuate the value and the gradient w.r.t. x of the Gaussian log-density

    Args:
        x: numpy.float 2d-array
        mu and sigma: scalar parameters of the Gaussian

    Returns:
        value: value of the log-density
        grad: gradient of the log-density w.r.t. x
    """

    value = -0.5*((x-mu)/(sigma))**2
    grad = -(x-mu)/sigma**2

    return value, grad

def stereo_log_prior(x, mu, sigma):
    """Evaluate gradient of pairwise MRF log prior with Gaussian distribution

    Args:
        x: numpy.float 2d-array (disparity)

    Returns:
        value: value of the log-prior
        grad: gradient of the log-prior w.r.t. x
    """
    value = 0
    H,W = np.shape(x)
    grad = np.zeros((H,W))
    for i in range(H):
         for j in range(W):
             neighbors = []
             if i > 0:
                 neighbors.append(x[i-1,j]) # upward pixel
             if i < H-1:
                 neighbors.append(x[i+1,j]) # downward pixel
             if j > 0:
                 neighbors.append(x[i,j-1]) # left pixel
             if j < W-1:
                 neighbors.append(x[i,j+1]) # right pixel
             value += sum(log_gaussian(x_neighbors-x[i,j], mu,sigma)[0] for x_neighbors in neighbors)
             grad[i,j] = sum(log_gaussian(x_neighbors-x[i,j],mu,sigma)[1] for x_neighbors in neighbors)
    value = value/2

    return  value, grad

def shift_interpolated_disparity(im1, d):
    """Shift image im1 by the disparity value d.
    Since disparity can now be continuous, use interpolation.

    Args:
        im1: numpy.float 2d-array  input image
        d: numpy.float 2d-array  disparity

    Returns:
        im1_shifted: Shifted version of im1 by the disparity value.
    """
    H, W = im1.shape
    shifted_im1 = np.zeros([H, W])
    xs = np.arange(W)
    for i in range(H):
        ys = im1[i, :].reshape(W)
        interp_func = interpolate.interp1d(xs, ys, kind='cubic')
        for j in range(W):
            x_new = j - d[i, j]
            if x_new > W - 1:
                shifted_im1[i, j] = 0
            elif x_new < 0:
                shifted_im1[i, j] = 0
            else:
                shifted_im1[i, j] = interp_func(x_new)

    return shifted_im1


def stereo_log_likelihood(x, im0, im1, mu, sigma):
    """Evaluate gradient of the log likelihood.

    Args:
        x: numpy.float 2d-array of the disparity
        im0: numpy.float 2d-array of image #0
        im1: numpy.float 2d-array of image #1

    Returns:
        value: value of the log-likelihood
        grad: gradient of the log-likelihood w.r.t. x

    Hint: Make use of shift_interpolated_disparity and log_gaussian
    """
    H,W = np.shape(im0)
    grad = np.zeros((H,W))
    shifted_im1 = shift_interpolated_disparity(im1, x)
    diff = im0 - shifted_im1
    ''''''
    logp = -0.5 * (diff ** 2) / sigma ** 2
    value = np.sum(logp)

    kernel = np.array([1, 0, -1]) / 2
    derivateH = ndimage.convolve1d(shifted_im1, kernel, axis=1, mode="constant")
    for i in range(H):
        for j in range (W):
            grad[i,j] = (im0[i,j]-shifted_im1[i,j] / sigma ** 2) * derivateH[i,j]

    return value, grad


def stereo_log_posterior(d, im0, im1, mu, sigma, alpha):
    """Computes the value and the gradient of the log-posterior

    Args:
        d: numpy.float 2d-array of the disparity
        im0: numpy.float 2d-array of image #0
        im1: numpy.float 2d-array of image #1

    Returns:
        value: value of the log-posterior
        grad: gradient of the log-posterior w.r.t. x
    """
    d = d.reshape(im1.shape)
    log_prior, log_prior_grad = stereo_log_prior(d, mu, sigma)
    log_likelihood, log_likelihood_grad = stereo_log_likelihood(d, im0, im1, mu, sigma)
    log_posterior = log_likelihood + alpha * log_prior
    log_posterior_grad = log_likelihood_grad + alpha * log_prior_grad
    print(-log_posterior)
    return log_posterior, log_posterior_grad


def optim_method():
    """Simply returns the name (string) of the method
    accepted by scipy.optimize.minimize, that you found
    to work well.
    This is graded with 1 point unless the choice is arbitrary/poor.
    """
    return "L-BFGS-B"
    # return "BFGS"


def stereo(d0, im0, im1, mu, sigma, alpha, method=optim_method()):
    """Estimating the disparity map

    Args:
        d0: numpy.float 2d-array initialisation of the disparity
        im0: numpy.float 2d-array of image #0
        im1: numpy.float 2d-array of image #1

    Returns:
        d: numpy.float 2d-array estimated value of the disparity
    """

    def fun(args):
        im0, im1, mu, sigma, alpha = args
        v = lambda d: -stereo_log_posterior(d, im0, im1, mu, sigma, alpha)[0]
        return v

    def grad(args):
        im0, im1, mu, sigma, alpha = args
        v = lambda d: stereo_log_posterior(d, im0, im1, mu, sigma, alpha)[1].flatten()
        return v

    args = (im0, im1, mu, sigma, alpha)

    res = optimize.minimize(fun(args), d0.flatten(), method=method, jac=grad(args), options={'gtol':1e-3})
    ''''''
    print(res.success)


    return res.x.reshape(im1.shape)


def coarse2fine(d0, im0, im1, mu, sigma, alpha, num_levels):
    """Coarse-to-fine estimation strategy. Basic idea:
        1. create an image pyramid (of size num_levels)
        2. starting with the lowest resolution, estimate disparity
        3. proceed to the next resolution using the estimated
        disparity from the previous level as initialisation

    Args:
        d0: numpy.float 2d-array initialisation of the disparity
        im0: numpy.float 2d-array of image #0
        im1: numpy.float 2d-array of image #1

    Returns:
        pyramid: a list of size num_levels containing the estimated
        disparities at each level (from finest to coarsest)
        Sanity check: pyramid[0] contains the finest level (highest resolution)
                      pyramid[-1] contains the coarsest level
    """

    # Reference CV1:
    def gauss2d(sigma, fsize):
        """ Create a 2D Gaussian filter
        """
        W, H = fsize
        x_limit = (int)(W / 2)
        y_limit = (int)(H / 2)
        X = np.linspace(-x_limit, x_limit, W)
        Y = np.linspace(-y_limit, y_limit, H)
        x, y = np.meshgrid(X, Y)

        gauss = np.exp(- ((x) ** 2 + (y) ** 2) / (2 * sigma ** 2)) / (2 * np.pi * sigma ** 2)
        normalized_gauss = gauss / np.sum(gauss)

        return normalized_gauss.T

    # Reference CV1:
    def downsample2(img, f):
        """ Downsample image by a factor of 2
        Filter with Gaussian filter then take every other row/column
        """
        img_filted = ndimage.convolve(img, f)
        get_img = img_filted[1::2, 1::2]
        return get_img

    # Reference CV1
    def binomial2d(fsize):
        """ Create a 2D binomial filter
        """
        '''Reference: https://stackoverflow.com/questions/56246970/how-to-apply-a-binomial-low-pass-filter-to-data-in-a-numpy-array'''
        W, H = fsize
        x = np.array(np.poly1d([1, 1]) ** (W - 1))
        y = np.array(np.poly1d([1, 1]) ** (H - 1))
        size_x = np.shape(x)[0]
        size_y = np.shape(y)[0]
        x = x.reshape(size_x, 1)
        y = y.reshape(1, size_y)
        kernel = x.dot(y)
        norm_kernel = kernel / np.sum(kernel)
        return norm_kernel.T

    # Reference CV1
    def upsample2(img, f):
        """ Upsample image by factor of 2
        """
        W, H = np.shape(img)
        upsampled_image = np.zeros((2 * W, 2 * H))
        upsampled_image[0::2, 0::2] = img
        upsampled_image = ndimage.convolve(upsampled_image, f * 4)
        return upsampled_image

    fsize = (5, 5)
    kernelsigma = 1.4
    gf = gauss2d(kernelsigma, fsize)
    bf = binomial2d(fsize)
    img0_pyramid = []
    img1_pyramid = []
    img0_pyramid.append(im0)
    img1_pyramid.append(im1)

    rtn = []

    for i in range(num_levels - 1):
        get_result = downsample2(img0_pyramid[-1], gf)
        img0_pyramid.append(get_result)
        get_result = downsample2(img1_pyramid[-1], gf)
        img1_pyramid.append(get_result)
        d0 = downsample2(d0, gf)
    for j in range(num_levels):
        rtn.insert(0, d0)
        disparity = stereo(d0, img0_pyramid[num_levels - 1 - j], img1_pyramid[num_levels - 1 - j], mu, sigma, alpha)
        d0 = upsample2(disparity, bf)

    return rtn


# Example usage in main()
# Feel free to experiment with your code in this function
# but make sure your final submission can execute this code
def main():
    # these are the same functions from Assignment 1
    # (no graded in this assignment)
    im0, im1, gt = load_data('./data/i0.png', './data/i1.png', './data/gt.png')
    im0, im1 = rgb2gray(im0), rgb2gray(im1)

    mu = 0.0
    sigma = 1.7

    # experiment with other values of alpha
    alpha = 1.0

    # initial disparity map
    # experiment with constant/random values
    #d0 = gt
    d0 = random_disparity(gt.shape)

    #d0 = constant_disparity(gt.shape, 6)

    # Display stereo: Initialized with noise
    plt.imshow(d0,cmap='gray')
    plt.show()
    disparity = stereo(d0, im0, im1, mu, sigma, alpha)
    plt.imshow(disparity,cmap='gray')
    plt.show()

    # Pyramid
    '''
    print("sum of GT", np.sum(gt))
    print("sum of d0", np.sum(d0))
    print("sum of d* with out pyramid", np.sum(disparity))
    print("difference with out pyramid", np.sum(disparity-gt))
    '''
    num_levels = 3
    #pyramid = coarse2fine(d0, im0, im1, mu, sigma, alpha, num_levels)
    '''
    print("sum of d*: ", np.sum(pyramid[0]))
    print("the difference is: ", np.sum(pyramid[0]-gt))
    '''


if __name__ == "__main__":
    '''
    i) initialise the d0 using gt, the Algorithm ends up near gt, indicating that gt is already the optimal answer.
    ii) initialise the d0 using constant value, the Algorithm stops near d0, showing no optimization effect, 
    cause the constant disparity indicates the highest compatibility, meaning every pixel is correlated to its neighbor.
    based on the Observation, we can make the conclusion, that constant value is not a good Initialisation.
    iii) initialise the d0 using random value, the Algorithm converge towards GT, showing an optimization effect,
    we can see the random value is a better initialization Option.
    with the increase of alpha , in our result we can see the boundary of object becomes sharper.
    

    '''
    main()