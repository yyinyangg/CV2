from PIL import Image
import numpy as np

np.random.seed(seed=2023)


def load_data(gt_path):
    """Loading the data.
    The same as loading the disparity from Assignment 1 
    (not graded here).

    Args:
        gt_path: path to the disparity image

    Returns:
        g_t: numpy array of shape (H, W)
    """
    g_t = Image.open(gt_path)
    g_t = np.array(g_t,dtype='float64')
    #print(g_t.max())
    return g_t


def random_disparity(disparity_size):
    """Compute a random disparity map.

    Args:
        disparity_size: tuple containg height and width (H, W)

    Returns:
        disparity_map: numpy array of shape (H, W)
    """
    disparity_map = np.random.randint(low=0, high=13, size=disparity_size, dtype=int)
    return disparity_map


def constant_disparity(disparity_size, a):
    """Compute a constant disparity map.

    Args:
        disparity_size: tuple containg height and width (H, W)
        a: the value to initialize with

    Returns:
        disparity_map: numpy array of shape (H, W)

    """
    disparity_map = np.full(disparity_size, a,dtype=float)
    return disparity_map


def log_gaussian(x, mu, sigma):
    """Compute the log gaussian of x.

    Args:
        x: numpy array of shape (H, W) (np.float64)
        mu: float
        sigma: float

    Returns:
        result: numpy array of shape (H, W) (np.float64)
    """
    result = -0.5 * ((x - mu) ** 2 / sigma ** 2)
    return result


def mrf_log_prior(x, mu, sigma):
    """Compute the log of the unnormalized MRF prior density.

    Args:
        x: numpy array of shape (H, W) (np.float64)
        mu: float
        sigma: float

    Returns:
        logp: float

    """

    H, W = x.shape
    diff_H = np.zeros((H, W - 1))
    diff_V = np.zeros((H - 1, W))
    for i, row in enumerate(np.split(x, H, axis=0)):

        if i == 0:
            old_row = row
        diff_V[i - 1, :] = row - old_row
        old_row = row

    for j, column in enumerate(np.split(x, W, axis=1)):
        if j == 0:
            old_column = column
        diff_H[:, j - 1] = (column - old_column).reshape(H)
        old_column = column
    logp = np.sum(log_gaussian(diff_H, mu, sigma)) + np.sum(log_gaussian(diff_V, mu, sigma))
    return logp


# Example usage in main()
# Feel free to experiment with your code in this function
# but make sure your final submission can execute this code
def main():
    gt = load_data('./data/gt.png')

    # Display log prior of GT disparity map
    logp = mrf_log_prior(gt, mu=0, sigma=1.1)
    print("Log Prior of GT disparity map:", logp)

    # Display log prior of random disparity ma
    random_disp = random_disparity(gt.shape)
    logp = mrf_log_prior(random_disp, mu=0, sigma=1.1)
    print("Log-prior of noisy disparity map:", logp)

    # Display log prior of constant disparity map
    constant_disp = constant_disparity(gt.shape, 6)
    logp = mrf_log_prior(constant_disp, mu=0, sigma=1.1)
    print("Log-prior of constant disparity map:", logp)

    '''
    the values of log-prior density indicate the Compatibility of the neighboring pixels of one image.
    constant disparity map shows the highest Compatibility,its Log-prior = 0, cause every pixel has the same disparity.
    due to the random noisy pixel, which is not correlated to its neighbor, 
    the noisy disparity map shows the lowest Compatibility its Log-prior = -2559991.
    GT disparity map shows moderate Compatibility, because the pixels belonging to one Object are correlated to the neighbors.
    
    increasing the sigma will also increases the log-prior density,GT-map from -50685 to -15332. 
    the Gaussian curve becomes smoother, pixels with different disparity will be considered as more similar.
    
    reducing the range of noise increase the log-prior density. noisy-map from -2559991 to -729330.
    because the effect of noise is milder, the pixels are more similar namely more correlated to each other.
    '''


if __name__ == "__main__":
    main()
