
from PIL import Image
import numpy as np

np.random.seed(seed=2023)

# convert a RGB image to grayscale
# input (rgb): numpy array of shape (H, W, 3)
# output (gray): numpy array of shape (H, W)
def rgb2gray(rgb):

	##############################################################################################
	#										IMPLEMENT											 #
	##############################################################################################
	H,W,channel = rgb.shape
	gray = np.zeros((H, W, 1))
	for  i in range (H):
		for j in range (W):
			gray[i,j] = (rgb[i,j,0]*1913+rgb[i,j,1]*4815+rgb[i,j,2]*3272)/10000
	return gray

#load the data
# input (i0_path): path to the first image
# input (i1_path): path to the second image
# input (gt_path): path to the disparity image
# output (i_0): numpy array of shape (H, W, 3)
# output (i_1): numpy array of shape (H, W, 3)
# output (g_t): numpy array of shape (H, W)
def load_data(i0_path, i1_path, gt_path):

	##############################################################################################
	#										IMPLEMENT											 #
	##############################################################################################
	i_0 = Image.open(i0_path)
	i_0 = np.array((i_0))/256
	i_1 = Image.open(i1_path)
	i_1 = np.array((i_1))/256
	g_t = Image.open(gt_path)
	g_t = np.array((g_t))/256

	return i_0, i_1, g_t

# image to the size of the non-zero elements of disparity map
# input (img): numpy array of shape (H, W)
# input (d): numpy array of shape (H, W)
# output (img_crop): numpy array of shape (H', W')
def crop_image(img, d):

	##############################################################################################
	#										IMPLEMENT											 #
	##############################################################################################
	sumOfRow = d.sum(axis=1).reshape(-1,1)
	sumOfColumn = d.sum(axis=0).reshape(1,-1)
	indexLeft = 0
	indexRight = sumOfColumn.size-1
	indexTop = 0
	indexBottom = sumOfRow.size-1
	while(sumOfColumn[0,indexLeft]==0):
		indexLeft = indexLeft+1
	while(sumOfColumn[0,indexRight]==0):
		indexRight = indexRight-1
	while(sumOfRow[indexTop,0]==0):
		indexTop = indexTop+1
	while(sumOfRow[indexBottom,0]==0):
		indexBottom = indexBottom-1
	img_crop = img[indexLeft:indexRight+1,indexTop:indexBottom+1]

	return img_crop

#	##############################################################################################
#	#										IMPLEMENT											 #
#	##############################################################################################
#
#	mask = d != 0
#	nonzero_indices = np.argwhere(mask)
#
#	if len(nonzero_indices) == 0:
#		return np.array([])  # 或者根据需要返回适当的默认值
#
#	min_row, min_col = np.min(nonzero_indices, axis=0)
#	max_row, max_col = np.max(nonzero_indices, axis=0)
#
#	img_crop = img[min_row:max_row + 1, min_col:max_col + 1]
#
#	return img_crop


# shift all pixels of i1 by the value of the disparity map
# input (i_1): numpy array of shape (H, W)
# input (d): numpy array of shape (H, W)
# output (i_d): numpy array of shape (H, W)
def shift_disparity(i_1,d):

	##############################################################################################
	#										IMPLEMENT											 #
	##############################################################################################
	H_i,W_i,channel_of_i = i_1.shape
	i_d = np.zeros((H_i,W_i,3))
	for i in  range(H_i):
		for j in range (W_i):
			if j - d[i, j] >= 0:
				i_d[i,j] = i_1[i,j-int(d[i,j])]
	return i_d

# compute the negative log of the Gaussian likelihood
# input (i_0): numpy array of shape (H, W)
# input (i_1_d): numpy array of shape (H, W)
# input (mu): float
# input (sigma): float
# output (nll): numpy scalar of shape ()
def gaussian_nllh(i_0, i_1_d, mu, sigma):

	##############################################################################################
	#										IMPLEMENT											 #
	##############################################################################################
	diff = i_0-i_1_d
	nll = 0.5 * ((diff - mu )** 2)/ (sigma ** 2) + np.log(sigma * (2*np.pi) ** 0.5)
	nll = np.sum(nll)
	return nll


# compute the negative log of the Laplacian likelihood
# input (i_0): numpy array of shape (H, W)
# input (i_1_d): numpy array of shape (H, W)
# input (mu): float
# input (s): float
# output (nll): numpy scalar of shape ()
def laplacian_nllh(i_0, i_1_d, mu,s):

	##############################################################################################
	#										IMPLEMENT											 #
	##############################################################################################
	diff = i_0 - i_1_d
	nll = np.log(1 / (2*s)) - np.absolute(diff - mu) / s
	nll = np.sum(nll)
	return nll

# replace p% of the image pixels with values from a normal distribution
# input (img): numpy array of shape (H, W)
# input (p): float
# output (img_noise): numpy array of shape (H, W)
def make_noise(img, p):

	##############################################################################################
	#										IMPLEMENT											 #
	##############################################################################################
	mean = 0.38
	standard_deviation = 0.78
	noise = np.random.normal(mean,standard_deviation,img.shape)
	flattened_img = img.flatten()
	mask = np.random.choice([True,False],size = flattened_img.shape,p=[p/100,1-p/100])
	img_noise = np.where(mask,noise.flatten(),flattened_img).reshape(img.shape)
	return img_noise

# apply noise to i1_sh and return the values of the negative lok-likelihood for both likelihood models with mu, sigma, and s
# input (i0): numpy array of shape (H, W)
# input (i1_sh): numpy array of shape (H, W)
# input (noise): float
# input (mu): float
# input (sigma): float
# input (s): float
# output (gnllh) - gaussian negative log-likelihood: numpy scalar of shape ()
# output (lnllh) - laplacian negative log-likelihood: numpy scalar of shape ()
def get_nllh_for_corrupted(i_0, i_1_d, noise, mu, sigma, s):

	##############################################################################################
	#										IMPLEMENT											 #
	##############################################################################################
	i1_sh = make_noise(i_1_d,noise)
	gnllh = gaussian_nllh(i_0,i1_sh,mu,sigma)
	lnllh = laplacian_nllh(i_0,i1_sh,mu,s)

	return gnllh, lnllh

# DO NOT CHANGE
def main():
	# load images
	i0, i1, gt = load_data('./data/i0.png', './data/i1.png', './data/gt.png')

	i0, i1 = rgb2gray(i0), rgb2gray(i1)

	# shift i1
	i1_sh = shift_disparity(i1, gt)

	# crop images
	i0 = crop_image(i0, gt)
	i1_sh = crop_image(i1_sh, gt)

	mu = 0.0
	sigma = 1.3
	s = 1.3
	for noise in [0.0, 15.0, 28.0]:

		gnllh, lnllh = get_nllh_for_corrupted(i0, i1_sh, noise, mu, sigma, s)
		print([f'noise percentage {noise}%: gnllh ={gnllh},lnllh ={lnllh} '])

if __name__ == "__main__":
	main()
