from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as tf
import torch.optim as optim

from utils import flow2rgb
from utils import rgb2gray
from utils import read_flo
from utils import read_image

np.random.seed(seed=2022)


def numpy2torch(array):
    """ Converts 3D numpy (H,W,C) ndarray to 3D PyTorch (C,H,W) tensor.

    Args:
        array: numpy array of shape (H, W, C)
    
    Returns:
        tensor: torch tensor of shape (C, H, W)
    """
    tensor = torch.from_numpy(np.transpose(array,(2,0,1)))
    return tensor


def torch2numpy(tensor):
    """ Converts 3D PyTorch (C,H,W) tensor to 3D numpy (H,W,C) ndarray.

    Args:
        tensor: torch tensor of shape (C, H, W)
    
    Returns:
        array: numpy array of shape (H, W, C)
    """
    array = np.transpose(tensor.numpy(),(1,2,0))
    return array


def load_data(im1_filename, im2_filename, flo_filename):
    """Loading the data. Returns 4D tensors. You may want to use the provided helper functions.

    Args:
        im1_filename: path to image 1
        im2_filename: path to image 2
        flo_filename: path to the ground truth flow
    
    Returns:
        tensor1: torch tensor of shape (B, C, H, W)
        tensor2: torch tensor of shape (B, C, H, W)
        flow_gt: torch tensor of shape (B, C, H, W)
    """
    tensor1 = numpy2torch(read_image(im1_filename)).unsqueeze(0)
    tensor2 = numpy2torch(read_image(im2_filename)).unsqueeze(0)
    flow_gt = numpy2torch(read_flo(flo_filename)).unsqueeze(0)
    return tensor1, tensor2, flow_gt


def evaluate_flow(flow, flow_gt):
    """Evaluate the average endpoint error w.r.t the ground truth flow_gt.
    Excludes pixels, where u or v components of flow_gt have values > 1e9.

    Args:
        flow: torch tensor of shape (B, C, H, W)
        flow_gt: torch tensor of shape (B, C, H, W)
    
    Returns:
        aepe: torch tensor scalar 
    """
    count = (flow_gt <= 1e9).sum().item()/2
    get = (flow - flow_gt)**2
    get[flow_gt > 1e9] = 0
    sum_channel = torch.sum(get,dim =1)**(0.5)
    aepe = torch.sum(sum_channel)/count

    return aepe


def visualize_warping_practice(im1, im2, flow_gt):
    """ Visualizes the result of warping the second image by ground truth.

    Args:
        im1: torch tensor of shape (B, C, H, W)
        im2: torch tensor of shape (B, C, H, W)
        flow_gt: torch tensor of shape (B, C, H, W)
    
    Returns:
    """
    warp_im2 = warp_image(im2,flow_gt)
    diff = im1 - warp_im2

    warp_im2 = torch2numpy(warp_im2[0])
    im1 = torch2numpy(im1[0])
    diff = torch2numpy(diff[0])

    fig, axs = plt.subplots(1,3)
    axs[0].imshow(im1)
    axs[0].set_title('Image1')

    axs[1].imshow(warp_im2)
    axs[1].set_title('Warped Image2')

    axs[2].imshow(diff)
    axs[2].set_title('Difference')

    plt.show()
    return


def warp_image(im, flow):
    """ Warps given image according to the given optical flow.

    Args:
        im: torch tensor of shape (B, C, H, W)
        flow: torch tensor of shape (B, C, H, W)
    
    Returns:
        x_warp: torch tensor of shape (B, C, H, W)
    """
    B, C, H, W = im.shape
    grid_y, grid_x = torch.meshgrid(torch.arange(H), torch.arange(W))
    grid = torch.stack((grid_x, grid_y), dim=0).float().unsqueeze(0).expand(B, -1, -1, -1)

    # Add flow to the grid
    flow_grid = grid + flow
    norm_grid = (flow_grid / torch.tensor([W - 1, H - 1]).view(2, 1, 1)).mul(2) - 1
    norm_grid = norm_grid.permute(0, 2, 3, 1)

    # Perform grid sampling/warping
    x_warp = tf.grid_sample(im, norm_grid, align_corners=True)
    
    return x_warp


def energy_hs(im1, im2, flow, lambda_hs):
    """ Evalutes Horn-Schunck energy function.

    Args:
        im1: torch tensor of shape (B, C, H, W)
        im2: torch tensor of shape (B, C, H, W)
        flow: torch tensor of shape (B, C, H, W)
        lambda_hs: float
    
    Returns:
        energy: torch tensor scalar
    """
    im2_warp = warp_image(im2, flow)
    diff = im2_warp-im1

    fx = torch.tensor([[[[1, -1]]]],dtype=torch.float32)
    fy = torch.tensor([[[[1], [-1]]]],dtype=torch.float32)
    delta_u = torch.nn.functional.conv2d(flow[:,0,:,:].unsqueeze(1),fx)
    delta_v = torch.nn.functional.conv2d(flow[:,1,:,:].unsqueeze(1),fy)
    energy = torch.sum(diff**2) + lambda_hs*(torch.sum(delta_u**2)+torch.sum(delta_v**2))

    return energy


def estimate_flow(im1, im2, flow_gt, lambda_hs, learning_rate, num_iter):
    """
    Estimate flow using HS with Gradient Descent.
    Displays average endpoint error.
    Visualizes flow field.

    Args:
        im1: torch tensor of shape (B, C, H, W)
        im2: torch tensor of shape (B, C, H, W)
        flow_gt: torch tensor of shape (B, C, H, W)
        lambda_hs: float
        learning_rate: float
        num_iter: int
    
    Returns:
        aepe: torch tensor scalar
    """
    flow = torch.zeros(flow_gt.shape,requires_grad=True)
    aepe = evaluate_flow(flow,flow_gt)
    print(f"AEPE before Optimization is {aepe}")

    for i in range(num_iter):
        e = energy_hs(im1,im2,flow,lambda_hs)
        #print(f"energy is {e}")
        delta = torch.autograd.grad(e,flow,create_graph=True)
        with torch.no_grad():
            flow -=learning_rate*delta[0]

    aepe = evaluate_flow(flow,flow_gt)
    print(f"AEPE after Optimization is {aepe}")
    flow.requires_grad_(False)
    getflow = torch2numpy(flow[0])
    getflow = flow2rgb(getflow)

    fig, axs = plt.subplots(1,1)
    axs.imshow(getflow)
    axs.set_title('flow')
    plt.show()
    return aepe

# Example usage in main()
# Feel free to experiment with your code in this function
# but make sure your final submission can execute this code
def main():

    # Loading data
    im1, im2, flow_gt = load_data("data/frame10.png", "data/frame11.png", "data/flow10.flo")

    # Parameters
    lambda_hs = 0.002
    num_iter = 500

    # Warping_practice
    visualize_warping_practice(im1, im2, flow_gt)

    # Gradient descent
    learning_rate = 18
    estimate_flow(im1, im2, flow_gt, lambda_hs, learning_rate, num_iter)


if __name__ == "__main__":
    main()
