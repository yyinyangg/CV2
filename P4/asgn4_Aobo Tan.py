from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import skimage.color
import torch

import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from matplotlib import pyplot as plt
from skimage import color
import utils

from utils import VOC_LABEL2COLOR
from utils import VOC_STATISTICS

from torchvision import models

#自己加的库
import random
from PIL import Image
from torchvision import transforms



class VOC2007Dataset(Dataset):
    """
    Class to create a dataset for VOC 2007
    Refer to https://pytorch.org/tutorials/beginner/basics/data_tutorial.html for an instruction on PyTorch datasets.
    """

    def __init__(self, root, train, num_examples):
        super().__init__()
        """
        Initialize the dataset by setting the required attributes.

        Args:
            root: root folder of the dataset (string)
            train: indicates if we want to load training (train=True) or validation data (train=False)
            num_examples: size of the dataset (int)

        Returns (just set the required attributes):
            input_filenames: list of paths to individual images
            target_filenames: list of paths to individual segmentations (corresponding to input_filenames)
            rgb2label: lookup table that maps RGB values to class labels using the constants in VOC_LABEL2COLOR.
        """
        # help function for line reading with random sequence
        def random_lines_from_file(file_path, num_lines):
            with open(file_path, 'r') as file:
                lines = file.readlines()
            random_lines = random.sample(lines, num_lines)
            random_lines = [line.strip() for line in random_lines]
            return random_lines

        label2rgb = {idx: VOC_LABEL2COLOR[idx] if idx < len(VOC_LABEL2COLOR) else (224, 224, 192) for idx in range(len(VOC_LABEL2COLOR) + 235)}

        rgb2label = {color: i for i, color in enumerate(VOC_LABEL2COLOR)}
        self.rgb2label = rgb2label

        root_of_image = os.path.join(root,'JPEGImages')
        root_of_segmentation = os.path.join(root,'SegmentationClass')
        input_filenames = []
        target_filenames = []

        # loading training data
        if (train == True):
            root_of_splitFile = os.path.join(root,'ImageSets','Segmentation','train.txt')
            train_photo_idx = random_lines_from_file(root_of_splitFile, num_examples)
            for idx in train_photo_idx:
                input_filenames.append(root_of_image + '/' + idx + '.jpg')
                target_filenames.append(root_of_segmentation + '/' + idx + '.png')
            self.input_filenames = input_filenames
            self.target_filenames = target_filenames

        # loading validation data
        else:
            root_of_splitFile = os.path.join(root, 'ImageSets', 'Segmentation', 'val.txt')
            val_photo_idx = random_lines_from_file(root_of_splitFile, num_examples)
            for idx in val_photo_idx:
                input_filenames.append(root_of_image + '/' + idx + '.jpg')
                target_filenames.append(root_of_segmentation+ '/' + idx + '.png')
            self.input_filenames = input_filenames
            self.target_filenames = target_filenames



    def __getitem__(self, index):
        """
        Return an item from the dataset.

        Args:
            index: index of the item (Int)

        Returns:
            item: dictionary of the form {'im': the_image, 'gt': the_label}
            with the_image being a torch tensor (3, H, W) (float) and 
            the_label being a torch tensor (1, H, W) (long) and 
        """
        path_of_im = self.input_filenames[index]
        path_of_gt = self.target_filenames[index]
        item = dict()
        im_jpg = np.array(Image.open(path_of_im))
        im = torch.from_numpy(im_jpg).permute(2, 0, 1).float()
        gt_png = np.array(Image.open(path_of_gt))
        gt = torch.from_numpy(gt_png).unsqueeze(0).long()
        item['im'] = im
        item['gt'] = gt
        assert (isinstance(item, dict))
        assert ('im' in item.keys())
        assert ('gt' in item.keys())

        return item

    def __len__(self):
        """
        Return the length of the dataset.

        Args:

        Returns:
            length: length of the dataset (int)
        """
        return len(self.input_filenames)


def create_loader(dataset, batch_size, shuffle, num_workers=1):
    """
    Return loader object.

    Args:
        dataset: PyTorch Dataset
        batch_size: int
        shuffle: bool
        num_workers: int

    Returns:
        loader: PyTorch DataLoader
    """

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    assert (isinstance(loader, DataLoader))
    return loader


def voc_label2color(np_image, np_label):
    """
    Super-impose labels on a given image using the colors defined in VOC_LABEL2COLOR.

    Args:
        np_image: numpy array (H,W,3) (float)
        np_label: numpy array  (H,W) (int)

    Returns:
        colored: numpy array (H,W,3) (float)
    """


    assert (isinstance(np_image, np.ndarray))
    assert (isinstance(np_label, np.ndarray))
    H,W = np_label.shape
    np_image = np_image/255
    np_image_YCbCr = skimage.color.rgb2ycbcr(np_image)
    label2rgb = {idx: VOC_LABEL2COLOR[idx] if idx < len(VOC_LABEL2COLOR) else (224, 224, 192) for idx in range(len(VOC_LABEL2COLOR) + 235)}
    label_rgb = np.zeros((H,W,3))
    for i in range(H):
        for j in range(W):
            label_rgb[i, j] = label2rgb[np_label[i][j]]
    label_rgb = label_rgb/255
    label_rgb_ycbcr = skimage.color.rgb2ycbcr(label_rgb)
    colored = np.zeros_like(np_image)
    colored[:, :, 0] = np_image_YCbCr[:, :, 0]  # this one is right
    colored[:, :, 1:] = label_rgb_ycbcr[:, :, 1:]
    colored_rgb = skimage.color.ycbcr2rgb(colored)
    assert (np.equal(colored.shape, np_image.shape).all())
    assert (np_image.dtype == colored.dtype)
    return colored_rgb


def show_dataset_examples(loader, grid_height, grid_width, title):
    """
    Visualize samples from the dataset.

    Args:
        loader: PyTorch DataLoader
        grid_height: int
        grid_width: int
        title: string
    """

    fig, axs = plt.subplots(grid_height, grid_width, figsize=(10, 10))
    fig.suptitle(title)

    for i, data in enumerate(loader):
        image = data['im'][0]
        label = data['gt'][0]
        # print(image.shape)

        # Convert tensors to NumPy arrays
        image_np = utils.torch2numpy(image)

        #print(image_np.max())

        # print(image_np.shape)
        label_np = utils.torch2numpy(label)



        # Apply voc_label2color to get color-coded labels
        colored_label = voc_label2color(image_np, label_np[:,:,0])

        # Plot the image and colored label
        row = i // grid_width
        col = i % grid_width
        colored_label = np.clip(colored_label,0,1)
        axs[row, col].imshow(colored_label)
        axs[row, col].axis('off')

        if i + 1 == grid_height * grid_width:
            break

    plt.tight_layout()
    plt.show()
    pass

def normalize_input(input_tensor):
    """
    Normalize a tensor using statistics in VOC_STATISTICS.

    Args:
        input_tensor: torch tensor (B,3,H,W) (float32)

    Returns:
        normalized: torch tensor (B,3,H,W) (float32)
    """
    mean = torch.tensor(VOC_STATISTICS["mean"]).view(1, 3, 1, 1)
    std = torch.tensor(VOC_STATISTICS["std"]).view(1, 3, 1, 1)

    normalized = (input_tensor - mean) / std

    assert (type(input_tensor) == type(normalized))
    assert (input_tensor.size() == normalized.size())
    return normalized

def run_forward_pass(normalized, model):
    """
    Run forward pass.

    Args:
        normalized: torch tensor (B,3,H,W) (float32)
        model: PyTorch model
        
    Returns:
        prediction: class prediction of the model (B,1,H,W) (int64)
        acts: activations of the model (B,21,H,W) (float 32)
    """
    model.eval()
    with torch.no_grad():
        acts = model(normalized)
        _, prediction = torch.max(acts,dim=1)

    assert (isinstance(prediction, torch.Tensor))
    assert (isinstance(acts, torch.Tensor))
    return prediction, acts

def show_inference_examples(loader, model, grid_height, grid_width, title):
    """
    Perform inference and visualize results.

    Args:
        loader: PyTorch DataLoader
        model: PyTorch model
        grid_height: int
        grid_width: int
        title: string
    """
    pass

def average_precision(prediction, gt):
    """
    Compute percentage of correctly labeled pixels.

    Args:
        prediction: torch tensor (B,1,H,W) (int)
        gt: torch tensor (B,1,H,W) (int)

    Returns:
        avg_prec: torch scalar (float32)
    """
    assert (prediction.shape == gt.shape)
    totaly_pixels = prediction.numel()
    prediction = prediction.view(-1)
    gt = gt.view(-1)
    correct_pixels = torch.sun(prediction==gt)
    avg_prec = (correct_pixels.float() / totaly_pixels )*100
    return avg_prec

### FUNCTIONS FOR PROBLEM 2 ###

def find_unique_example(loader, unique_foreground_label):
    """Returns the first sample containing (only) the given label

    Args:
        loader: dataloader (iterable)
        unique_foreground_label: the label to search

    Returns:
        sample: a dictionary with keys 'im' and 'gt' specifying
                the image sample 
    """
    example = []

    assert (isinstance(example, dict))
    return example


def show_unique_example(example_dict, model):
    """Visualise the results produced for a given sample (see Fig. 3).

    Args:
        example_dict: a dict with keys 'gt' and 'im' returned by an instance of VOC2007Dataset
        model: network (nn.Module)
    """
    pass


def show_attack(example_dict, model, src_label, target_label, learning_rate, iterations):
    """Modify the input image such that the model prediction for label src_label
    changes to target_label.

    Args:
        example_dict: a dict with keys 'gt' and 'im' returned by an instance of VOC2007Dataset
        model: network (nn.Module)
        src_label: the label to change
        target_label: the label to change to
        learning_rate: the learning rate of optimisation steps
        iterations: number of optimisation steps

    This function does not return anything, but instead visualises the results (see Fig. 4).
    """
    pass


# Example usage in main()
# Feel free to experiment with your code in this function
# but make sure your final submission can execute this code
def main():
    # Please set an environment variables 'VOC2007_HOME' pointing to your '../VOCdevkit/VOC2007' folder
    os.environ["VOC2007_HOME"] = "C:/Users/Aobo Tan/PycharmProjects/pythonProject_Computer_Vision/Assignment/Assignment-4/VOCtrainval_06-Nov-2007/VOCtrainval_06-Nov-2007/VOCdevkit/VOC2007"
    root = os.environ["VOC2007_HOME"]

    # create datasets for training and validation
    train_dataset = VOC2007Dataset(root, train=True, num_examples=128)
    valid_dataset = VOC2007Dataset(root, train=False, num_examples=128)

    # create data loaders for training and validation
    train_loader = create_loader(train_dataset, batch_size=1, shuffle=True, num_workers=1)
    valid_loader = create_loader(valid_dataset, batch_size=1, shuffle=False, num_workers=1)

    # show some images for the training and validation set
    show_dataset_examples(train_loader, grid_height=2, grid_width=3, title='training examples')
    show_dataset_examples(valid_loader, grid_height=2, grid_width=3, title='validation examples')

    # Load FCN network
    model = models.segmentation.fcn_resnet101(pretrained=True, num_classes=21)

    # Apply fcn. Switch to training loader if you want more variety.
    show_inference_examples(valid_loader, model, grid_height=2, grid_width=3, title='inference examples')

    # attack1: convert cat to dog
    #cat_example = find_unique_example(valid_loader, unique_foreground_label=8)
    #show_unique_example(cat_example, model=model)
    #show_attack(cat_example, model, src_label=8, target_label=12, learning_rate=1.0, iterations=10)

    # feel free to try other examples..

if __name__ == '__main__':
    main()
