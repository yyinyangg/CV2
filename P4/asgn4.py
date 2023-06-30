from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import torch
import utils
import skimage
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from matplotlib import pyplot as plt
from skimage import color

from utils import VOC_LABEL2COLOR
from utils import VOC_STATISTICS

from torchvision import models
from torchvision import transforms
from PIL import Image

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
        self.num = num_examples
        self.image_path = root + "/JPEGImages/"
        self.segmentation_path = root + "/SegmentationClass/"
        self.imagenames_path = root + "/ImageSets/Segmentation/train.txt"
        self.segmentationnames_path = root + "/ImageSets/Segmentation/val.txt"

        lines = []
        if train:
            with open(self.imagenames_path, 'r') as file:
                for l in file.readlines():
                    lines.append(l.strip())
        else:
            with open(self.segmentationnames_path) as file:
                for l in file.readlines():
                    lines.append(l.strip())
        if self.num is not None:
            self.lines = lines[:self.num]

        self.img = sorted([os.path.join(self.image_path,fileName) for fileName in self.lines])
        self.mask = sorted([os.path.join(self.segmentation_path, fileName) for fileName in self.lines])

    def __getitem__(self, index):
        """
        Return an item from the datset.

        Args:
            index: index of the item (Int)

        Returns:
            item: dictionary of the form {'im': the_image, 'gt': the_label}
            with the_image being a torch tensor (3, H, W) (float) and 
            the_label being a torch tensor (1, H, W) (long) and 
        """
        item = dict()
        img_path = self.img[index]+'.jpg'
        gt_path =  self.mask[index]+'.png'

        img = utils.numpy2torch(np.array(Image.open(img_path).convert('RGB'))).to(torch.float32)
        gt = utils.numpy2torch(np.array(Image.open(gt_path).convert('RGB'))).to(torch.int64)

        gt = self.convertSeg(gt)

        item.update({'im':img, 'gt':gt})

        assert (isinstance(item, dict))
        assert ('im' in item.keys())
        assert ('gt' in item.keys())

        return item

    def __len__(self):
        """
        Return the length of the datset.

        Args:

        Returns:
            length: length of the dataset (int)
        """
        return len(self.img)

    def convertSeg(self, gt):
        C,H,W = gt.shape
        rtn = torch.full((1,H,W),21) #21--ambiguous

        # Convert color labels to corresponding label IDs
        for label, color in enumerate(VOC_LABEL2COLOR):
            mask = (gt == torch.Tensor(color).view(3, 1, 1)).all(dim=0)
            mask = torch.unsqueeze(mask, dim=0)
            rtn[mask] = label

        return rtn


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
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers
    )
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

    # Convert color image to YCbCr color space
    ycbcr = skimage.color.rgb2ycbcr(np_image)

    # Extract texture component (luminance)
    texture = ycbcr[:, :, 0]

    # Set color channels (e.g., hue) based on labels
    color_channels = np.zeros_like(ycbcr[:, :, 1:])  # Initialize color channels


    for i, color in enumerate(VOC_LABEL2COLOR):
        # Find pixels with the corresponding label
        label_pixels = np_label == i
        color = skimage.color.rgb2ycbcr(tuple(np.array(color)/255))
        #print(label_pixels.shape)
        #print(color_channels.shape)
        # Set color channels for label pixels
        color_channels[label_pixels[:,:,0],0] = color[1]
        color_channels[label_pixels[:,:,0],1] = color[2]

    # Assemble texture component and color channels in YCbCr color space
    ycbcr_modified = np.dstack((texture, color_channels))

    # Convert color-coded representation back to RGB
    colored = skimage.color.ycbcr2rgb(ycbcr_modified)

    assert (np.equal(colored.shape, np_image.shape).all())
    assert (np_image.dtype == colored.dtype)
    return colored


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
        #print(image.shape)

        # Convert tensors to NumPy arrays
        image_np = utils.torch2numpy(image)/255.0
        print(image_np.max())

        #print(image_np.shape)
        label_np = utils.torch2numpy(label)/255.0

        # Apply voc_label2color to get color-coded labels
        colored_label = voc_label2color(image_np, label_np)

        # Plot the image and colored label
        row = i // grid_width
        col = i % grid_width
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
    normalized = []

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
    prediction = []
    acts = []

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
    return None

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
    cwd = os.getcwd()
    get = cwd + "\Data\VOCtrainval_06-Nov-2007\VOCdevkit\VOC2007"
    root = os.environ["VOC2007_HOME"] = get

    # create datasets for training and validation
    train_dataset = VOC2007Dataset(root, train=True, num_examples=128)
    valid_dataset = VOC2007Dataset(root, train=False, num_examples=128)


    # create data loaders for training and validation
    train_loader = create_loader(train_dataset, batch_size=1, shuffle=True, num_workers=1)
    valid_loader = create_loader(valid_dataset, batch_size=1, shuffle=False, num_workers=1)

    # show some images for the training and validation set
    show_dataset_examples(train_loader, grid_height=2, grid_width=3, title='training examples')
    show_dataset_examples(valid_loader, grid_height=2, grid_width=3, title='validation examples')
'''
    # Load FCN network
    model = models.segmentation.fcn_resnet101(pretrained=True, num_classes=21)

    # Apply fcn. Switch to training loader if you want more variety.
    show_inference_examples(valid_loader, model, grid_height=2, grid_width=3, title='inference examples')

    # attack1: convert cat to dog
    cat_example = find_unique_example(valid_loader, unique_foreground_label=8)
    show_unique_example(cat_example, model=model)
    show_attack(cat_example, model, src_label=8, target_label=12, learning_rate=1.0, iterations=10)

    # feel free to try other examples..
'''
if __name__ == '__main__':
    main()
