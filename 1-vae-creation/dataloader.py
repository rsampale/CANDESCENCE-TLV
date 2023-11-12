import torch
from torch.utils.data import DataLoader, Dataset, random_split, Subset
from torchvision import datasets, transforms
import os
from PIL import Image
import glob
import random
import matplotlib.pyplot as plt # plotting library
import torchvision.transforms.functional as TF
import numpy as np
from utils import get_one_hot_encoder, get_plate_id, get_growth_medium, get_day, get_size_onehot_from_tensor, get_intensity_onehot_from_tensor
import logging

### Logging Info ###

logging.basicConfig(filename='dataloading.log', level=logging.DEBUG, format='%(asctime)s %(levelname)s %(name)s %(message)s')
logger=logging.getLogger(__name__)

####################

class CustomTransform:
    def __init__(self):
        pass

    def __call__(self, img):
        if random.random() > 0.5:
            img = TF.hflip(img)
        if random.random() > 0.5:
            img = TF.vflip(img)

        angle = random.uniform(-25, 25)
        img = TF.rotate(img, angle)

        # Add a RANDOM ERASE?

        return img

transform = transforms.Compose([
    transforms.Resize([135, 135]),
    transforms.Grayscale(),
    # CustomTransform(), # applies the custom transforms defined above
    transforms.ToTensor()
])

class CandidaDataset(Dataset):
    def __init__(self, dir_path, image_type, one_hot_encoder, arguments, transform=None):
        super(CandidaDataset, self).__init__()

        self.transform = transform
        self.arguments = arguments

        # Load all image file paths
        self.image_paths = glob.glob(dir_path + '/*.bmp')

        self.dataset_categorical_info = collect_categories(dir_path)  # get this specific dataset's categorical info as a list of lists      # change
        self.one_hot_encoder = one_hot_encoder
        
        # Use either 'wash' or 'non-wash' images in the datasets
        if image_type == 'wash':
            self.image_paths = [path for path in self.image_paths if 'wash' in path]  # MAY NOT BE ENOUGH IMAGES FOR THE TRAIN/TEST/VAL SIZES SPECIFIED IF ONLY WASH ARE USED 
        elif image_type == 'non-wash':
            self.image_paths = [path for path in self.image_paths if 'wash' not in path]

    def __getitem__(self, index):
        image_path = self.image_paths[index]

        # Load image
        image = Image.open(image_path)

        # Apply transformations
        if self.transform is not None:
            image = self.transform(image)

        # Get one-hot encoding of the plate ID
        indiv_categorical_info = self.dataset_categorical_info[index]
        logger.debug(f"Contents of indiv_categorical_info: {indiv_categorical_info}")
        # Represent the list of categories as a string, so it can get a unique and proper one-hot
        indiv_categorical_info_string = [f"{indiv_categorical_info[0]}_{indiv_categorical_info[1]}_{indiv_categorical_info[2]}"]
        logger.debug(f"Contents of indiv_categorical_info_string: {indiv_categorical_info_string}")
        indiv_categorical_info_string_2d = np.array(indiv_categorical_info_string).reshape(-1, 1) # make it 2d
        logger.debug(f"Contents of indiv_categorical_info_string_2d: {indiv_categorical_info_string_2d}")
        OH_encoded_categories = self.one_hot_encoder.transform(indiv_categorical_info_string_2d)
        torch_tensor_OH_encoded_categories = torch.from_numpy(OH_encoded_categories)

        # no_var = torch.tensor([0])
        # no_var = no_var.unsqueeze(-1)

        # Concatenate/add the colony size encoding:
        image = image.to(self.arguments['DEVICE'])
        size_OH_encoding = get_size_onehot_from_tensor(image, self.arguments['size_bins'])
        logger.debug(f"CONTENTS of size_OH_encoding, in dataloader: {size_OH_encoding}")
        logger.info(f"SHAPE of size_OH_encoding, in dataloader: {size_OH_encoding.shape}")
        logger.info(f"Shape of torch_tensor_OH_encoded_categories, prior to joining with size/other encoded info: {torch_tensor_OH_encoded_categories.shape}")
        torch_tensor_OH_encoded_categories = torch.cat((torch_tensor_OH_encoded_categories, size_OH_encoding), dim=1)

        # Concatenate/add the colony intensity encoding:
        # image = image.to(self.arguments['DEVICE'])
        # intensity_OH_encoding = get_intensity_onehot_from_tensor(image, self.arguments['intensity_bins'])
        # logger.debug(f"CONTENTS of intensity_OH_encoding, in dataloader: {intensity_OH_encoding}")
        # logger.info(f"SHAPE of intensity_OH_encoding, in dataloader: {intensity_OH_encoding.shape}")
        # # logger.info(f"Shape of torch_tensor_OH_encoded_categories, prior to joining with intensity/other encoded info: {torch_tensor_OH_encoded_categories.shape}")
        # torch_tensor_OH_encoded_categories = torch.cat((size_OH_encoding, intensity_OH_encoding), dim=1)

        # print it, for test
        logger.debug(f"Contents of torch_tensor_OH_encoded_categories, returned by dataset __getitem__: {torch_tensor_OH_encoded_categories}")
        logger.info(f"Shape of torch_tensor_OH_encoded_categories, returned by dataset __getitem__: {torch_tensor_OH_encoded_categories.shape}")

        return image, torch_tensor_OH_encoded_categories # return both the colony tensor AND the one-hot encoding of the image from which it is from
        # Alternate returns, for testing only one variable or another:
        # return image, size_OH_encoding
        # return image, no_var

    def __len__(self):
        return len(self.image_paths)
    
    def get_image_filename(self, indicies):
        if indicies is None:
            return self.image_paths
        else:
            return [self.image_paths[i] for i in indicies]
        
    def get_dataset_OH_encoder(self):
        return self.one_hot_encoder


def create_dataloader(arguments):

    # Find plate and medium identifiers, and create the one-hot encoder
    # To make a one-hot encoder that is fitted on all possible combinations of plate and medium across the ENTIRE dataset
    categories = collect_categories(arguments['CUT_IMAGES'])
    one_hot_encoder = get_one_hot_encoder(categories)
    
    # Initialize dataset
    dataset = CandidaDataset(arguments['CUT_IMAGES'], arguments['image_type'], one_hot_encoder, arguments, transform)

    # Set the seed
    torch.manual_seed(arguments['dataset_seed'])

    if 0 <= arguments['train_num'] <= 1 and 0 <= arguments['val_num'] <= 1 and 0 <= arguments['test_num'] <= 1:
        assert 0 <= arguments['train_num'] <= 1 and 0 <= arguments['val_num'] <= 1 and 0 <= arguments['test_num'] <= 1
        assert arguments['train_num'] + arguments['val_num'] + arguments['test_num'] == 1
        total_size = len(dataset)
        test_size = int(total_size * arguments['test_num'])
        val_size = int(total_size * arguments['val_num'])
        train_size = total_size - test_size - val_size

        train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
    else:
        indicies = list(range(len(dataset)))
        random.shuffle(indicies)
        indicies = indicies[:(arguments['train_num']+arguments['test_num']+arguments['val_num'])]
        subset_dataset = Subset(dataset,indicies)

        train_num = arguments['train_num']
        val_num = arguments['val_num']
        test_num = arguments['test_num']
        train_dataset, val_dataset, test_dataset = random_split(subset_dataset, [train_num, val_num, test_num])

        # Save the datasets with their corresponding indices
        torch.save((train_dataset, indicies[:train_num]), os.path.join(arguments['VAES'], f'{arguments["dataset_seed"]}_train'))
        torch.save((val_dataset, indicies[train_num:train_num + val_num]), os.path.join(arguments['VAES'], f'{arguments["dataset_seed"]}_val'))
        torch.save((test_dataset, indicies[train_num + val_num:train_num + val_num + test_num]), os.path.join(arguments['VAES'], f'{arguments["dataset_seed"]}_test'))

    # Save the datasets
    # torch.save(train_dataset, os.path.join(arguments['VAES'], f'{arguments["dataset_seed"]}_train'))
    # torch.save(val_dataset, os.path.join(arguments['VAES'], f'{arguments["dataset_seed"]}_val'))
    # torch.save(test_dataset, os.path.join(arguments['VAES'], f'{arguments["dataset_seed"]}_test'))

    # Create dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=arguments['batch_size'], shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=arguments['batch_size'], shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=arguments['batch_size'], shuffle=False)

    return train_dataloader, val_dataloader, test_dataloader

def collect_categories(dir_path):
    image_paths = glob.glob(dir_path + '/*.bmp')
    categories = [[get_plate_id(path), get_growth_medium(path), get_day(path)] for path in image_paths] # change: DONE

    # unique_categories = set(plate_ids_and_mediums)

    # Print unique categories
    # print(plate_ids_and_mediums)
    # print("Unique categories:")
    # for category in unique_categories:
    #     print(category)

    return categories