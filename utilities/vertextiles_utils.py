from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import os, torch
from tqdm import tqdm
import numpy as np
from src.utils.image import clip_and_scale_image, read_img_CHW, random_flip_and_rotate
from src.utils.common import find_files_by_pattern, remove_indices_from_list

class PolygonTileDataset(Dataset):

    def __init__(self, tile_dir, transform=None, n_samples=None,
            num_vertices=2, tile_format='npy', check_corruption=False,
            quick_load=False):
        
        self.tile_dir = tile_dir
        self.transform = transform
        self.n_samples = n_samples
        self.num_vertices = num_vertices
        self.extension = '.'+tile_format
        
        self.prep_dataset(check_corruption)
        self.tile_reader = np.load if tile_format == 'npy' else load_image
        if not self.n_samples is None:
            assert(self.n_samples <= len(self.center_tile_files))
    
    
    def prep_dataset(self, check_corruption=False):
        self.vertex_tile_files = []
        self.center_tile_files = []
        base_tiles = find_files_by_pattern(self.tile_dir, '*-V1'+self.extension, True)
        base_tiles = sorted(base_tiles)
        for i in range(self.num_vertices):
            self.vertex_tile_files.append([])
        skip_list = []
        
        # Checks if corresponding tiles for the base are present, and adds to data
        for base_tile in tqdm(base_tiles, desc='Processing data...'):
            all_present = True
            prefix = base_tile.split('-V1'+self.extension)[0]
            for v in range(self.num_vertices):
                path = prefix + '-V{}'.format(v+1) + self.extension
                if not os.path.isfile(path):
                    all_present = False
                    break
            path = prefix + '-Cen' + self.extension
            if not os.path.isfile(path):
                all_present = False
            
            # Only if all pairs are present, add them to dataset
            if all_present:
                for v in range(self.num_vertices):
                    self.vertex_tile_files[v].append(prefix + '-V{}'.format(v+1)
                                                     + self.extension)
                self.center_tile_files.append(prefix + '-Cen' + self.extension)
            else:
                skip_list.append(base_tile)
        
        if skip_list:
            print('The following data sets were skipped:')
            for tile in skip_list:
                print(tile)
                
        if check_corruption: self.check_if_dataset_corrupted()
        
    def check_if_dataset_corrupted(self, verbose=True):
        corrupt_tiles = set()
        corrupt_tile_names = [] # For research purposes
        for i in tqdm(range(self.__len__()), desc='Checking dataset for corruption...'):
            if self.is_tile_corrupt(self.center_tile_files[i]):
                corrupt_tiles.add(i)
                corrupt_tile_names.append(self.center_tile_files[i])
            for v in range(self.num_vertices):
                if self.is_tile_corrupt(self.vertex_tile_files[v][i]):
                    corrupt_tiles.add(i)
                    corrupt_tile_names.append(self.vertex_tile_files[v][i])
        
        if corrupt_tiles:
            self.remove_tiles(corrupt_tiles)
            if verbose:
                print('List of corrupted tiles:')
                for tile in corrupt_tile_names:
                    print(tile)
            print('\nTotal no. of corrupted tiles:', str(len(corrupt_tile_names)))
            
        return
    
    def remove_tiles(self, index_list):
        self.center_tile_files = remove_indices_from_list(self.center_tile_files,
                                                          list(index_list))
        for v in range(self.num_vertices):
            self.vertex_tile_files[v] = remove_indices_from_list(self.vertex_tile_files[v],
                                                                 list(index_list))
        if self.n_samples:
            if self.n_samples > len(self.center_tile_files):
                self.n_samples = len(self.center_tile_files)
        return
    
    def is_tile_corrupt(self, tile_path):
        return np.isnan(np.load(tile_path)).any() # True if corrupt
    
    def __len__(self):
        return self.n_samples if self.n_samples else len(self.center_tile_files)

    def __getitem__(self, idx):
        vertex_tiles = []
        for v in range(self.num_vertices):
            tile = self.tile_reader(self.vertex_tile_files[v][idx])
            vertex_tiles.append(tile)
        centroid_tile = self.tile_reader(self.center_tile_files[idx])

        sample = {'vertices': vertex_tiles, 'centroid': centroid_tile}
        return self.transform(sample) if self.transform else sample


### TRANSFORMS ###

class GetBands(object):
    """
    Gets the first X bands of the tile triplet.
    """
    def __init__(self, bands):
        assert bands >= 0, 'Must get at least 1 band'
        self.bands = bands

    def __call__(self, sample):
        for i in range(len(sample['vertices'])):
            tile = sample['vertices'][i]
            sample['vertices'][i] = tile[:self.bands,:,:]
        centroid_tile = sample['centroid']
        sample['centroid'] = centroid_tile[:self.bands,:,:]
        return sample

class RandomFlipAndRotate(object):
    """
    Does data augmentation during training by randomly flipping (horizontal
    and vertical) and randomly rotating (0, 90, 180, 270 degrees). Keep in mind
    that pytorch samples are CxWxH.
    """
    def __call__(self, sample):
        
        for i in range(len(sample['vertices'])):
            sample['vertices'][i] = random_flip_and_rotate(sample['vertices'][i])
        
        sample['centroid'] = random_flip_and_rotate(sample['centroid'])
        return sample

class ClipAndScale(object):
    """
    Clips and scales bands to between [0, 1] for NAIP, RGB, and Landsat
    satellite images. Clipping applies for Landsat only.
    """
    def __init__(self, img_type):
        assert img_type in ['naip', 'rgb', 'landsat']
        self.img_type = img_type

    def __call__(self, sample):
        for i in range(len(sample['vertices'])):
            sample['vertices'][i] = clip_and_scale_image(sample['vertices'][i],
                                                         self.img_type)
            
        sample['centroid'] = clip_and_scale_image(sample['centroid'], self.img_type)
        return sample

class ToFloatTensor(object):
    """
    Converts numpy arrays to float Variables in Pytorch.
    """
    def __call__(self, sample):
        for i in range(len(sample['vertices'])):
            tile = sample['vertices'][i]
            sample['vertices'][i] = torch.from_numpy(tile).float()
        sample['centroid'] = torch.from_numpy(sample['centroid']).float()
        return sample

### TRANSFORMS ###


def polygon_tiles_dataloader(img_type, tile_dir, bands=4, augment=True,
                             batch_size=4, shuffle=True, num_workers=4,
                             n_samples=None, num_vertices=2, tile_format='npy',
                             check_corruption=False, quick_load=False):
    """
    Returns a DataLoader with either NAIP (RGB/IR), RGB, or Landsat tiles.
    Turn shuffle to False for producing embeddings that correspond to original
    tiles.
    """
    assert img_type in ['landsat', 'rgb', 'naip']
    ## Note: All the transforms are in-place
    transform_list = []
    if img_type in ['landsat', 'naip']: transform_list.append(GetBands(bands))
    transform_list.append(ClipAndScale(img_type))
    if augment: transform_list.append(RandomFlipAndRotate())
    transform_list.append(ToFloatTensor())
    transform = transforms.Compose(transform_list)
    dataset = PolygonTileDataset(tile_dir, transform=transform, n_samples=n_samples,
                                 num_vertices=num_vertices, tile_format=tile_format,
                                 check_corruption=check_corruption, quick_load=quick_load)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                            num_workers=num_workers)
    return dataloader