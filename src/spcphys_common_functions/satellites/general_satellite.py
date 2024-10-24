

from typing import Type
import numpy as np
from astropy.coordinates import SkyCoord


from ..datasets.general_dataset import GeneralDataset


class GeneralSatellite(object):
    '''
    这个类应当能保存单颗卫星的所有数据，组织形式为：卫星-数据集-变量
    '''
    
    # 初始化
    def __init__(self, name: str, datasets: dict|None =None):
        
        self.name = name
        self.datasets = datasets
        self.dataset_marks = {'Position': None, 'Proton': None, 'VDF': None, 'IMF': None}
        
        
    def __repr__(self):
            
        return f'{self.__class__.__name__}({self.name})'
    
    
    def __getitem__(self, key: str) -> Type[GeneralDataset]:
        
        if key in self.dataset_marks and self.dataset_marks[key] is not None:
            return self.get_dataset(self.dataset_marks[key])
        else:
            return self.get_dataset(key)
        
    
    def add_dataset(self, dataset: Type[GeneralDataset], mark_as: str|None =None, overwrite: bool =False):
        
        if dataset.name in self.datasets and not overwrite:
            raise ValueError(f'Dataset {dataset.name} already exists.')
        
        self.datasets[dataset.name] = dataset
        if mark_as is not None:
            self.mark_dataset(dataset.name, mark_as)
        

    def remove_dataset(self, dataset_name: str):
        
        if dataset_name not in self.datasets:
            raise ValueError(f'Dataset {dataset_name} does not exist.')
        
        self.datasets.pop(dataset_name)
        
        
    def get_dataset(self, dataset_name: str) -> Type[GeneralDataset]:
        
        if dataset_name not in self.datasets:
            raise ValueError(f'Dataset {dataset_name} does not exist.')
        
        return self.datasets[dataset_name]
    
    
    def mark_dataset(self, dataset_name: str, mark_as: str):
        
        if dataset_name not in self.datasets:
            raise ValueError(f'Dataset {dataset_name} does not exist.')
        
        self.dataset_marks[mark_as] = dataset_name
        
    
    
    
    
    

