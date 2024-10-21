

import numpy as np
from astropy.coordinates import SkyCoord
from ..datasets.general_dataset import GeneralDataset


class GeneralSatellite(object):
    '''
    这个类应当能保存单颗卫星的所有数据，组织形式为：卫星-数据集-变量
    除了需要包含数据集外，可以直接含有：卫星位置、卫星速度
    '''
    
    # 初始化
    def __init__(self, name: str, position: GeneralDataset|None =None, datasets: dict|None =None):
        '''
        初始化卫星数据
        '''
        
        self.name = name
        self.position = position
        self.datasets = datasets
        
    
    def add_dataset(self, dataset: GeneralDataset):
        '''
        添加数据集
        '''
        
        if dataset.name in self.datasets:
            raise ValueError('Dataset already exists.')
        
        self.datasets[dataset.name] = dataset
        
    
    
    
    
    

