

from typing import List, Type
from datetime import datetime
import numpy as np

from ..variables.general_variable import GeneralVariable
from ..utils.preprocess import _npdt64_to_dt


class GeneralDataset(object):
    '''
    这个类应当作为每个数据集的基础类
    '''
    
    def __init__(self, name: str, epoch: List[datetime]|np.ndarray, variables: dict|None =None):
        
        self.name = name
        self.epoch = self._check_epoch(epoch)
        self.variables = variables
        
    
    def __repr__(self):
            
        return f'{self.__class__.__name__}({self.name})'
    
    
    def __getitem__(self, key: str) -> Type[GeneralVariable]:
        
        return self.get_variable(key)
        
        
    def _check_epoch(self, epoch: List[datetime]|np.ndarray) -> np.ndarray:
        if all(isinstance(i, datetime) for i in epoch):
            if isinstance(epoch, list):
                return np.array(epoch)
            elif isinstance(epoch, np.ndarray):
                return epoch
        elif all(isinstance(i, np.datetime64) for i in epoch):
            if isinstance(epoch, np.ndarray):
                return _npdt64_to_dt(epoch)
            
            
    def add_variable(self, variable: Type[GeneralVariable]):
        
        if variable.name in self.variables:
            raise ValueError(f'Variable {variable.name} already exists.')
        
        if variable.data.shape[0] != self.epoch.shape[0]:
            raise ValueError(f'Variable {variable.name} has a different epoch from the dataset.')
        
        self.variables[variable.name] = variable
        variable.epoch = self.epoch
        
    
    def get_variable(self, variable_name: str) -> Type[GeneralVariable]:
        
        if variable_name not in self.variables:
            raise ValueError(f'Variable {variable_name} does not exist.')
        
        return self.variables[variable_name]
        