

from typing import Type
import numpy as np
from astropy import units as u
from matplotlib import pyplot as plt

from ..utils.vdf_process import vdf_sph_to_cart


class GeneralVariable(object):
    '''
    这个类应当作为每个数据集中变量的基础类
    '''
    
    def __init__(self, name: str, data: u.Quantity, metadata: dict|None =None, description: str|None =None):
        
        self.name = name
        self.data = data
        self.metadata = metadata
        self.description = description
        self.epoch = None
    
    
    def __repr__(self):
        
        return f'{self.__class__.__name__}({self.name}: {self.description})'
    

class ScalarVariable(GeneralVariable):
    '''
    标量类
    '''
    
    def __init__(self, name: str, data: u.Quantity, metadata: dict|None =None, description: str|None =None):
        
        if len(data.shape) != 1:
            raise ValueError(f'{name} must have shape (epoch,).')
        
        super().__init__(name, data, metadata, description)
        
        self.data = self.data.si
        
        
    def plot(self, axes: plt.Axes|None =None, **kwargs):
        
        if axes is None:
            fig, ax = plt.subplots()
        else:
            ax = axes
        
        ax.plot(self.epoch, self.data, **kwargs)
        
        if axes is None:
            plt.show()
        else:
            return ax
        
    
    
class TriDimVecVariable(GeneralVariable):
    '''
    3维矢量类
    '''
    
    def __init__(self, name: str, data: u.Quantity, metadata: dict|None =None, description: str|None =None):
        
        if data.shape[1] != 3 or len(data.shape) != 2:
            raise ValueError(f'{name} must have shape (epoch, 3).')
        
        super().__init__(name, data, metadata, description)
        
        self.data = self.data.si
        
    
    @property
    def magnitude(self):
        return np.linalg.norm(self.data, axis=1)
    
    
class VDFVariable(GeneralVariable):
    '''
    VDF类
    '''
    
    def __init__(self, name: str, data: u.Quantity, metadata: dict|None =None, description: str|None =None):
        
        if len(data.shape) != 4:
            raise ValueError(f'{name} must have shape (epoch, azimuth, elevation, energy).')
        if metadata is None or set(metadata.keys()) & {'azimuth', 'elevation', 'energy'} != {'azimuth', 'elevation', 'energy'}:
            raise ValueError('metadata must contain azimuth, elevation, and energy.')
        if not all(isinstance(i, u.Quantity) for i in metadata.values()):
            raise ValueError('metadata values must be astropy quantities.')
        if not metadata['azimuth'].unit.is_equivalent(u.deg) or not metadata['elevation'].unit.is_equivalent(u.deg) or not metadata['energy'].unit.is_equivalent(u.J):
            raise ValueError('metadata units must be equivalent to deg, deg, and J.')
        if not all(len(metadata[i].shape) == 1 for i in metadata.keys()):
            raise ValueError('metadata values must be 1-dimensional.')
        if metadata['azimuth'].shape[0] != data.shape[1] or metadata['elevation'].shape[0] != data.shape[2] or metadata['energy'].shape[0] != data.shape[3]:
            raise ValueError('metadata shapes must match the data shape.')
        
        super().__init__(name, data, metadata, description)
        
        self.metadata['azimuth'] = self.metadata['azimuth'].to(u.deg)
        self.metadata['elevation'] = self.metadata['elevation'].to(u.deg)
        self.metadata['energy'] = self.metadata['energy'].si
        self.interpolated_data = None
        
    
    def interpolate(self, v_unit_new: np.ndarray|None =None):
        
        self.interpolated_data = vdf_sph_to_cart(self.metadata['azimuth'], self.metadata['elevation'], self.metadata['energy'], self.data, v_unit_new)
    
    


def generate_variable(name: str, data: u.Quantity, metadata: dict|None =None, description: str|None =None, ) -> Type[GeneralVariable]:
    