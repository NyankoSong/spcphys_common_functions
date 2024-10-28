

from typing import Type, List
import warnings
import numpy as np
from astropy import units as u
from matplotlib import pyplot as plt

from ..utils.vdf_process import vdf_sph_to_cart
from ..utils.plot_tools import plot
from ..utils.vth_E_T import *

from ..utils.utils import check_parameters


class GeneralVariable(object):
    
    @check_parameters
    def __init__(self, name: str, data: u.Quantity, description: str|None =None, **metadata):
        
        if 'unit' in metadata and not isinstance(metadata['unit'], u.UnitBase):
            raise ValueError('metadata unit must be an astropy unit.')
        
        self.name = name
        self.data = data.si
        self.metadata = metadata
        self.description = description
        self.epoch = None
    
    
    def __repr__(self):
        
        return f'{self.__class__.__name__}({self.name}: {self.description})'
    
    
    def get_data(self):
        if 'unit' in self.metadata:
            if self.data.unit.is_equivalent(self.metadata['unit']):
                return self.data.to(self.metadata['unit'])
            else:
                warnings.warn(f'Unit of {self.name} ({self.data.unit}) is not equivalent to setting unit {self.metadata["unit"]}.')
        return self.data
        
        
    def plot(self, axes: plt.Axes|None =None, **kwargs):
        
        plot(self.epoch, self.data, axes, **kwargs)
        axes.set_ylabel(f'{self.name} ({self.data.unit})')
    

class ScalarVariable(GeneralVariable):
    
    def __init__(self, name: str, data: u.Quantity, description: str|None =None, **metadata):
        
        if len(data.shape) != 1:
            raise ValueError(f'{name} must have shape (epoch,).')
        
        super().__init__(name, data, description, **metadata)
        
    
    
class TriDimVecVariable(GeneralVariable):
    
    def __init__(self, name: str, data: u.Quantity, description: str|None =None, **metadata):
        
        if data.shape[1] != 3 or len(data.shape) != 2:
            raise ValueError(f'{name} must have shape (epoch, 3).')
        
        super().__init__(name, data, description, **metadata)
        
    
    @property
    def magnitude(self):
        return np.linalg.norm(self.data, axis=1)
    
    
class DualDimScalarVariable(GeneralVariable):
    
    def __init__(self, name: str, data: u.Quantity, description: str|None =None, **metadata):
        
        if data.shape[1] != 2 or len(data.shape) != 2:
            raise ValueError(f'{name} must have shape (epoch, 2).')
        
        super().__init__(name, data, description, **metadata)
        

    

class VDFVariable(GeneralVariable):
    
    def __init__(self, name: str, data: u.Quantity, description: str|None =None, **metadata):
        
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
        
        super().__init__(name, data, description, **metadata)
        
        self.metadata['azimuth'] = self.metadata['azimuth'].to(u.deg)
        self.metadata['elevation'] = self.metadata['elevation'].to(u.deg)
        self.metadata['energy'] = self.metadata['energy'].si
        self.interpolate_unit_vec = None
        self.interpolated_data = None
        
    
    def interpolate(self, v_unit_new: np.ndarray|None =None):
        
        self.interpolated_data = vdf_sph_to_cart(self.metadata['azimuth'], self.metadata['elevation'], self.metadata['energy'], self.data, v_unit_new)
        self.interpolate_unit_vec = v_unit_new
        
    
    def plot(self, axes: plt.Axes|None =None, **kwargs):
        pass


class MagneticFieldVariable(TriDimVecVariable):
    
    def __init__(self, name: str, data: u.Quantity, description: str|None =None, **metadata):
        
        if metadata is None or set(metadata.keys()) & {'coordinate_system'} != {'coordinate_system'}:
            raise ValueError('metadata must contain coordinate_system.')
        if not data.unit.is_equivalent(u.T):
            raise ValueError('data unit must be equivalent to T.')
        
        super().__init__(name, data, description, **metadata)
        
        self.metadata['coordinate_system'] = metadata['coordinate_system']
        
        
class VelocityVariable(TriDimVecVariable):
    
    def __init__(self, name: str, data: u.Quantity, description: str|None =None, **metadata):
        
        if metadata is None or set(metadata.keys()) & {'coordinate_system'} != {'coordinate_system'}:
            raise ValueError('metadata must contain coordinate_system.')
        if not data.unit.is_equivalent(u.m / u.s):
            raise ValueError('data unit must be equivalent to m/s.')
        
        super().__init__(name, data, description, **metadata)
        
        self.metadata['coordinate_system'] = metadata['coordinate_system']
        

class NumDensityVariable(ScalarVariable):
    
    def __init__(self, name: str, data: u.Quantity, description: str|None =None, **metadata):
        
        if not data.unit.is_equivalent(u.m ** -3):
            raise ValueError('data unit must be equivalent to m^-3.')
        
        super().__init__(name, data, description, **metadata)
        
        
class TemperatureVariable(ScalarVariable):
    
    def __init__(self, name: str, data: u.Quantity, description: str|None =None, **metadata):
        
        if not data.unit.is_equivalent(u.K):
            if not data.unit.is_equivalent(u.J) and not data.unit.is_equivalent(u.m/u.s):
                raise ValueError('data unit must be equivalent to K, J, or m/s.')
            elif data.unit.is_equivalent(u.m/u.s):
                if 'mass' not in metadata or not metadata['mass'].unit.is_equivalent(u.kg):
                    raise ValueError('metadata must contain mass with unit equivalent to kg.')
                if 'velocity_type' not in metadata or metadata['velocity_type'] not in {'most_prob', 'rms', 'average', 'rms_1d'}:
                    raise ValueError('metadata must contain velocity_type with value in most_prob, rms, average, and rms_1d.')
        
        super().__init__(name, data, description, **metadata)
                
        if 'mass' in metadata:
            self.metadata['mass'] = self.metadata['mass'].si
        if 'velocity_type' in metadata:
            if metadata['velocity_type'] == 'most_prob':
                self.metadata['velocity_type'] = 2
            elif metadata['velocity_type'] == 'rms':
                self.metadata['velocity_type'] = 3
            elif metadata['velocity_type'] == 'average':
                self.metadata['velocity_type'] = 8/np.pi
            elif metadata['velocity_type'] == 'rms_1d':
                self.metadata['velocity_type'] = 1
        
        if self.data.unit.is_equivalent(u.J):
            self.data = E_to_T(self.data)
        if self.data.unit.is_equivalent(u.m/u.s):
            self.data = vth_to_T(self.data, metadata['mass'], metadata['velocity_type'])
            
    
    def to_energy(self):
        self.data = T_to_E(self.data)
    
    
    def to_vth(self):
        self.data = T_to_vth(self.data, self.metadata['mass'], self.metadata['velocity_type'])
    
    
    def to_default(self):
        if self.data.unit.is_equivalent(u.J):
            self.data = E_to_T(self.data)
        elif self.data.unit.is_equivalent(u.m/u.s):
            self.data = vth_to_T(self.data, self.metadata['mass'], self.metadata['velocity_type'])
        else:
            self.data = self.data.si
    

def _get_variable_class(data_type: str) -> List[str]:
    
    class_names = []
    for subclass in GeneralVariable.__subclasses__():
        class_names.append(subclass.__name__)
        subsubclasses = subclass.__subclasses__()
        if subsubclasses:
            for subsubclass in subsubclasses:
                class_names.append(subsubclass.__name__)
        
    return class_names
                

@check_parameters
def generate_variable(data_type: str|Type[GeneralVariable], name: str, data: u.Quantity, description: str|None =None, **metadata) -> Type[GeneralVariable]:
    
    if isinstance(data_type, str):
        class_names = _get_variable_class(data_type)
            
        if data_type not in class_names and f'{data_type}Variable' not in class_names:
            raise ValueError(f'data_type must be one of {class_names}.')
        else:
            DataClass = globals()[data_type] if 'Variable' in data_type else globals()[f'{data_type}Variable']
    else:
        DataClass = data_type
            
    return DataClass(name, data, description, **metadata)