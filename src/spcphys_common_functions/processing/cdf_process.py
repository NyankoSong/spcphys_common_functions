from typing import List
import warnings
from multiprocessing import Pool
import pickle
import os
import cdflib
import numpy as np
import pandas as pd

from .preprocess import _get_boundary, npdt64_to_dt
from ..utils.utils import check_parameters, _determine_processes

def _recursion_traversal_dir(path:str) -> List[str]:
    
    '''
    Recursively traversal the directory to find all satellite data directories.
    
    :param path: str, the root directory of the satellite data.
    :return: list, all satellite data directories.
    '''
    
    satellite_paths = []
    for file in os.listdir(path):
        file_path = os.path.join(path, file)
        file_path = file_path.replace('\\', '/')
        if os.path.isdir(file_path):
            satellite_path = _recursion_traversal_dir(file_path)
            satellite_paths += satellite_path
        else:
            if file_path.endswith('.cdf') or file_path.endswith('.csv'):
                satellite_paths.append('/'.join(file_path.split('/')[:-1]) + '/')
                break
    satellite_paths = list(set(satellite_paths))
    
    return satellite_paths


def _get_satellite_file_infos(dir_path:str, info_filename: str|None=None):
    
    '''
    Get all satellite file infos in satellite directory.
    
    :param dir_path: str, the root directory of the satellite data.
    :param info_filename: str, the name of the info file.
    :return: dict, the satellite file infos.
    '''
    
    satellite_paths = _recursion_traversal_dir(dir_path)
    satellite_file_infos = dict()
    for satellite_path in satellite_paths:
        # satellite_name = satellite_path.split('/')[-2]
        satellite_info = {'PATH': satellite_path, 'INFO': {}, 'CDFs': []}
        for file in os.listdir(satellite_path):
            file_path = os.path.join(satellite_path, file)
            file_path = file_path.replace('\\', '/')
            if file_path.endswith('.cdf'):
                satellite_info['CDFs'].append(file)
            if (info_filename is not None and file_path.endswith(info_filename)) or (info_filename is None and file_path.endswith('.csv')):
                info = pd.read_csv(file_path)
                satellite_info['INFO']['startswith'] = info.iloc[:, 0].tolist()
                satellite_info['INFO']['dataset'] = info.iloc[:, 1].tolist()
                satellite_info['INFO']['timeres'] = info.iloc[:, 2].tolist()
                satellite_info['INFO']['varname'] = [s.split(' ') for s in info.iloc[:, 3].tolist()]
                satellite_info['INFO']['condition'] = [_get_boundary(sub_s) for sub_s in [s.split(' ') for s in info.iloc[:, 4].tolist()]]
        
        satellite_file_infos[satellite_path.split('/')[-2]] = satellite_info
        
    return satellite_file_infos


def _convert_epoches(epoch_slice):
    return npdt64_to_dt(cdflib.cdfepoch.to_datetime(epoch_slice))


def _chunks(data, n):
    k, m = divmod(len(data), n)
    return [data[i * (k + 1):(i + 1) * (k + 1)] if i < m else data[i * k + m:(i + 1) * k + m] for i in range(n)]


def _parallel_convert_epoches(epoch, num_processes=None):
    
    num_processes = _determine_processes(num_processes)
        
    if len(epoch) < num_processes:
        num_processes = len(epoch)
    epoch_chunks = _chunks(epoch, num_processes)
        
    with Pool(processes=num_processes) as p:
        results = p.map(_convert_epoches, epoch_chunks)
    
    return np.concatenate(results)

@check_parameters
def process_satellite_data(dir_path:str, info_filename: str|None=None, output_dir: str|None=None, num_processes: float|int=1, epoch_varname_default: List[str]|str='epoch'):
    
    '''
    Combine all satellite data into a single file for each satellite. 
    
    :param dir_path: str, the root directory of the satellite data.
    :param info_filename: str, the name of the info file.
    :param output_dir: str, the output directory of the processed data.
    :param num_processes: float|int, the number of processes used to convert epoches, 1 for single process, 0.9 for 90% of the cpu cores, etc.
    
    This function assumes that the satellite data is stored in the following structure:
    - dir_path
        - satellite1_name
            - cdf_files
            - info_file
        - satellite2_name
            - cdf_files
            - info_file
        ...
    
    if the info_filename is None, the function will search for the csv file in the satellite directory. 
    Make sure that there is only one csv file in each satellite directory if the info_filename is not specified.
    The info file should be a csv file with the following structure:
    | startswith, | dataset, | timeres, | varname,                | condition,|
    |-------------|----------|----------|-------------------------|-----------|
    | startswith1,| dataset1,| timeres1,| varname11 varname12 ...,| condition1|
    | startswith2,| dataset2,| timeres2,| varname21 varname22 ...,| condition2|
    ...
    
    startswith is the prefix of the cdf files, this is used to identify the cdf files that belong to the same dataset.
    dataset is the name of the dataset, this will be the key of the output data dict.
    timeres is the time resolution of the dataset, this will directly be the value of the 'TIMERES' key in the output data dict.
    varname is the name of the variables in the dataset, and should be separated by space.
    condition should be a string with two elements separated by space, which are the lower and upper boundary of the variable. 
    if the variable has no condition, use 'none' instead, this will set the boundary to [-1E30, 1E30].
    '''
    
    if not os.path.exists(dir_path):
        raise FileNotFoundError(f'{dir_path} not found!')
    if output_dir is not None and not os.path.exists(output_dir):
        raise FileNotFoundError(f'{output_dir} not found!')
    if num_processes < 0 or num_processes > os.cpu_count():
        raise ValueError(f'num_processes should be in the range of (0, 1) or [1, {os.cpu_count()})!')
    
    if isinstance(epoch_varname_default, str):
        epoch_varname_default = [epoch_varname_default]
    
    satellite_file_infos = _get_satellite_file_infos(dir_path, info_filename)
    
    for satellite_name, satellite_info in satellite_file_infos.items():
        if output_dir is not None:
            dir_path = output_dir
        data_file_name = satellite_name + '_data.pkl'
        output_file = os.path.join(dir_path, data_file_name)
        if os.path.exists(output_file):
            print(f'{data_file_name} already exists in {dir_path}, skip this directory!')
            continue
        
        data_dict = dict()
        for dataset, startswith, varnames, timeres, condition in zip(satellite_info['INFO']['dataset'],
                                                                     satellite_info['INFO']['startswith'],
                                                                     satellite_info['INFO']['varname'], 
                                                                     satellite_info['INFO']['timeres'], 
                                                                     satellite_info['INFO']['condition']):
            print(f'Processing {satellite_name} {dataset}...')
            data_dict[dataset] = dict()
            data_dict[dataset]['TIMERES'] = timeres
            date_flag = True
            for varname in varnames:
                print(f'Processing {varname}...')
                data_dict[dataset][varname] = dict()
                var_tmp = None
                date_tmp = None
                err_flag = False
                dataset_cdfs = [cdf for cdf in satellite_info['CDFs'] if cdf.startswith(startswith)]
                for cdf_i, cdf in enumerate(dataset_cdfs):
                    print(f'({cdf_i+1}/{len(dataset_cdfs)}) Reading {cdf}...')
                    cdf_path = os.path.join(satellite_info['PATH'], cdf)
                    cdf_file = cdflib.CDF(cdf_path)
                    try:
                        cdf_var = cdf_file.varget(varname)
                    except Exception as e:
                        warnings.warn(f'Error when reading {varname} from {cdf}: {str(e)}\nDataset {dataset} will be deleted from the data dict.')
                        err_flag = True
                        break
                    else:
                        cdf_var = np.array(cdf_var)
                        ################################################
                        # 补丁，强制类型转换，可能造成不可预料的后果
                        try:
                            cdf_var = cdf_var.astype(float)
                        except Exception as e:
                            warnings.warn(f'{varname} can not be forcefully converted to float: {str(e)}')
                            if cdf_var is not None:
                                var_tmp = cdf_var
                        else:
                            cdf_var[(cdf_var < condition[0]) | (cdf_var > condition[1])] = np.nan
                            var_tmp = cdf_var if var_tmp is None else np.concatenate((var_tmp, cdf_var), axis=0)
                            
                            if date_flag:
                                print(f'({cdf_i+1}/{len(dataset_cdfs)}) Encoding epochs, this might take a long time...')
                                epoch_varname = [zvarname for zvarname in cdf_file.cdf_info().zVariables for epoch_default in epoch_varname_default if epoch_default.lower() in zvarname.lower()][0]
                                if num_processes == 1:
                                    cdf_date = _convert_epoches(cdf_file.varget(epoch_varname)) # This is TOO SLOW
                                else:
                                    cdf_date = _parallel_convert_epoches(cdf_file.varget(epoch_varname), num_processes)
                                
                                date_tmp = cdf_date if date_tmp is None else np.concatenate((date_tmp, cdf_date), axis=0)
                if err_flag:
                    data_dict[dataset].pop(varname)
                    break
                else:
                    if date_flag:
                        sorted_indices = np.argsort(date_tmp)
                        data_dict[dataset]['DATE'] = date_tmp[sorted_indices]
                        date_flag = False
                        
                    if len(var_tmp) == len(sorted_indices):
                        data_dict[dataset][varname] = var_tmp[sorted_indices]
                    else:
                        warnings.warn(f'Length of {varname} are not the same as other variables, so it is not sorted.')
                        data_dict[dataset][varname] = var_tmp
        
        print(f'Saving data to {data_file_name}, this might use lots of RAM...')
        ########################################
        # np.save(data_file_name, data_dict) # OverflowError: serializing a bytes object larger than 4 GiB requires pickle protocol 4 or higher
        with open(output_file, 'wb') as f:
            pickle.dump(data_dict, f, protocol=4)
        
        print(f'{data_file_name} saved to {dir_path}!')


@check_parameters
def generate_cdf_info_csv(dir_path:str, info_filename:str='info.csv', epoch_varname_default: List[str]|str='epoch', ignore_varname: List[str]|str|None =None) -> dict:
    """
    Generate a CSV file containing information about CDF files in the specified directory.

    :param dir_path: str, the root directory of the satellite data.
    :param info_filename: str, the name of the info file to be generated. Default is 'info.csv'.
    :param epoch_varname_default: List[str] | str, the default name(s) of the epoch variable. Default is 'epoch'. This is identified by if the epoch_varname_default is in the variable name instead of the exact match.
    :param ignore_varname: List[str] | str | None, the variable name(s) to be ignored. Default is None. This is identified by if the ignore_varname is not in the variable name instead of the exact match.
    
    :return: dict, a dictionary containing information about the satellite files.
    """
    
    if not os.path.exists(dir_path):
        raise FileNotFoundError(f'{dir_path} not found!')
    
    if isinstance(epoch_varname_default, str):
        epoch_varname_default = [epoch_varname_default.lower()]
    elif isinstance(epoch_varname_default, list):
        epoch_varname_default = [epoch_default.lower() for epoch_default in epoch_varname_default]
    if isinstance(ignore_varname, str):
        ignore_varname = [ignore_varname.lower()]
    elif isinstance(ignore_varname, list):
        ignore_varname = [varname.lower() for varname in ignore_varname]
    
    satellite_file_infos = _get_satellite_file_infos(dir_path, info_filename)
    satellite_info_dict = {}
    for satellite_name, satellite_info in satellite_file_infos.items():
        output_file = os.path.join(satellite_info['PATH'], info_filename)
        if os.path.exists(output_file):
            warnings.warn(f'{info_filename} already exists in {satellite_info["PATH"]}, skip this directory!')
            continue
        
        satellite_info_dict[satellite_name] = {}
        for cdf_filename in satellite_info['CDFs']:
            cdf_file = cdflib.CDF(os.path.join(satellite_info['PATH'], cdf_filename))
            dataset_name = ''
            for cdf_filename_parts in cdf_filename.split('_'):
                if not cdf_filename_parts.isdigit():
                    dataset_name += cdf_filename_parts + '_'
                else:
                    dataset_name = dataset_name[:-1]
                    break
            cdf_varnames = cdf_file.cdf_info().zVariables
            epoch_varname = [zvarname for zvarname in cdf_varnames for epoch_default in epoch_varname_default if epoch_default in zvarname.lower()]
            if len(epoch_varname) == 0:
                warnings.warn(f'No epoch variable found in {cdf_filename}, skip this file!')
                continue
            cdf_varnames.remove(epoch_varname[0])
            if ignore_varname is not None:
                cdf_varnames = [zvarname for zvarname in cdf_varnames for ignored_name in ignore_varname if ignored_name not in zvarname.lower()]
            if dataset_name not in satellite_info_dict[satellite_name]:
                satellite_info_dict[satellite_name][dataset_name] = cdf_varnames
            else:
                if set(cdf_varnames) & set(satellite_info_dict[satellite_name][dataset_name]) != set(satellite_info_dict[satellite_name][dataset_name]):
                    warnings.warn(f'Variable names in {cdf_filename} are not the same as other files in the same dataset!\n{cdf_filename} has {cdf_varnames}, while others have {satellite_info_dict[satellite_name][dataset_name]}')
                    satellite_info_dict[satellite_name][dataset_name] = list(set(satellite_info_dict[satellite_name][dataset_name]) | set(cdf_varnames))
                    
        result = pd.DataFrame(columns=['startswith', 'dataset', 'timeres', 'varname', 'condition'])
        for dataset_name, varnames in satellite_info_dict[satellite_name].items():
            result = pd.concat([result, pd.DataFrame([{'startswith': dataset_name, 'dataset': dataset_name.upper(), 'timeres': '-1', 'varname': ' '.join(varnames), 'condition': 'none'}])], ignore_index=True)
            
        result.to_csv(output_file, index=False)
        
    return satellite_info_dict

                    
    