
import os
import warnings
from typing import List, Tuple
from datetime import datetime, timedelta
import asyncio
semaphore = asyncio.Semaphore(5) 
import nest_asyncio
nest_asyncio.apply()
import aiohttp
from tqdm.asyncio import tqdm
import pandas as pd
from cdasws import CdasWs
cdas = CdasWs()

from .cdf_process import _get_satellite_file_infos
from . import config
from .utils import check_parameters


async def _fetch_file(dataset, varname, start_time, chunk_end_time, save_path):
    async with semaphore:
        loop = asyncio.get_event_loop()
        http_status, results = await loop.run_in_executor(None, cdas.get_data_file, dataset, varname, start_time, chunk_end_time)
        if http_status == 200:
            url = results['FileDescription'][0]['Name']
            filename = url.split('/')[-1]
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(url) as response:
                        if response.status == 200:
                            with open(os.path.join(save_path, filename), 'wb') as f:
                                while True:
                                    chunk = await response.content.read(1024*1024)
                                    if not chunk:
                                        break
                                    f.write(chunk)
                        else:
                            warnings.warn(f'Failed to fetch {filename}: HTTP {response.status}')
                print(f'{filename} saved to {save_path}')
            except Exception as e:
                warnings.warn(f'Error occured while fetching {filename}: {str(e)}')
        else:
            warnings.warn(f'Failed to fetch {dataset}: HTTP {http_status}')
        
        
async def _fetch_files(file_info_list):
    tasks = [_fetch_file(dataset, varname, start_time, chunk_end_time, save_path) for dataset, varname, start_time, chunk_end_time, save_path in file_info_list]
    for task in tqdm.as_completed(tasks, total=len(tasks), desc="Downloading"):
        await task


def fetch_cdf_from_cdaweb(cdf_info: dict|str, time_range: List[datetime]|Tuple[datetime], time_chunk: timedelta =timedelta(days=30), save_path: str|dict|None =None, info_filename: str|None =None, original_cdf_file: bool =False):
    
    DEFAULT_SAVE_DIRNAME = 'cdf_files'
    
    if isinstance(cdf_info, str):
        if not os.path.exists(cdf_info):
            raise FileNotFoundError(f'{cdf_info} must be a valid directory path or file path when it is a string.')
    
    if isinstance(cdf_info, str):
        if os.path.isdir(cdf_info):
            satellite_file_infos = _get_satellite_file_infos(cdf_info, info_filename)
            cdf_info = {}
            save_path = {}
            for satellite, info in satellite_file_infos.items():
                cdf_info[satellite] = {key: value for key, value in info['INFO'].items() if key in ['dataset', 'varname']}
                save_path[satellite] = info['PATH']
        elif os.path.isfile(cdf_info):
            satellite_file_info = pd.read_csv(cdf_info)
            cdf_info = {}
            save_path = {}
            cdf_info[DEFAULT_SAVE_DIRNAME] = {info[0]: info[1].split() for info in satellite_file_info.loc[:, ['dataset', 'varname']].values}
            save_path[DEFAULT_SAVE_DIRNAME] = os.path.join(os.path.dirname(cdf_info), DEFAULT_SAVE_DIRNAME)
            
    if isinstance(cdf_info, dict) and all(isinstance(value, list) for value in cdf_info.values()):
        cdf_info = {DEFAULT_SAVE_DIRNAME: cdf_info}
    
    if save_path is None:
        save_path = {satellite: os.path.join(os.getcwd(), satellite) for satellite in cdf_info.keys()}
    elif isinstance(save_path, str):
        save_path = {satellite: save_path for satellite in cdf_info.keys()}
        
    for path in save_path.values():
        if not os.path.exists(path):
            os.makedirs(path)
            
    
    file_info_list = []
    if not original_cdf_file:
        for satelite, info in cdf_info.items():
            for dataset, varname in info.items():
                start_time = time_range[0]
                end_time = time_range[1]
                while start_time < end_time:
                    chunk_end_time = min(start_time + time_chunk, end_time)
                    file_info_list.append((dataset, varname, start_time, chunk_end_time, save_path[satelite]))
                    start_time = chunk_end_time
                    
    try:
        asyncio.run(_fetch_files(file_info_list))
    except Exception as e:
        print(f'Error occured while fetching files: {str(e)}')
        
        
    