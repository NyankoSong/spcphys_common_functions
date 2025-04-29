
import os
import warnings
from typing import List, Tuple
from datetime import datetime, timedelta
import asyncio
import nest_asyncio
nest_asyncio.apply()
import aiohttp
from tqdm.asyncio import tqdm
import pandas as pd
from cdasws import CdasWs
cdas = CdasWs()

from .cdf_process import _get_satellite_file_infos



async def _fetch_file(semaphore, dataset, varname, start_time, chunk_end_time, save_path, retries, delay):
    async with semaphore:
        loop = asyncio.get_event_loop()
        for attempt in range(1, retries + 1):
            try:
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
                                    # print(f'{filename} saved to {save_path}')
                                    return  # 成功，退出函数
                                else:
                                    warnings.warn(f'Failed to fetch {filename}: HTTP {response.status}')
                    except Exception as e:
                        warnings.warn(f'Error occurred while fetching {filename}: {str(e)}')
                else:
                    warnings.warn(f'Failed to fetch {dataset}: HTTP {http_status}')
            except Exception as e:
                warnings.warn(f'Attempt {attempt} failed with error: {str(e)}\nIf the error shows "Connection Aborted" or "ConnectTimeoutError" repeatedly, your IP might have been banned by CDAWeb. Try again later.')
            if attempt < retries:
                await asyncio.sleep(delay)
                warnings.warn(f'Retrying to fetch {dataset} ({attempt}/{retries})...')
        warnings.warn(f'Failed to fetch {dataset} after {retries} attempts')
        
        
async def _fetch_files(semaphore, file_info_list, retries, delay):
    tasks = [_fetch_file(semaphore, dataset, varname, start_time, chunk_end_time, save_path, retries, delay) for dataset, varname, start_time, chunk_end_time, save_path in file_info_list]
    for task in tqdm.as_completed(tasks, total=len(tasks), desc="Downloading"):
        await task


def fetch_cdf_from_cdaweb(cdf_info: dict|str, time_range: List[datetime]|Tuple[datetime], time_chunk: timedelta =timedelta(days=30), save_path: str|dict|None =None, info_filename: str|None =None, retries: int =3, delay: int =2, max_concurrent: int =3) -> List[Tuple[str, List[str], datetime, datetime, str]]:
    """
    Fetch CDF files from CDAWeb based on provided dataset information and time range.

    :param cdf_info: A dictionary or string containing dataset and variable name information.
                     - If a dictionary, it maps satellite names to their datasets and variable names. It should be in the format {satellite1: {dataset1: [varname1, varname2, ...], dataset2: [...], ...}, satellite2: {...}, ...} or {dataset1: [varname1, varname2, ...], dataset2: [...], ...} if there is no need to differentiate between satellites.
                     - If a string, it can be a directory path containing satellite info files or a CSV file with dataset and variable names. The CSV file should have columns 'dataset' and 'varname', detailed information can be found in cdf_process.process_satellite_data.
    :param time_range: A list or tuple of two datetime objects specifying the start and end times for data retrieval.
    :param time_chunk: Optional; a timedelta object specifying the time chunk size for data retrieval. Defaults to 30 days.
    :param save_path: Optional; a string or dictionary specifying the path(s) to save the downloaded CDF files.
                      - If None, defaults to the current working directory.
                      - If a string, the same path is used for all satellites.
                      - If a dictionary, it maps satellite names to their respective save paths.
    :param info_filename: Optional; a string specifying the filename of the info file when `cdf_info` is a directory.
    :param retries: Optional; an integer specifying the number of retries for each file download. Defaults to 3.
    :param delay: Optional; an integer specifying the delay in seconds between retries. Defaults to 2.
    :param max_concurrent: Optional; an integer specifying the maximum number of concurrent downloads. Defaults to 3.
    
    :return file_info_list: A list of tuples containing dataset, variable name, start time, end time, and save path for each file.
    """
    
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
            cdf_info[DEFAULT_SAVE_DIRNAME] = {info[0]: info[1].split() for info in satellite_file_info.loc[:, ['dataset', 'varname']].values}
            if save_path is None:
                save_path = os.path.dirname(cdf_info)
            
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
    for satelite, info in cdf_info.items():
        for dataset, varname in info.items():
            start_time = time_range[0]
            end_time = time_range[1]
            while start_time < end_time:
                chunk_end_time = min(start_time + time_chunk, end_time)
                file_info_list.append((dataset, varname, start_time, chunk_end_time, save_path[satelite]))
                start_time = chunk_end_time
             
    semaphore = asyncio.Semaphore(max_concurrent)
    asyncio.run(_fetch_files(semaphore, file_info_list, retries, delay))
        
    return file_info_list
        
        
    