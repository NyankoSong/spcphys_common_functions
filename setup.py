"""
Setup script for spcphys-common-functions.

This file is maintained for backward compatibility.
The preferred configuration is in pyproject.toml.
"""

from os import path
from setuptools import setup, find_packages

here = path.abspath(path.dirname(__file__))

# Read long description from README
with open(path.join(here, 'Readme.md'), 'r', encoding='utf-8') as f:
    long_description = f.read()

# Read requirements from requirements.txt
with open(path.join(here, 'requirements.txt'), 'r', encoding='utf-8') as f:
    all_reqs = f.read().strip().split('\n')

install_requires = [x.strip() for x in all_reqs if x.strip() and not x.startswith('#') and 'git+' not in x]

setup(
    name='spcphys-common-functions',
    version='0.1.0',
    author='NyankoSong',
    author_email='nyankosong@gmail.com',
    maintainer='NyankoSong',
    maintainer_email='nyankosong@gmail.com',
    description='A collection of commonly used functions for space physics research',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/NyankoSong/spcphys-common-functions',
    project_urls={
        'Documentation': 'https://github.com/NyankoSong/spcphys-common-functions#readme',
        'Source': 'https://github.com/NyankoSong/spcphys-common-functions',
        'Issues': 'https://github.com/NyankoSong/spcphys-common-functions/issues',
    },
    license='MIT',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    include_package_data=True,
    install_requires=install_requires,
    extras_require={
        'dev': [
            'pytest>=7.0.0',
            'pytest-cov>=4.0.0',
            'black>=23.0.0',
            'isort>=5.12.0',
            'flake8>=6.0.0',
            'mypy>=1.0.0',
            'pre-commit>=3.0.0',
        ],
        'test': [
            'pytest>=7.0.0',
            'pytest-cov>=4.0.0',
            'pytest-asyncio>=0.21.0',
        ],
    },
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'Topic :: Scientific/Engineering :: Astronomy',
        'Topic :: Scientific/Engineering :: Physics',
        'Typing :: Typed',
    ],
    keywords=[
        'space physics',
        'solar wind',
        'plasma physics',
        'alfven waves',
        'magnetohydrodynamics',
        'heliophysics',
    ],
    python_requires='>=3.11',
    zip_safe=False,
)