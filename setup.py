# -*- coding: utf-8 -*-
"""
Created on Wed May 11 20:56:21 2022

@author: Jinyu-Sun
"""

from setuptools import setup
from os import path
from io import open

this_directory = path.abspath(path.dirname(__file__))

def readme():
    with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
        return f.read()

with open(path.join(this_directory, 'requirements.txt'), encoding='utf-8') as f:
    requirements = f.read().splitlines()

setup(
    name="D-GCAN", 
    packages = ['D-GCAN'],
    package_data={'D-GCAN': ['dataset/*']},
    version="0.1.1",
    author="Jinyu-Sun",
    license="BSD-3-Clause",
    author_email="kexinhuang@hsph.harvard.edu",
    description="a Deep Learning Based Toolkit for drug-likeness prediction",
    long_description=readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/Jinyu-Sun1/D-GCAN",
    install_requires=requirements,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
    ],
)