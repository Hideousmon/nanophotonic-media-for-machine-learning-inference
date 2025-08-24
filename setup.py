#!/usr/bin/env python
from setuptools import setup, find_packages

with open("nao/__init__.py") as fin:
    for line in fin:
        if line.startswith("__version__ ="):
            version = eval(line[14:])
            break

setup(name='nao',
      version=version,
      description='Nano-array optimization for integrated photonics design.',
      author='Zhenyu ZHAO',
      author_email='mailtozyzhao@163.com',
      install_requires=['numpy==1.24.3', 'scipy==1.10.1', 'splayout==0.5.15', 'jaxlib==0.4.30', 'jax==0.4.30'],
      url="https://github.com/Hideousmon/nanophotonic-media-for-machine-learning-inference",
      packages=find_packages()
      )