"""
# File Name: setup.py
# Description:
"""
from setuptools import setup, find_packages

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(name='rescue',
      version='1.0.0',
      packages=find_packages(),
      description='Resnet model employing scRNA-seq for characterizing cell composition by using expression of whole genome',
      long_description='',


      author='mxy',
      author_email='xyMa00@126.com',
      url='https://github.com/xyMa00/Rescue.git',
      scripts=['Rescue.py'],
      install_requires=requirements,
      python_requires='>3.8.18',
      license='MIT',

      classifiers=[
          'Intended Audience :: Science/Research',
          'License :: OSI Approved :: MIT License',
          'Programming Language :: Python :: 3.8',
          'Operating System :: MacOS :: MacOS X',
          'Operating System :: Microsoft :: Windows',
          'Operating System :: POSIX :: Linux',
          'Topic :: Scientific/Engineering :: Bio-Informatics',
     ],
     )
