from setuptools import find_packages, setup

from os import path
this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md')) as f:
    long_description = f.read()

setup(
    name = 'pycotem',
    packages=find_packages(exclude=("doc","test")),
    version = '2.5.0',  
    description = 'A python package for working with crystal orientations in transmission electron microscopy',
    long_description=long_description,
    long_description_content_type="text/markdown",
    author = 'f. mompiou',
    author_email = 'frederic.mompiou@cemes.fr',
    license='GPL-3.0',
    url = 'https://mompiou.github.io/pycotem/',
    download_url = 'https://github.com/mompiou/pycotem',
    keywords = ['scientific', 'crystallography', 'electron microscopy'],
    classifiers = ['Intended Audience :: Science/Research',
        'Programming Language :: Python :: 3',],
    include_package_data=True,
)
