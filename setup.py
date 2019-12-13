from distutils.core import setup

setup(
    name = 'pycotem',
    packages = ['pycotem'],
    version = '0.1',  
    description = 'A crystal orientation toolbox for electron microscopy',
    author = 'f. mompiou',
    author_email = 'frederic.mompiou@cemes.fr',
    license='GPL-3.0',
    url = 'https://github.com/mompiou/pycotem',
    download_url = 'download link you saved',
    keywords = ['scientific', 'crystallography', 'electron microscopy'],
    classifiers = ['Intended Audience :: Science/Research',
        'Programming Language :: Python :: 2.7',],
    include_package_data=True,
    install_requires=[
        'numpy>=1.13.3',
        'pillow>=5.3.0',
        'matplotlib>=2.1.1',
    ],
)
