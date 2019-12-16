# pycotem

pycotem is a python package for working with crystal orientations in transmission electron microscopy. It provides 6 GUI tools to:

- determine orientation from diffraction patterns: ```diffraction```

- determine orientation from Kikuchi patterns: ```kikuchi```

- work with stereographic projections: ```stereoproj```

- determine interface normal and direction from images: ```interface```

- determine orientation relationship and misorientation between two crystals: ```misorientation```

- display crystal projection on a plane and draw dichromatic patterns: ```crystal```

## Getting Started

### Prerequisites

pycotem relies on ```python2.7``` with ```numpy>=1.13.3```, ```pillow>=5.3.0```, ```matplotlib>=2.1.1``` packages which can be installed with ```pip```. GUI relies on ```pyqt4```.

### Installing and running

pycotem can be installed with pip: ```pip install pycotem```
Run ```python -m pycotem.xxxx``` for the different tools: ```stereoproj```, ```diffraction```, ```kikuchi```,```misorientation```,```interface``` and ```crystal```.

### Examples

Image files and setting parameters are provided in the ```test``` directory in the github repository for testing ```diffraction```, ```interface``` and ```kikuchi```.


## Contributing

Contributions, bug issues, requests and comments can be adressed directly or through pull-request on github.

## Authors

[F. Mompiou](https://github.com/mompiou), with idea from  [RX Xie](https://github.com/XIEruixun) (Tsinghua Univ), J. Du (Tsinghua Univ.) and the contribution of G. Perret (Univ. Toulouse).


## License

This project is licensed under the GPL-3.0 License.





