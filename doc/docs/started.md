## Executables

Windows executables can be downloaded [here](http://mompiou.free.fr/pycotem-1.4.0-win.zip)

Unpack the folder and launch the ```.exe``` files for the different tools ```stereoproj```, ```diffraction```, ```kikuchi```,```misorientation```,```interface``` and ```crystal```.

## From python

!!! note
	Consider that it is often recommended to install packages in a virtual environnement to avoid breaking dependancies for other scripts. Please refer to the document to use virtualenv.
	
### Prerequisites

pycotem relies on ```python2.7``` with ```numpy>=1.13.3```, ```pillow>=5.3.0```, ```matplotlib>=2.1.1``` packages which can be installed with ```pip```. GUI relies on ```pyqt4```.

### Installing and running

#### On windows 

-  Install python 2.7

```
wget https://www.python.org/ftp/python/2.7.9/python-2.7.9.amd64.msi
msiexec /i python-2.7.9.amd64.msi /qb

```
	
- Download PyQt4 (choose amd64 and python 2.7 cp27) wheel and install (do not place the whl file in a folder with non ascii characters).

```
pip install PyQt4-4.11.4-cp27-cp27m-win_amd64.whl
```

- Install pycotem

```
pip install pycotem
```

Run the packages with
```
python -m pycotem.xxxx
``` 
where ```xxxx``` is one of the tool ```stereoproj```, ```diffraction```, ```kikuchi```,```misorientation```,```interface``` and ```crystal```.

#### On Ubuntu 18.04

Python 2.7 distribution should be installed by default. 

- Install ```pip```

```
sudo apt-get install python-pip
sudo pip install --upgrade pip

```

- Install pycotem from ```pip```:

```
pip install pycotem
```

- Install ```pyqt4```:

```
sudo apt-get install python-qt4

```

- Run the script as above.

#### On MacOS X (not tested)

- Install Homebrew:

```
/usr/bin/ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"

```

- Add the following line at the bottom of your ~/.profile file:

```
export PATH="/usr/local/bin:/usr/local/sbin:$PATH"
```

- Install python from Homebrew:

```
brew install python@2
```

 - and modify the PATH to the new installation:
 
 ```
 export PATH="/usr/local/opt/python@2/libexec/bin:$PATH"
 ```
 
- Install pyqt4:

```
brew install cartr/qt4/pyqt
```


- Install ```pip```:

```
sudo easy_install pip
sudo pip install --upgrade pip

```

- Install pycotem with ```pip``` as above.

- Run the script as above

### Updating

- Use ```pip install --upgrade pycotem```

## Examples

Image files and setting parameters for testing ```diffraction```, ```interface``` and ```kikuchi``` can be found [here](https://github.com/mompiou/pycotem/tree/master/test)

