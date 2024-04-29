# Visage Generator

## Terms of use

This program uses the following resources:
- [FLAME PyTorch](https://github.com/soubhiksanyal/FLAME_PyTorch)
- [FLAME PyTorch Texture Fitting](https://github.com/HavenFeng/photometric_optimization)

Before any use of Visage Generator please read
conditions of use of resources.

***
## Installation

The code uses **Python 3.10**
### Clone the project and install requirements

```
git clone https://github.com/Carlier-Maxime/Visage-Generator
cd Visage-Generator
conda env create -f environment.yml
conda activate vgen
```

if you not use anaconda or miniconda then can be installed manually package with pip (information for name packages required is in environment.yml, dependencies section)
if you are any problem during installation with conda, use etc/full_environment.yml instead of environment.yml. (full_environment.yml is complete conda environment detail used for development Visage-Generator)

### Download Models

The information necessary for the download is indicated in the [readme.md](model/readme.md) of the model folder or in the [link](https://github.com/Carlier-Maxime/Visage-Generator/blob/master/model/readme.md).

### Execute **VisageGenerator.py**

```
python VisageGenerator.py
```

***
## Config

All parameters and descriptions can be found in [default.yml](configs/default.yml) or in the [link](https://github.com/Carlier-Maxime/Visage-Generator/blob/master/configs/default.yml)

You can define the parameters either by modifying the default values in `configs/default.yml` file
or use other configuration file:
```bash
python VisageGenerator.py --cfg=<file_path>
```
for example:
```bash
python VisageGenerator.py --cfg=configs/save_all.yml
```

***
## Keys
If you use parameter ```--general-view``` you have different keys for manipulation view and data :
- **V** : Show Vertices
- **B** : Show Marker (Not default landmarks)
- **J** : Show Joints (that default landmarks)
- **E** : Edit Marker (Beta)
- **S** : Save markers
- **L** : Load markers
- **Edit marker (enable)** :
    - :arrow_left: (**Left arrow**) : direction in negative X axis
    - :arrow_right: (**Right arrow**) : direction in positif X axis
    - :arrow_down: (**Down arrow**) : direction in negative Y axis
    - :arrow_up: (**Up arrow**) : direction in positif Y axis
    - :arrow_double_down: (**Down Page**) : direction in negative Z axis
    - :arrow_double_up: (**Up Page**) : direction in positif Z axis
    - **Enter** : add marker

## Tips

- If you want remove pygame welcome message use ```export PYGAME_HIDE_SUPPORT_PROMPT=hide``` in linux