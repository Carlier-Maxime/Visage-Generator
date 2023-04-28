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

if you not use anaconda or miniconda then can be install manually package with pip (information for name packages required is in environment.yml, dependencies section)
if you are any problem during installation with conda, use etc/full_environment.yml instead of environment.yml. (full_environment.yml is complete conda environment detail used for developement Visage-Generator)

### Download Models

The information necessary for the download is indecate in the readme.md of the model folder or in the [link](https://github.com/Carlier-Maxime/Visage-Generator/blob/master/model/readme.md).

### Execute **VisageGenerator.py**

```
python VisageGenerator.py
```

***
## Config

Here are the different settings you can change:

### General
- ```--nb-faces``` : number faces generate
- ```--not-texturing', 'texturing``` : disable texture
- ```--device``` : choice your device for generate face. ("cpu" or "cuda")
- ```--view``` : enable view
- ```--batch-size``` : number of visage generate in the same time
- ```--texture-batch-size``` : number of texture generate in same time

### Generator parameter
- ```--min-shape-param``` : minimum value for shape param
- ```--max-shape-param``` : maximum value for shape param
- ```--min-expression-param``` : minimum value for expression param
- ```--max-expression-param``` : maximum value for expression param
- ```--global-pose-param1``` : value of first global pose param
- ```--global-pose-param2``` : value of second global pose param
- ```--global-pose-param3``` : value of third global pose param
- ```--min-jaw-param1``` : minimum value for jaw param 1
- ```--max-jaw-param1``` : maximum value for jaw param 1
- ```--min-jaw-param2-3``` : minimum value for jaw param 2-3
- ```--max-jaw-param2-3``` : maximum value for jaw param 2-3
- ```--min-texture-param``` : minimum value for texture param
- ```--max-texture-param``` : maximum value for texture param
- ```--min-neck-param``` : minimum value for neck param
- ```--max-neck-param``` : maximum value for neck param
- ```--fixed-shape``` : fixed the same shape for all visage generated
- ```--fixed-expression``` : fixed the same expression for all visage generated
- ```--fixed-jaw``` : fixed the same jaw for all visage generated
- ```--fixed-texture``` : fixed the same texture for all visage generated
- ```--fixed-neck``` : fixed the same neck for all visage generated

### Flame parameter
- ```--not-use-face-contour', 'use_face_contour``` : not use face contour for generate visage
- ```--not-use-3D-translation', 'use_3D_translation``` : not use 3D translation for generate visage
- ```--shape-params``` : a number of shape parameter used
- ```--expression-params``` : a number of expression parameter used

### Saving
- ```--outdir``` : path directory for output
- ```--lmk2D-format', 'lmk2D_format``` : format used for save lmk2d. (npy and pts is supported)
- ```--save-obj``` : enable save into file obj
- ```--save-png``` : enable save into file png
- ```--save-lmks3D-npy', 'save_lmks3D_npy``` : enable save landmarks 3D into file npy
- ```--save-lmks3D-png', 'save_lmks3D_png``` : enable save landmarks 3D with visage into file png
- ```--save-lmks2D', 'save_lmks2D``` : enable save landmarks 2D into file npy
- ```--save-markers``` : enable save markers into png file
- ```--img-resolution``` : resolution of image
- ```--show-window``` : show window during save png (enable if images is the screenshot or full black)
- ```--not-pts-in-alpha', 'pts_in_alpha``` : not save landmarks/markers png version to channel alpha

### Path
- ```--flame-model-path``` : path for acess flame model
- ```--static-landmark-embedding-path``` : path for static landmark embedding file
- ```--dynamic-landmark-embedding-path``` : path for dynamic landmark embedding file

You can define the parameters either by modifying the default values ​​in config.py file 
or by launching the program, for example:
```
python ./VisageGenerator.py --nb-faces=1 --view --save-png --save-markers
```

***
## Keys
If you use parameter ```--view``` you have different keys for manipulation view and data :
- **V** : Show Vertices
- **B** : Show Marker/Balises (Not default marker)
- **J** : Show Joints
- **E** : Edit Marker/Balises (Beta)
- **S** : Save balises
- **L** : Load balises
- **Edit marker (enable)** :
    - :arrow_left: (**Left arrow**) : direction in negatif X axis
    - :arrow_right: (**Right arrow**) : direction in positif X axis
    - :arrow_down: (**Down arrow**) : direction in negatif Y axis
    - :arrow_up: (**Up arrow**) : direction in positif Y axis
    - :arrow_double_down: (**Down Page**) : direction in negatif Z axis
    - :arrow_double_up: (**Up Page**) : direction in positif Z axis
    - **Enter** : add marker