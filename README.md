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

Here are the different settings you can change:

### General
- ```--nb-faces``` : number faces generate
- ```--not-texturing``` : disable texture
- ```--device``` : choice your device for generate face. ("cpu" or "cuda")
- ```--view``` : enable view
- ```--batch-size``` : number of visage generate in the same time
- ```--camera``` : default camera for renderer [fov, tx, ty, tz, rx, ry, rz] (rotation in degree)
- ```--camera-type``` : camera type used for renderer (change utilisation of camera parameter) [default, vector]

### Generator parameter
- ```--input-folder``` : input folder for load parameter
- ```--zeros-params``` : zeros for all params not loaded
- ```--shape-params``` : Shape parameter intervals. Format: [n1,min1,max1,n2,min2,max2,...]. default : sum(nX)==300
- ```--expression-params``` : Expression parameter intervals. Format: [n1,min1,max1,n2,min2,max2,...]. default : sum(nX)==100
- ```--pose-params``` : Pose parameter intervals. Format: [n1,min1,max1,n2,min2,max2,...]. sum(nX)==6 (min, max in degree)
- ```--texture-params``` : Texture parameter intervals. Format: [n1,min1,max1,n2,min2,max2,...]. default : sum(nX)==50, maximum : 200 (increase memory used)
- ```--neck-params``` : Neck parameter intervals. Format: [n1,min1,max1,n2,min2,max2,...]. sum(nX)==3 (min, max in degree)
- ```--eye-params``` : Eye parameter intervals. Format: [n1,min1,max1,n2,min2,max2,...]. sum(nX)==6
- ```--camera-params```: Camera parameter intervals. Format: [n1,min1,max1,n2,min2,max2,...]. sum(nX)==7, params order : [fov, tx, ty, tz, rx, ry, rz]. (rotation in degree)
- ```--fixed-shape``` : fixed the same shape for all visage generated
- ```--fixed-expression``` : fixed the same expression for all visage generated
- ```--fixed-pose``` : fixed the same jaw for all visage generated
- ```--fixed-texture``` : fixed the same texture for all visage generated
- ```--fixed-neck``` : fixed the same neck for all visage generated
- ```--fixed-eye``` : fixed the same eye for all visage generated
- ```--fixed-cameras``` : fixed the same cameras for all visage generated

### Flame parameter
- ```--not-use-face-contour``` : not use face contour for generate visage
- ```--not-use-3D-translation``` : not use 3D translation for generate visage

### Saving
- ```--outdir``` : path directory for output
- ```--lmk2D-format``` : format used for save landmarks 2D. (npy and pts is supported)
- ```--save-obj``` : enable save into file obj
- ```--save-png``` : enable save into file png
- ```--random-bg``` : enable random background color for renderer
- ```--save-lmks3D-npy``` : enable save landmarks 3D into file npy
- ```--save-lmks3D-png``` : enable save landmarks 3D with visage into file png
- ```--save-lmks2D``` : enable save landmarks 2D into file npy
- ```--save-markers``` : enable save markers into png file
- ```--img-resolution``` : resolution of image
- ```--show-window``` : show window during save png (enable if images is the screenshot or full black)
- ```--not-pts-in-alpha``` : not save landmarks/markers png version to channel alpha
- ```--save-camera-default``` : save camera in default format
- ```--save-camera-matrices``` : save camera in matrices format
- ```--save-camera-json``` : save camera in json format

### Path
- ```--flame-model-path``` : path for access flame model
- ```--static-landmark-embedding-path``` : path for static landmark embedding file
- ```--dynamic-landmark-embedding-path``` : path for dynamic landmark embedding file

You can define the parameters either by modifying the default values ​​in VisageGenerator.py file (@click.option(...))
or by launching the program, for example:
```
python ./VisageGenerator.py --nb-faces=1 --view --save-png --save-markers
```

***
## Keys
If you use parameter ```--view``` you have different keys for manipulation view and data :
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