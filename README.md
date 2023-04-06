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
pip install -r requirements.txt
```

### Download Models

The information necessary for the download is indecate in the readme.md of the model folder or in the [link](https://github.com/Carlier-Maxime/Visage-Generator/blob/master/model/readme.md).

### Execute **VisageGenerator.py**

```
python VisageGenerator.py
```

***
## Config

Here are the different settings you can change:
- ```--nb-faces``` : number faces generate
- ```--lmk2D-format``` : format used for save lmk2d. (npy and pts is supported)
- ```--not-texturing``` : enable texture
- ```--save-obj``` : enable save into file obj
- ```--save-png``` : enable save into file png
- ```--save-lmks3D-npy``` : enable save landmarks 3D into file npy
- ```--save-lmks3D-png``` : enable save landmarks 3D with visage into file png
- ```--save-lmks2D``` : enable save landmarks 2D into file npy
- ```--min-shape-param``` : minimum value for shape param
- ```--max-shape-param``` : maximum value for shape param
- ```--min-expression-param``` : minimum value for expression param
- ```--max-expression-param``` : maximum value for expression param
- ```--global-pose-param1``` : value of first global pose param
- ```--global-pose-param2``` : value of second global pose param
- ```--global-pose-param3``` : value of third global pose param
- ```--device``` : choice your device for generate face. ("cpu" or "cuda")
- ```--view``` : enable view
- ```--flame-model-path``` : path for acess flame model
- ```--batch-size``` : number of visage generate in the same time
- ```--not-use-face-contour``` : not use face contour for generate visage
- ```--not-use-3D-translation``` : not use 3D translation for generate visage
- ```--shape-params``` : a number of shape parameter used
- ```--expression-params``` : a number of expression parameter used
- ```--static-landmark-embedding-path``` : path for static landmark embedding file
- ```--dynamic-landmark-embedding-path``` : path for dynamic landmark embedding file
- ```--not-optimize-eyeballpose``` : not optimize eyeballpose for generate visage
- ```--not-optimize-neckpose``` : not optimise neckpoes for generate visage
- ```--texture-batch-size``` : number of texture generate in same time
- ```--save-markers``` : enable save markers into png file
- ```--img-resolution``` : resolution of image

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
- **Ctrl** : Switch Ctrl on/off
- **Ctrl (On)** :
    - **S** : Save balises
    - **L** : Load balises
- **Ctrl (Off)** :
    - Use default pyrender action key
- **Edit marker (enable)** :
    - :arrow_left: (**Left arrow**) : direction in negatif X axis
    - :arrow_right: (**Right arrow**) : direction in positif X axis
    - :arrow_down: (**Down arrow**) : direction in negatif Y axis
    - :arrow_up: (**Up arrow**) : direction in positif Y axis
    - :arrow_double_down: (**Down Page**) : direction in negatif Z axis
    - :arrow_double_up: (**Up Page**) : direction in positif Z axis
    - **Enter** : add marker