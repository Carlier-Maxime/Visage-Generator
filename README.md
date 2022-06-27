# Visage Generator

## Terms of use

This program uses the following resources:
- [FLAME PyTorch](https://github.com/soubhiksanyal/FLAME_PyTorch)
- [FLAME PyTorch Texture Fitting](https://github.com/HavenFeng/photometric_optimization)

Before any use of Visage Generator please read
conditions of use of resources.

***
## Config

Here are the different settings you can change:
- ```--min_shape_param``` : minimum value of shape param.
- ```--max_shape_param``` : maximum value of shape param.
- ```--min_expression_param``` : minimum value of expression param.
- ```--max_expression_param``` : maximum value of expression param.
- ```--global_pose_param_1``` : value of first global pose param.
- ```--global_pose_param_2``` : value of second global pose param.
- ```--global_pose_param_3``` : value of third global pose param.
- ```--device``` : choice your device for generate face.
- ```--number_faces``` : the number of face to generate.
- ```--flame_model_path``` : path to model flame.
- ```--static_landmark_embedding_path``` : path to static landmark embedding for FLAME.
- ```--dynamic_landmark_embedding_path``` : path to dynamic landmark embedding for FLAME.
- ```--shape_params``` : the number of shape parameters.
- ```--expression_params``` : the number of expression parameters.
...

For more information check config.py
You can define the parameters either by modifying the default values ​​in my config.py file 
or by launching the program, for example:
```
python ./VisageGenerator.py --device=cuda --number_faces=10
```

***
## Keys
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
they are advised to go into the **config.py** file and **set** the **device** variable to "**cuda**" if you have a **graphics card**,
or specify the device used for the launch :
```
python VisageGenerator.py --device=cuda
```