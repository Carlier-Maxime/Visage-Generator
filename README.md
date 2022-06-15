# Visage Generator

## Terms of use

This program uses the following resources:
- [FLAME PyTorch](https://github.com/soubhiksanyal/FLAME_PyTorch)
- [FLAME PyTorch Texture Fitting](https://github.com/HavenFeng/photometric_optimization)

Before any use of Visage Generator please read
conditions of use of resources.

***
## Config

to modify the config you must go to the config.py file
- **parser**.. : config path for model and other setting for FLAME pytorch
- **nbFace** : the number of face to generate. (default=1)
- **device** : choice your device for generate face. (default="cpu")
    - **cpu** : use processor
    - **cuda** : use graphic card (not compatible with all GC)

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

### Execute **main.py**

```
python main.py
```
they are advised to go into the **config.py** file and **set** the **device** variable to "**cuda**" if you have a **graphics card**.