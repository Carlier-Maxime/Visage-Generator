class Config:
    def __init__(self):
        self.nb_faces=1000
        self.lmk2D_format="npy"
        self.texturing=True
        self.save_obj=False
        self.save_png=False
        self.save_lmks3D_npy=False
        self.save_lmks3D_png=False
        self.save_lmks2D=False
        self.pts_in_alpha=True
        self.min_shape_param=-2
        self.max_shape_param=2
        self.min_expression_param=-2
        self.max_expression_param=2
        self.min_rotation_param = -30
        self.max_rotation_param = 30
        self.min_jaw_param1=0
        self.max_jaw_param1=30
        self.min_jaw_param2_3=-10
        self.max_jaw_param2_3=10
        self.min_texture_param=-2
        self.max_texture_param=2
        self.min_neck_param = -30
        self.max_neck_param = 30
        self.fixed_shape = False
        self.fixed_expression = False
        self.fixed_jaw = False
        self.fixed_texture = False
        self.fixed_neck = False
        self.device="cuda"
        self.view=False
        self.flame_model_path='./model/generic_model.pkl'
        self.batch_size=128
        self.use_face_contour=True
        self.use_3D_translation=True
        self.shape_params=300
        self.expression_params=100
        self.static_landmark_embedding_path='./model/flame_static_embedding.pkl'
        self.dynamic_landmark_embedding_path='./model/flame_dynamic_embedding.npy'
        self.texture_batch_size=16
        self.save_markers=False
        self.img_resolution=[512,512]
        self.show_window=False
        self.outdir = 'output'
        self.input_folder = None
        self.zeros_params = False
        self.save_camera_default = False
        self.save_camera_matrices = False
        self.save_camera_json = False
        self.camera = [10.,0.,0.,-2.,0.,0.,0.]
        self.pose_for_camera = False
        self.random_bg = False

    def set(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
        self.img_resolution = self._str_to_list(self.img_resolution, int)
        self.camera = self._str_to_list(self.camera, float)

    def _str_to_list(self, value, type):
        value = value[1:-1].split(",")
        value = [type(v) for v in value]
        return value