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
        self.global_pose_param1=45
        self.global_pose_param2=45
        self.global_pose_param3=90
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
        self.img_resolution=[256,256]
        self.show_window=False

    def set(self, **kwargs):
        d = dict(kwargs)
        self.nb_faces=d['nb_faces']
        self.lmk2D_format=d['lmk2D_format']
        self.texturing=d['texturing']
        self.save_obj=d['save_obj']
        self.save_png=d['save_png']
        self.save_lmks3D_npy=d['save_lmks3D_npy']
        self.save_lmks3D_png=d['save_lmks3D_png']
        self.save_lmks2D=d['save_lmks2D']
        self.pts_in_alpha=d['pts_in_alpha']
        self.min_shape_param=d['min_shape_param']
        self.max_shape_param=d['max_shape_param']
        self.min_expression_param=d['min_expression_param']
        self.max_expression_param=d['max_expression_param']
        self.global_pose_param1=d['global_pose_param1']
        self.global_pose_param2=d['global_pose_param2']
        self.global_pose_param3=d['global_pose_param3']
        self.min_jaw_param1=d['min_jaw_param1']
        self.max_jaw_param1=d['max_jaw_param1']
        self.min_jaw_param2_3=d['min_jaw_param2_3']
        self.max_jaw_param2_3=d['max_jaw_param2_3']
        self.min_texture_param=d['min_texture_param']
        self.max_texture_param=d['max_texture_param']
        self.min_neck_param=d['min_neck_param']
        self.max_neck_param=d['max_neck_param']
        self.fixed_shape=d['fixed_shape']
        self.fixed_expression=d['fixed_expression']
        self.fixed_jaw=d['fixed_jaw']
        self.fixed_texture=d['fixed_texture']
        self.fixed_neck=d['fixed_neck']
        self.device=d['device']
        self.view=d['view']
        self.flame_model_path=d['flame_model_path']
        self.batch_size=d['batch_size']
        self.use_face_contour=d['use_face_contour']
        self.use_3D_translation=d['use_3D_translation']
        self.shape_params=d['shape_params']
        self.expression_params=d['expression_params']
        self.static_landmark_embedding_path=d['static_landmark_embedding_path']
        self.dynamic_landmark_embedding_path=d['dynamic_landmark_embedding_path']
        self.texture_batch_size=d['texture_batch_size']
        self.save_markers=d['save_markers']
        self.img_resolution=d['img_resolution']
        self.show_window=d['show_window']