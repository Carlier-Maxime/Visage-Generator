class Config:
    nb_faces=1000
    lmk2D_format="npy"
    texturing=True
    save_obj=False
    save_png=False
    save_lmks3D=False
    save_lmks2D=False
    min_shape_param=-2
    max_shape_param=2
    min_expression_param=-2
    max_expression_param=2
    global_pose_param1=45
    global_pose_param2=45
    global_pose_param3=90
    device="cuda"
    view=False
    flame_model_path='./model/generic_model.pkl'
    batch_size=128
    use_face_contour=True
    use_3D_translation=True
    shape_params=300
    expression_params=100
    static_landmark_embedding_path='./model/flame_static_embedding.pkl'
    dynamic_landmark_embedding_path='./model/flame_dynamic_embedding.npy'
    optimize_eyeballpose=True
    optimize_neckpose=True
    texture_batch_size=16