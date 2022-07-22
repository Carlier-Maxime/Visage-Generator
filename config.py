import argparse

parser = argparse.ArgumentParser(description='FLAME model')

parser.add_argument(
    '--save_obj',
    type=bool,
    default=True,
    help='enable save into file obj'
)

parser.add_argument(
    '--save_png',
    type=bool,
    default=False,
    help='enable save into file png'
)

parser.add_argument(
    '--save_lmks3D',
    type=bool,
    default=False,
    help='enable save landmarks 3D into file npy'
)

parser.add_argument(
    '--save_lmks2D',
    type=bool,
    default=False,
    help='enable save landmarks 2D into file npy'
)

parser.add_argument(
    '--min_shape_param',
    type=float,
    default=-2,
    help='minimum value for shape param'
)

parser.add_argument(
    '--max_shape_param',
    type=float,
    default=2,
    help='maximum value for shape param'
)

parser.add_argument(
    '--min_expression_param',
    type=float,
    default=-2,
    help='minimum value for expression param'
)

parser.add_argument(
    '--max_expression_param',
    type=float,
    default=2,
    help='maximum value for expression param'
)

parser.add_argument(
    '--global_pose_param_1',
    type=float,
    default=45,
    help='value of first global pose param'
)

parser.add_argument(
    '--global_pose_param_2',
    type=float,
    default=45,
    help='value of second global pose param'
)

parser.add_argument(
    '--global_pose_param_3',
    type=float,
    default=90,
    help='value of third global pose param'
)

parser.add_argument(
    '--device',
    type=str,
    default="cpu",
    help='choice your device for generate face. ("cpu" or "cuda")'
)

parser.add_argument(
    '--number_faces',
    type=int,
    default=1,
    help='number of faces to generate'
)

parser.add_argument(
    '--flame_model_path',
    type=str,
    default='./model/generic_model.pkl',
    help='flame model path'
)

parser.add_argument(
    '--static_landmark_embedding_path',
    type=str,
    default='./model/flame_static_embedding.pkl',
    help='Static landmark embeddings path for FLAME'
)

parser.add_argument(
    '--dynamic_landmark_embedding_path',
    type=str,
    default='./model/flame_dynamic_embedding.npy',
    help='Dynamic contour embedding path for FLAME'
)

# FLAME hyper-parameters

parser.add_argument(
    '--shape_params',
    type=int,
    default=300,
    help='the number of shape parameters'
)

parser.add_argument(
    '--expression_params',
    type=int,
    default=100,
    help='the number of expression parameters'
)

parser.add_argument(
    '--pose_params',
    type=int,
    default=6,
    help='the number of pose parameters'
)

# Training hyper-parameters

parser.add_argument(
    '--use_face_contour',
    default=True,
    type=bool,
    help='If true apply the landmark loss on also on the face contour.'
)

parser.add_argument(
    '--use_3D_translation',
    default=True,  # Flase for RingNet project
    type=bool,
    help='If true apply the landmark loss on also on the face contour.'
)

parser.add_argument(
    '--optimize_eyeballpose',
    default=True,  # False for For RingNet project
    type=bool,
    help='If true optimize for the eyeball pose.'
)

parser.add_argument(
    '--optimize_neckpose',
    default=True,  # False For RingNet project
    type=bool,
    help='If true optimize for the neck pose.'
)

parser.add_argument(
    '--num_worker',
    type=int,
    default=4,
    help='pytorch number worker.'
)

parser.add_argument(
    '--ring_margin',
    type=float,
    default=0.5,
    help='ring margin.'
)

parser.add_argument(
    '--ring_loss_weight',
    type=float,
    default=1.0,
    help='weight on ring loss.'
)


def get_config():
    config = parser.parse_args()
    return config
