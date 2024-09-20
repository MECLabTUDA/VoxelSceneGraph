from scene_graph_prediction.utils.config import AccessTrackingCfgNode

# -------------------------------------------------------------------------------------------------------------------- #
# Config related to input processing / data augmentation
# -------------------------------------------------------------------------------------------------------------------- #

INPUT = AccessTrackingCfgNode()

# Number of dimensions in an input image (excluding channels)
INPUT.N_DIM = 2
# Number of channels in an input image
INPUT.N_CHANNELS = 3
# Number of object (/ segmentation mask) classes (including the "background" class)
INPUT.N_OBJ_CLASSES = 2
# Number of object classes that have exactly one object per image
# These classes are assumed to be ordered last e.g. for stroke (bg, bleed, ventricle system, middle line)
# Currently only used by the Stroke RetinaUNet
INPUT.N_UNIQUE_OBJ_CLASSES = 0
# Number of attribute classes
INPUT.N_ATT_CLASSES = 0
# Number of image attribute classes
INPUT.N_IMG_ATT_CLASSES = 0
# Number of keypoint classes (including the "background" class)
INPUT.N_KP_CLASSES = 1
# Number of relation classes (including the "background" class)
INPUT.N_REL_CLASSES = 1
# Proposal height and width both need to be greater than MIN_SIZE
INPUT.MIN_SIZE = (0,)

# -------------------------------------------------------------------------------------------------------------------- #
# Config related to data transforms/augmentation
# -------------------------------------------------------------------------------------------------------------------- #
# Key used to load the correct list of transforms from the transform-scheme registry (see data/transforms/build.py)
# This is here we sometimes load RGB PIL Image or 3D tensors from Nifti image,
# and all transformations are not compatible or do not always make sense.
INPUT.TRANSFORM_SCHEME = "default"

# ResizeImage2D: Size of the smallest side of the image during training
INPUT.MIN_SIZE_TRAIN = 800
# ResizeImage2D: Maximum size of the side of the image during training
INPUT.MAX_SIZE_TRAIN = 1333
# ResizeImage2D: Size of the smallest side of the image during testing
INPUT.MIN_SIZE_TEST = 800
# ResizeImage2D: Maximum size of the side of the image during testing
INPUT.MAX_SIZE_TEST = 1333

# ResizeTensor: Size of the resized tensor XYZ ordered
# Note: one value per dimension
INPUT.RESIZE = (0, 0)

# RandomHorizontalFlip: flip prob
# No test prob defined as it is always 0 during testing
INPUT.HORIZONTAL_FLIP_PROB_TRAIN = 0.0

# RandomVerticalFlip: flip prob
# No test prob defined as it is always 0 during testing
INPUT.VERTICAL_FLIP_PROB_TRAIN = 0.0

# RandomDepthFlip: flip prob
# No test prob defined as it is always 0 during testing
INPUT.DEPTH_FLIP_PROB_TRAIN = 0.0

# ColorJitter:
INPUT.BRIGHTNESS = 0.0
INPUT.CONTRAST = 0.0
INPUT.SATURATION = 0.0
INPUT.HUE = 0.0

# Normalize: Values to be used for image normalization
INPUT.PIXEL_MEAN = [102.9801, 115.9465, 122.7717]
# Normalize: Values to be used for image normalization
INPUT.PIXEL_STD = [1., 1., 1.]

# ClipAndRescale: Clip to range and rescale to [0, 1]
INPUT.CLIP_MIN = 0.
INPUT.CLIP_MAX = 100.

# BoundingBoxPerturbation: Maximum shift for bounding boxes.
# Note: must be an integer
INPUT.MAX_BB_SHIFT = 0

# RandomAffine: Affine transformation within following range
INPUT.AFFINE_MAX_TRANSLATE = (0,)  # Either one int or one per dim
INPUT.AFFINE_SCALE_RANGE = ((1., 1.),)  # Either a pair of float or one per dim
INPUT.AFFINE_MAX_ROTATE = (0.,)  # Either one float or one per dim
