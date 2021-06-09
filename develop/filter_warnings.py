import warnings

# A bunch of warnings that clutter my logs, maybe I should start using loguru

# https://github.com/dmlc/dgl/pull/2902
warnings.filterwarnings("ignore", message="DGLGraph\.__len__")
# See readme
warnings.filterwarnings("ignore", message="Undefined\ type\ encountered")
# Uncomment this whenever you change something in the symbol encoder!
warnings.filterwarnings("ignore", message="Misaligned\ pointer\ detected")
