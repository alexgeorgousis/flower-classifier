import argparse
from utils import *

IMAGE_SIZE = 224

parser = argparse.ArgumentParser()

# Specify CLI args and parse them
parser.add_argument("image_path")
parser.add_argument("model_path")
args = parser.parse_args()

# Get arg values
image_path = args.image_path
model_path = args.model_path

# Load model
model = load_model(model_path)

# Load and process image
image = load_process_image(image_path, IMAGE_SIZE)
