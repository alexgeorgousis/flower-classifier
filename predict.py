import argparse
import json
from utils import *

IMAGE_SIZE = 224
N_CLASSES = 102

parser = argparse.ArgumentParser()

# Specify CLI args and parse them
parser.add_argument("image_path", help="specify the path to the image you want to classify")
parser.add_argument("model_path", help="specify the path to the Keras model (HDF5 file) you want to use to make the prediction")
parser.add_argument("--category_names", help="specify a JSON file that contains the category names")
parser.add_argument("--top_k", help="display only the top k classes and their probabilities")
args = parser.parse_args()

# Get arg values
image_path = args.image_path
model_path = args.model_path

# Load model
model = load_model(model_path)

# Load and process image
image = load_process_image(image_path, IMAGE_SIZE)

# Make prediction to get the probabilities for each class
probs = model.predict(image)[0]

# Get class names
class_names = list(range(1, N_CLASSES+1))

# Map class numbers to category names (if relevant CLI argument was passed)
if args.category_names:

    # Load dictionary from JSON file
    with open(args.category_names, 'r') as f:
        class_names_map = json.load(f)

    # Map class numbers to category names
    class_names = [class_names_map[str(n)] for n in class_names]

# Report probability for each class
print("\n|-----------|")
print("Class: Probability")

for i in range(len(class_names)):
    print("{}: {}".format(class_names[i], np.format_float_positional(probs[i], precision=4)))
