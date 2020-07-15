import argparse
from utils import *

IMAGE_SIZE = 224
N_CLASSES = 102

parser = argparse.ArgumentParser()

# Specify CLI args and parse them
parser.add_argument("image_path", help="specify the path to the image you want to classify")
parser.add_argument("model_path", help="specify the path to the Keras model (HDF5 file) you want to use to make the prediction")
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

# Report probability for each class
print("\n|-----------|")
print("Class: Probability")

for i in range(len(class_names)):
    print("{}: {}".format(class_names[i], np.format_float_positional(probs[i], precision=4)))
