import argparse

parser = argparse.ArgumentParser()

parser.add_argument("image_path")
parser.add_argument("model_path")
args = parser.parse_args()

image_path = args.image_path
model_path = args.model_path
