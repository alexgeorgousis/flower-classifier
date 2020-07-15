import tensorflow as tf
import tensorflow_hub as hub
from PIL import Image
import numpy as np

def load_model(model_path):
    reloaded_model = tf.keras.models.load_model(model_path, custom_objects={'KerasLayer':hub.KerasLayer})
    return reloaded_model

def load_process_image(image_path, image_size):
    
    # Open image from file and turn to numpy array
    image = Image.open(image_path)
    image = np.asarray(image)

    # Conver to tensor, resize, normalise
    image = tf.convert_to_tensor(image, tf.float32)
    image = tf.image.resize(image, (image_size, image_size))
    image /= 255

    # Convert back to numpy array and add an extra dimension (needed by the model)
    image = image.numpy()
    image = np.expand_dims(image, axis=0)

    return image
