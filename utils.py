import tensorflow as tf
import tensorflow_hub as hub

def load_model(model_path):
    reloaded_model = tf.keras.models.load_model(model_path, custom_objects={'KerasLayer':hub.KerasLayer})
    return reloaded_model
