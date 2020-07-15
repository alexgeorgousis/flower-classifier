# Overview
Image classifier built with TensorFlow to classify different kinds of flowers. This project includes a Jupyter Notebook which shows how the model was trained, and a Python application that can be used to classify new images of flowers using the trained model. You can read the report of the project [here](https://alexgeorgousis.github.io/flower-classifier/flower_classifier.html).

This project was completed as part of the Introduction to Machine Learning Nanodegree on [Udacity](https://www.udacity.com/course/intro-to-machine-learning-with-tensorflow-nanodegree--nd230). 

# Usage
To classify a new image, clone this repository and run `predict.py` with an image of a flower and `best_model.h5` or a model of your choice:
```
$ git clone https://github.com/alexgeorgousis/flower-classifier.git
$ cd flower-classifier
$ python predict.py ./test_images/wild_pansy.jpg ./best_model.h5
```

Run `$ python predict.py -h` for a list of optional CLI arguments.

# Dependencies
See `environment.yml` for the full list of dependencies. If you use conda, you can set up a local Python environment that includes all of the dependencies:
```
$ conda env create --file environment.yml
```
