## ImageGenie

* Training a model to classify images between different classes . 
* This single package lets us harness the power of the state of the art models without any hassle of
  coding them ourselves.
* Just 3 lines of code and we're done.

### Installation

```
pip install imageGenie
```

### COLAB Notebook Demo
https://colab.research.google.com/drive/1DGgrENv-XTVeRz7PsOm0tpofJFZWn6PU?usp=sharing

### Usage

1) Fully Automated Mode

* Folder Structure

main folder 

```
from imageGenie.classify import Classifier # import the Classifier Class

cl = Classifier("/root", "/models") # arg1 -> base directory containing train & test ; arg2 -> saving directory

cl.run() # this trains the model by automatically finding out number of classes, types of images and optimum training epochs.

```

2) Controlled mode (Work in progress)


## TODO
* Handle all image formats
* Parse the specifications provided by the uer from a config file. That may include the priority
  of speed, accuracy, emphasis on False Positives or negatives, time available to experiment and train.
* Include all other model architectures like EfficientNet, MobileNet, Inception, VGG.
* Algorithm to figure out what architecture and hyper-params would be the best (in the fully automated mode) as per hardware.
* Save all other artefacts like pipeline, metrics, plots, etc 
* Allow user to construct a model by themselves
* Allow to either have a proper folder structure or a json with labels.