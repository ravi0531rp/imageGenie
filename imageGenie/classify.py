from imageGenie.trainingModule import TrainingModule

"""

# pip install . >> every time you make changes , run from here
# python -m build >> get the .gz and .whl files
# python -m twine upload --repository pypi dist/*

                  /// TODO ///
* Handle all image formats
* Parse the specifications provided by the uer from a config file. That may include the priority
  of speed, accuracy, emphasis on False Positives or negatives, time available to experiment and train.
* Include all other model architectures like EfficientNet, MobileNet, Inception, VGG.
* Algorithm to figure out what architecture and hyper-params would be the best (in the fully automated mode) as per hardware.
* Save all other artefacts like pipeline, metrics, plots, etc 
* Allow user to construct a model by themselves
* Allow to either have a proper folder structure or a json with labels.
"""

class Classifier(object):
    def __init__(self,root_folder,model_folder):
        self.root_folder = root_folder
        self.model_folder = model_folder

    def run(self):
        model =  TrainingModule(self.root_folder, self.model_folder)
        model.train()


if __name__ == "__main__":
    cl = Classifier("/media/heisenberg/Storage/PersonalFiles/Others/ModelTraining/BigData", "/media/heisenberg/Storage/AllCodesSectionwise/DL_PROJECTS/auto-classifier/models")
    cl.classifier()