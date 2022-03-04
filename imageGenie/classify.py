from imageGenie.trainingModule import TrainingModule
"""

# pip install twine build

# pip install . >> every time you make changes , run from here

# python -m build >> get the .gz and .whl files

# python -m twine upload --repository testpypi dist/*
# pip install -i https://test.pypi.org/simple/ imageGenie==0.0.1

# python -m twine upload --repository pypi dist/*
# pip install imageGenie==0.0.5

* Figure out batch size and other params as per hardware resources
* allow user to customize the parameters like accuracy, epochs
* allow user to construct a model by themselves

"""

class Classifier(object):
    def __init__(self,root_folder,model_folder):
        self.root_folder = root_folder
        self.model_folder = model_folder

    def run(self):
        model =  TrainingModule(self.root_folder, self.model_folder)
        model.train()


if __name__ == "__main__":
    cl = Classifier("/media/heisenberg/Storage/PersonalFiles/Others/BPOST/bpost/ModelTraining/BigData", "/media/heisenberg/Storage/AllCodesSectionwise/DL_PROJECTS/auto-classifier/models")
    cl.classifier()