"""
Chest ImaGenome dataset path should have a (sub-)directory called "silver_dataset" in its directory.
MIMIC-CXR and MIMIC-CXR-JPG dataset paths should both have a (sub-)directory called "files" in their directories.

Note that we only need the report txt files from MIMIC-CXR, which are in the file mimic-cxr-report.zip at
https://physionet.org/content/mimic-cxr/2.0.0/.

path_full_dataset specifies the path where the folder will be created (by module src/dataset/create_dataset.py) that will hold the
train, valid, test files, which will be used for training, evaluation and testing. See doc string of create_dataset.py for more information.
That means in my case, "/u/home/tanida/datasets/" should already exist as a directory, and the folder "dataset-with-reference-reports" would
be created in that directory by the module.

path_chexbert_weights specifies the path to the weights of the CheXbert labeler needed to extract the disease labels from the generated and reference reports.
The weights can be downloaded here: https://github.com/stanfordmlgroup/CheXbert#checkpoint-download

path_runs_* specify the directories where all the run folders (containing checkpoints, tensorboard files etc.) will be created
when training the object detector and full model (with and without the language model), respectively.
That means the directories specified by path_runs_* should already exist before starting the training.

path_test_set_evaluation_scores_txt_files will be the path where the txt files will be stored which contain the test set scores.
"""

path_chest_imagenome = "/home/user/yatpan/yatpan/rgrg/datasets/chest-imagenome-dataset"
path_mimic_cxr = "/home/user/yatpan/yatpan/rgrg/datasets/mimic-cxr"
path_mimic_cxr_jpg = "/home/user/yatpan/yatpan/rgrg/datasets/mimic-cxr-jpg"
path_full_dataset = "/home/user/yatpan/yatpan/rgrg/datasets/created_dataset"
path_chexbert_weights = "/home/user/yatpan/yatpan/rgrg/rgrg/src/CheXbert/src/models/chexbert.pth"
path_runs_object_detector = "/home/user/yatpan/yatpan/rgrg/runs/object_detector"
path_runs_full_model = "/home/user/yatpan/yatpan/rgrg/runs/full_model"
path_test_set_evaluation_scores_txt_files = "/home/user/yatpan/yatpan/rgrg/rgrg/src"