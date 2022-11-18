# Human-in-the-Loop: Region-guided Radiology Report Generation

## Setup

1. Create virtual environment (Python 3.10.4)
2. Run "**pip install -e .**" in the root directory (i.e. region-guided-chest-x-ray-report-generation) to install src as a standalone package. This allows imports from sibling directories.
3. Run "**pip install -r requirements.txt**"[^1]
4. In src/path_datasets_and_weights.py, specify the paths to the various datasets (Chest ImaGenome, MIMIC-CXR, MIMIC-CXR-JPG), CheXbert weights, and folders in which the runs are saved. Follow the instructions of the doc string of path_datasets_and_weights.py.

## Create train, validation and test csv files

After following the 4 setup steps, run "**python create_dataset.py**" in src/dataset/ to create training, validation and test csv files, in which each row contains specific information about a single image. See doc string of create_dataset.py for more details.

## Training

The full model is trained in 3 training stages:

1. Object detector
2. Object detector + abnormality classification module + region selection module
3. Full model end-to-end

### Object detector

For training the object detector, specify the training configurations (e.g. batch size etc.) in lines 32 - 49 of src/object_detector/training_script_object_detector.py, then run "**python training_script_object_detector.py**".
The weights of the trained object detector model will be stored in the folder specified in src/path_datasets_and_weights.py

### Object detector + abnormality classification module + region selection module

For the second training stage, first specify the path to the best trained object detector in report generation model (see line 26 of src/full_model/report_generation_model.py), such that the trained object detector will be trained together with the 2 binary classifiers.
Next, specify the run configurations in src/full_model/run_configurations.py. In particular, set "**PRETRAIN_WITHOUT_LM_MODEL = True**",
such that the language model is fully excluded from training. See doc string of src/full_model/run_configurations.py for more details.
Start training by running "**python train_full_model.py**" in src/full_model/.

### Full model

For the third training stage, again adjust the run configurations in src/full_model/run_configurations.py (e.g., the batch size may have to be lowered, since the full model requires a lot of memory). In particular, set "**PRETRAIN_WITHOUT_LM_MODEL = False**", such that the full model is trained end-to-end. Next, specify the checkpoint of the best pre-trained model of training stage 2 in the main function (line 567) of src/full_model/train_full_model.py, such this pre-trained model is loaded at beginning of training. Start training by running "**python train_full_model.py**" in src/full_model/.
 
During each training stage, the validation metrics and other useful information (such as images with bounding boxes and generated sentences etc.) are logged to tensorboard files saved in the corresponding run folders (specified in path_datasets_and_weights.py). Additionally, for the 3rd training stage, txt files with generated reports and sentences are saved in the run folders. 

## Testing

Specify the run and checkpoint to be tested in lines 40 - 41 of src/full_model/test_set_evaluation.py, then run "**python test_set_evaluation.py**". Txt files with the test set scores (and generated reports/sentences) will be saved in src/.

[^1]: Recommended: first install torch via https://pytorch.org/get-started/locally/ (since your local setup is likely different to mine), then comment out torch and torchvision in the requirements file before running the pip install command.

