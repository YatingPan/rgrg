"""
Creates train, valid and test sets to train and evaluate object detector (as a standalone module), object detector + binary classifiers, and full model.

The original train.csv, valid.csv and test.csv in ChestIma have the following information each row:
1. Index(int): index of the column, starting from 0. Example: 0
2. subject_id(str): id of the patient. Example:10000980
3. study_id(str): id of the study, 1 patient can have one or more studies. Example: 50985099
4. dicom_id(str): id of the image. Example: 6ad03ed1-97ee17ee-9cf8b320-f7011003-cd93b42d
5. path(str): path of the image in mimic-cxr. Example: files/p10/p10000980/s50985099/6ad03ed1-97ee17ee-9cf8b320-f7011003-cd93b42d.dcm
6. ViewPosition(str): “AP”(beams enter from the font to back) or “PA”(beams enter from the back to font)
 
The new processed train.csv have the following information each row:
1. subject_id(str): same to the original
2. study_id(str): same to the original
3. image_id(str): dicom_id
4. path(str): same to the original
5. bbox_coordinates(list[list[int]]): bounding boxes’ coordinates of 29 regions in the image. The outlier list has a length less or equal to 29 and the inner liest has a length of 4(x1, x2, y1, y2). Some images don’t have bbox coordinates for all 29 regions. These images and missing images will be stored in log_file_dataset_creation.txt
6. bbox_labels(list[int]): a list of (usually) 29 region labels per image, corresponding to the bbox_coordinates.
7. bbox_phrases(list[int]): a list of (usually) 29 phrases per image, corresponding to the bbox_coordinates. If the region doesn’t have phrases, it will be “”
8. bbox_phrases_exists(list[boolean]): a list of (always) 29 booleans that indicates if a bbox has a phrases. 
9. bbox_is_abnormal(list[bool]): a list of (always) 29 booleans that indicates if a bbox is abnormal (True) by its phrases

The new valid.csv and test.csv have the additional information:
10. report(str): the “findings” of the mimic-cxr report, corresponding to the image 

For validation, we only include images that all 29 regions have bbox_coordinates and bbox_labels, to facilitate coding and vectorzition

For test,we split the test set into 2 parts:
1. test_bbox_all_regions.csv: this includes images whose all 29 regions have bbox_coordinates and bbox_labels. This part is about 95% of the original test
2. test_bbox_not_all_regions.csv: this includes image left
"""
import csv
import json
import logging
import os
import re

import imagesize
import spacy
import torch
from tqdm import tqdm

from src.dataset.constants import ANATOMICAL_REGIONS, IMAGE_IDS_TO_IGNORE, SUBSTRINGS_TO_REMOVE
import src.dataset.section_parser as sp
from src.path_datasets_and_weights import path_chest_imagenome, path_mimic_cxr, path_mimic_cxr_jpg, path_full_dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#print log to log_file_dataset_creation.txt to check the missing data points 
txt_file_for_logging = "log_file_dataset_creation.txt"

# out
logging.basicConfig(level=logging.INFO, format="[%(levelname)s]: %(message)s")
log = logging.getLogger(__name__)

# constant specifies how many rows to create in the customized csv files
# can be useful to create small sample datasets (e.g. of len 200) for testing things
# if NUM_ROWS_TO_CREATE_IN_NEW_CSV_FILES is None, then all possible rows are created
NUM_ROWS_TO_CREATE_IN_NEW_CSV_FILES = None

def write_stats_to_log_file(
    dataset: str,
    num_images_ignored_or_avoided: int,
    missing_images: list[str],
    missing_reports: list[str],
    num_faulty_bboxes: int,
    num_images_without_29_regions: int
):
    with open(txt_file_for_logging, "a") as f:
        f.write(f"{dataset}:\n")
        f.write(f"\tnum_images_ignored_or_avoided: {num_images_ignored_or_avoided}\n")

        f.write(f"\tnum_missing_images: {len(missing_images)}\n")
        for missing_img in missing_images:
            f.write(f"\t\tmissing_img: {missing_img}\n")

        f.write(f"\tnum_missing_reports: {len(missing_reports)}\n")
        for missing_rep in missing_reports:
            f.write(f"\t\tmissing_rep: {missing_rep}\n")

        f.write(f"\tnum_faulty_bboxes: {num_faulty_bboxes}\n")
        f.write(f"\tnum_images_without_29_regions: {num_images_without_29_regions}\n\n")


# Create new csv files for train, valid and test
def write_rows_in_new_csv_file(dataset: str, csv_rows: list[list]) -> None:
    log.info(f"Writing rows into new {dataset}.csv file...")

    if dataset == "test":
    # we write the test set into 2 csv files, one that contains all images that have bbox coordinates for all 29 regions, and one that contains the rest of the images
        csv_rows, csv_rows_less_than_29_regions = csv_rows

    new_csv_file_path = os.path.join(path_full_dataset, dataset) # e.g. path_full_dataset = "/home/user/yatpan/yatpan/rgrg/datasets/dataset-with-reference-reports", dataset = "train"
    new_csv_file_path += ".csv" if not NUM_ROWS_TO_CREATE_IN_NEW_CSV_FILES else f"-{NUM_ROWS_TO_CREATE_IN_NEW_CSV_FILES}.csv" # e.g. new_csv_file_path = "/home/user/yatpan/yatpan/rgrg/datasets/dataset-with-reference-reports/train.csv"

    # header of the csv file for train
    header = ["subject_id", "study_id", "image_id", "mimic_image_file_path", "bbox_coordinates", "bbox_labels", "bbox_phrases", "bbox_phrase_exists", "bbox_is_abnormal"]
    if dataset in ["valid", "test"]:
        # if dataset is valid or test, then we also have the reference report as a column
        header.append("reference_report")

    with open(new_csv_file_path, "w") as fp:
        csv_writer = csv.writer(fp)
        csv_writer.writerow(header)
        csv_writer.writerows(csv_rows)

    # if dataset is test, we write 2 csv files, one that contains all images that have bbox coordinates for all 29 regions, and one that contains the rest of the images
    if dataset == "test":
        new_csv_file_path = new_csv_file_path.replace("_bbox_all_regions.csv", "_bbox_not_all_regions.csv")

        with open(new_csv_file_path, "w") as fp:
            csv_writer = csv.writer(fp)
            csv_writer.writerow(header)
            csv_writer.writerows(csv_rows_less_than_29_regions)


def coordinates_faulty(x1, y1, x2, y2) -> bool:
    """
    Checks if bbox coordinates are faulty:
    1. if the area of the bbox is zero, i.e. if x1 == x2 or y1 == y2
    2. if the bounding box is within the image, i.e. if the bottom right corner ((x2, y2)) and the top left corner ((x1, y1)) are within the image
    Returns True if coordinates are faulty, False otherwise
    """
    if x1 >= x2 or y1 >= y2:  # bbox area <= 0
        return True
    elif x1 < 0 or x2 < 0 or y1< 0 or y2 < 0:  # coordinates not in the image
        return True
    elif x1 > 224 or x2 > 224 or y1 > 224 or y2 > 224:  #coordinates not in the image, here 224 is the resized image width and height
        return True
    return False

def determine_if_abnormal(attributes_list: list[list]) -> bool:
    """
    Extract the "nlp|yes|abnormal" attribute from the attributes_list and return True if it exists, else False
    """
    for attributes in attributes_list:
        for attribute in attributes:
            if attribute == "nlp|yes|abnormal":
                return True
    return False


def convert_phrases_to_single_string(phrases: list[str], sentence_tokenizer) -> str:
    """
    Input: list of phrases in the "attributes" dict for each bbox in the scene graph json file
    Output: single string that contains all phrases concatenated, so one bbox has one phrase string

    This function also performs text preprocessing on the phrases:
        - removes irrelevant substrings (like "PORTABLE UPRIGHT AP VIEW OF THE CHEST:")
        - removes whitespace characters (e.g. \n or \t) and redundant whitespaces
        - capitalizes the first word in each sentence
        - removes duplicate sentences
    """
    def remove_substrings(phrases):
        def remove_wet_read(phrases):
            # Removes substring like 'WET READ: ___ ___ 8:19 AM' that is irrelevant.
            # since there can be multiple WET READS's, collect the indices where they start and end in index_slices_to_remove
            index_slices_to_remove = []
            for index in range(len(phrases)):
                if phrases[index:index + 8] == "WET READ":

                    # curr_index searches for "AM" or "PM" that signals the end of the WET READ substring
                    for curr_index in range(index + 8, len(phrases)):
                        # since it's possible that a WET READ substring does not have an"AM" or "PM" that signals its end, we also have to break out of the iteration
                        # if the next WET READ substring is encountered
                        if phrases[curr_index:curr_index + 2] in ["AM", "PM"] or phrases[curr_index:curr_index + 8] == "WET READ":
                            break

                    # only add (index, curr_index + 2) (i.e. the indices of the found WET READ substring) to index_slices_to_remove if an "AM" or "PM" were found
                    if phrases[curr_index:curr_index + 2] in ["AM", "PM"]:
                        index_slices_to_remove.append((index, curr_index + 2))

            # remove the slices in reversed order, such that the correct index order is preserved
            for indices_tuple in reversed(index_slices_to_remove):
                start_index, end_index = indices_tuple
                phrases = phrases[:start_index] + phrases[end_index:]

            return phrases

        phrases = remove_wet_read(phrases)
        phrases = re.sub(SUBSTRINGS_TO_REMOVE, "", phrases, flags=re.DOTALL)

        return phrases

    def remove_whitespace(phrases):
        # remove all whitespace characters, e.g. \n or \t
        phrases = " ".join(phrases.split())
        return phrases

    def capitalize_first_word_in_sentence(phrases, sentence_tokenizer):
        sentences = sentence_tokenizer(phrases).sents
        # capitalize the first letter of each sentence
        phrases = " ".join(sent.text[0].upper() + sent.text[1:] for sent in sentences)

        return phrases

    def remove_duplicate_sentences(phrases):
        # remove the last period
        if phrases[-1] == ".":
            phrases = phrases[:-1]

        # dicts are insertion ordered as of Python 3.6
        phrases_dict = {phrase: None for phrase in phrases.split(". ")}
        phrases = ". ".join(phrase for phrase in phrases_dict)

        # add last period
        return phrases + "."

    # convert list of phrases into a single phrase
    phrases = " ".join(phrases)

    # remove "PORTABLE UPRIGHT AP VIEW OF THE CHEST:" and similar substrings from phrases, since they don't add any relevant information
    phrases = remove_substrings(phrases)

    # remove all whitespace characters (multiple whitespaces, newlines, tabs etc.)
    phrases = remove_whitespace(phrases)

    # for consistency, capitalize the 1st word in each sentence
    phrases = capitalize_first_word_in_sentence(phrases, sentence_tokenizer)

    phrases = remove_duplicate_sentences(phrases)

    return phrases


def get_attributes_dict(image_scene_graph: dict, sentence_tokenizer) -> dict[tuple]:
    attributes_dict = {}
    for attribute in image_scene_graph["attributes"]:
        region_name = attribute["bbox_name"]

        # ignore region_names that are not part of the 29 anatomical regions defined in constants.py
        if region_name not in ANATOMICAL_REGIONS:
            continue

        phrases = convert_phrases_to_single_string(attribute["phrases"], sentence_tokenizer)
        is_abnormal = determine_if_abnormal(attribute["attributes"])

        attributes_dict[region_name] = (phrases, is_abnormal)

    return attributes_dict


def get_reference_report(subject_id: str, study_id: str, missing_reports: list[str]):
    # custom_section_names and custom_indices specify reports that don't have "findings" sections
    custom_section_names, custom_indices = sp.custom_mimic_cxr_rules()

    if f"s{study_id}" in custom_section_names or f"s{study_id}" in custom_indices:
        return -1  # skip all reports without "findings" sections

    path_to_report = os.path.join(path_mimic_cxr, "files", f"p{subject_id[:2]}", f"p{subject_id}", f"s{study_id}.txt")

    if not os.path.exists(path_to_report):
        shortened_path_to_report = os.path.join(f"p{subject_id[:2]}", f"p{subject_id}", f"s{study_id}.txt")
        missing_reports.append(shortened_path_to_report)
        return -1

    with open(path_to_report) as f:
        report = "".join(f.readlines())

    # split report into sections
    # section_names is a list that specifies the found sections, e.g. ["indication", "comparison", "findings", "impression"]
    # sections is a list of same length that contains the corresponding text from the sections specified in section_names
    sections, section_names, _ = sp.section_text(report)

    if "findings" in section_names:
        # get index of "findings" by matching from reverse (has to do with how section_names is constructed)
        findings_index = len(section_names) - section_names[-1::-1].index("findings") - 1
        report = sections[findings_index]
    else:
        return -1  # skip all reports without "findings" sections

    # remove unnecessary whitespaces
    report = " ".join(report.split())

    return report


def get_total_num_rows(path_csv_file: str) -> int:
    with open(path_csv_file) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=",")

        # skip the first line (i.e. the header line)
        next(csv_reader)

        return sum(1 for row in csv_reader)


def get_rows(dataset: str, path_csv_file: str, image_ids_to_avoid: set) -> list[list]:
    """
    Args:
        dataset (str): "train", "valid" or "test
        path_csv_file (str): path to one of the csv files in the folder silver_dataset/splits of the chest-imagenome-dataset
        image_ids_to_avoid (set): as specified in "silver_dataset/splits/images_to_avoid.csv"

    Returns:
        csv_rows (list[list]): inner list contains information about a single image:
            - subject_id (str)
            - study_id (str)
            - image_id (str)
            - mimic_image_file_path (str): file path to image in mimic-cxr-jpg dataset
            - bbox_coordinates (list[list[int]]), where outer list usually has len 29 and inner list contains 4 bbox coordinates
            - bbox_labels (list[int]): list with class labels for each ground-truth box
            - bbox_phrases (list[str]): list with phrases for each bbox (note: phrases can be empty, i.e. "")
            - bbox_phrase_exist_vars (list[bool]): list that specifies if a phrase is non-empty (True) or empty (False) for a given bbox
            - bbox_is_abnormal_vars (list[bool]): list that specifies if a region depicted in a bbox is abnormal (True) or normal (False)

        valid.csv, test.csv and test-2.csv have the additional field:
            - reference_report (str): the findings section of the report extracted via https://github.com/MIT-LCP/mimic-cxr/tree/master/txt
    """
    csv_rows = []
    num_rows_created = 0

    # we split the test set into 1 that contains all images that have bbox coordinates for all 29 regions
    # (which will be around 31271 images in total, or around 95% of all test set images),
    # and 1 that contains the rest of the images (around 1440 images) that do not have bbox coordinates for all 29 regions
    # this is done such that we can efficiently evaluate the first test set (since vectorized code can be written for it),
    # and evaluate the second test set a bit more inefficiently (using for loops) afterwards
    if dataset == "test":
        csv_rows_less_than_29_regions = []

    total_num_rows = get_total_num_rows(path_csv_file)

    # used in function convert_phrases_to_single_string
    sentence_tokenizer = spacy.load("en_core_web_trf")

    # stats will be logged in path_to_log_file
    num_images_ignored_or_avoided = 0
    num_faulty_bboxes = 0
    num_images_without_29_regions = 0
    missing_images = []
    missing_reports = []

    with open(path_csv_file) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=",")

        # skip the first line (i.e. the header line)
        next(csv_reader)

        # iterate over all rows of the given csv file (i.e. over all images), if NUM_ROWS_TO_CREATE_IN_NEW_CSV_FILES is not set to a specific value
        for row in tqdm(csv_reader, total=total_num_rows):
            subject_id = row[1]
            study_id = row[2]
            image_id = row[3]

            # all images in set IMAGE_IDS_TO_IGNORE seem to be failed x-rays and thus have to be discarded
            # (they also don't have corresponding scene graph json files anyway)
            # all images in set image_ids_to_avoid are image IDs for images in the gold standard dataset,
            # which should all be excluded from model training and validation
            if image_id in IMAGE_IDS_TO_IGNORE or image_id in image_ids_to_avoid:
                num_images_ignored_or_avoided += 1
                continue

            # image_file_path is of the form "files/p10/p10000980/s50985099/6ad03ed1-97ee17ee-9cf8b320-f7011003-cd93b42d.dcm"
            # i.e. f"files/p{subject_id[:2]}/p{subject_id}/s{study_id}/{image_id}.dcm"
            # since we have the MIMIC-CXR-JPG dataset, we need to replace .dcm by .jpg
            image_file_path = row[4].replace(".dcm", ".jpg")
            mimic_image_file_path = os.path.join(path_mimic_cxr_jpg, image_file_path)

            if not os.path.exists(mimic_image_file_path):
                missing_images.append(mimic_image_file_path)
                continue

            # for the validation and test sets, we only want to include images that have corresponding reference reports with "findings" sections
            if dataset in ["valid", "test"]:
                reference_report = get_reference_report(subject_id, study_id, missing_reports)

                # skip images that don't have a reference report with "findings" section
                if reference_report == -1:
                    continue

                # the reference_report will be appended to new_image_row (declared further below, which contains all information about a single image)
                # just before new_image_row itself is appended to csv_rows (because the image could still be rejected from the validation set,
                # if it doesn't have 29 bbox coordinates)

            chest_imagenome_scene_graph_file_path = os.path.join(path_chest_imagenome, "silver_dataset", "scene_graph", image_id) + "_SceneGraph.json"

            with open(chest_imagenome_scene_graph_file_path) as fp:
                image_scene_graph = json.load(fp)

            # get the attributes specified for the specific image in its image_scene_graph
            # the attributes contain (among other things) phrases used in the reference report to describe different bbox regions and
            # information whether a described bbox region is normal or abnormal
            #
            # anatomical_region_attributes is a dict with bbox_names as keys and lists that contain 2 elements as values. The 2 list elements are:
            # 1. (normalized) phrases, which is a single string that contains the phrases used to describe the region inside the bbox
            # 2. is_abnormal, a boolean that is True if the region inside the bbox is considered abnormal, else False for normal
            anatomical_region_attributes = get_attributes_dict(image_scene_graph, sentence_tokenizer)

            # new_image_row will store all information about 1 image as a row in the csv file
            new_image_row = [subject_id, study_id, image_id, mimic_image_file_path]
            bbox_coordinates = []
            bbox_labels = []
            bbox_phrases = []
            bbox_phrase_exist_vars = []
            bbox_is_abnormal_vars = []

            # counter to see if given image contains bbox coordinates for all 29 regions
            # if image does not bbox coordinates for 29 regions, it's still added to the train and test dataset,
            # but not the val dataset
            num_regions = 0

            region_to_bbox_coordinates_dict = {}
            # objects is a list of obj_dicts where each dict contains the bbox coordinates for a single region
            for obj_dict in image_scene_graph["objects"]:
                region_name = obj_dict["bbox_name"]
                x1 = obj_dict["original_x1"]
                y1 = obj_dict["original_y1"]
                x2 = obj_dict["original_x2"]
                y2 = obj_dict["original_y2"]

                region_to_bbox_coordinates_dict[region_name] = [x1, y1, x2, y2]

            for anatomical_region in ANATOMICAL_REGIONS:
                bbox_coords = region_to_bbox_coordinates_dict.get(anatomical_region, None)

                # if there are no bbox coordinates or they are faulty, then don't add them to image information
                if bbox_coords is None or coordinates_faulty(x1, y1, x2, y2):
                    num_faulty_bboxes += 1
                else:
                    # get bbox coordinates
                    bbox_coords = [x1, y1, x2, y2]

                    # since background has class label 0 for object detection, shift the remaining class labels by 1
                    class_label = ANATOMICAL_REGIONS[anatomical_region] + 1

                    bbox_coordinates.append(bbox_coords)
                    bbox_labels.append(class_label)

                    num_regions += 1

                # get bbox_phrase (describing the region inside bbox) and bbox_is_abnormal boolean variable (indicating if region inside bbox is abnormal)
                # if there is no phrase, then the region inside bbox is normal and thus has "" for bbox_phrase (empty phrase) and False for bbox_is_abnormal
                bbox_phrase, bbox_is_abnormal = anatomical_region_attributes.get(anatomical_region, ("", False))
                bbox_phrase_exist = True if bbox_phrase != "" else False

                bbox_phrases.append(bbox_phrase)
                bbox_phrase_exist_vars.append(bbox_phrase_exist)
                bbox_is_abnormal_vars.append(bbox_is_abnormal)

            new_image_row.extend([bbox_coordinates, bbox_labels, bbox_phrases, bbox_phrase_exist_vars, bbox_is_abnormal_vars])

            # for train set, add all images (even those that don't have bbox information for all 29 regions)
            # for val set, only add images that have bbox information for all 29 regions
            # for test set, distinguish between test set 1 that contains test set images that have bbox information for all 29 regions
            # (around 95% of all test set images)
            if dataset == "train" or (dataset in ["valid", "test"] and num_regions == 29):
                if dataset in ["valid", "test"]:
                    new_image_row.append(reference_report)

                csv_rows.append(new_image_row)

                num_rows_created += 1
            # test set 2 will contain the remaining 5% of test set images, which do not have bbox information for all 29 regions
            elif dataset == "test" and num_regions != 29:
                new_image_row.append(reference_report)
                csv_rows_less_than_29_regions.append(new_image_row)

            if num_regions != 29:
                num_images_without_29_regions += 1

            # break out of loop if NUM_ROWS_TO_CREATE_IN_NEW_CSV_FILES is specified
            if NUM_ROWS_TO_CREATE_IN_NEW_CSV_FILES and num_rows_created >= NUM_ROWS_TO_CREATE_IN_NEW_CSV_FILES:
                break

    write_stats_to_log_file(dataset, num_images_ignored_or_avoided, missing_images, missing_reports, num_faulty_bboxes, num_images_without_29_regions)

    if dataset == "test":
        return csv_rows, csv_rows_less_than_29_regions
    else:
        return csv_rows


def create_new_csv_file(dataset: str, path_csv_file: str, image_ids_to_avoid: set) -> None:
    log.info(f"Creating new {dataset}.csv file...")

    # get rows to create new csv_file
    # csv_rows is a list of lists, where an inner list specifies all information about a single image
    csv_rows = get_rows(dataset, path_csv_file, image_ids_to_avoid)

    # write those rows into a new csv file
    write_rows_in_new_csv_file(dataset, csv_rows)

    log.info(f"Creating new {dataset}.csv file... DONE!")


def create_new_csv_files(csv_files_dict, image_ids_to_avoid):
    # Check if the directory already exists
    if not os.path.exists(path_full_dataset):
        # If it doesn't exist, create the directory
        os.mkdir(path_full_dataset)
        log.info(f"Created new dataset folder at {path_full_dataset}.")
    else:
        # If it does exist, log that we're updating the existing folder
        log.info(f"Updating existing dataset folder at {path_full_dataset}.")

    # Create or update the CSV files within the existing directory
    for dataset, path_csv_file in csv_files_dict.items():
        create_new_csv_file(dataset, path_csv_file, image_ids_to_avoid)


def get_images_to_avoid():
    path_to_images_to_avoid = os.path.join(path_chest_imagenome, "silver_dataset", "splits", "images_to_avoid.csv")

    image_ids_to_avoid = set()

    with open(path_to_images_to_avoid) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=",")

        # skip the first line (i.e. the header line)
        next(csv_reader)

        for row in csv_reader:
            image_id = row[2]
            image_ids_to_avoid.add(image_id)

    return image_ids_to_avoid


def get_train_val_test_csv_files():
    """Return a dict with datasets as keys and paths to the corresponding csv files in the chest-imagenome dataset as values"""
    path_to_splits_folder = os.path.join(path_chest_imagenome, "silver_dataset", "splits")
    return {dataset: os.path.join(path_to_splits_folder, dataset) + ".csv" for dataset in ["train", "valid", "test"]}


def main():
    csv_files_dict = get_train_val_test_csv_files()

    # the "splits" directory of chest-imagenome contains a csv file called "images_to_avoid.csv",
    # which contains image IDs for images in the gold standard dataset, which should all be excluded
    # from model training and validation
    image_ids_to_avoid = get_images_to_avoid()

    create_new_csv_files(csv_files_dict, image_ids_to_avoid)


if __name__ == "__main__":
    main()