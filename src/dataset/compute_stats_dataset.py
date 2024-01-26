from collections import defaultdict
import csv
import json
import os

from tqdm import tqdm

from src.dataset.constants import ANATOMICAL_REGIONS, IMAGE_IDS_TO_IGNORE

from src.path_datasets_and_weights import path_chest_imagenome

"""
This script is to check the original Chest ImaGenome dataset and compute statistics about it.
Specifically, it identified the following data points that we should not use:
1. data points with faulty bbox coordinates, i.e. if the area of the bbox is zero or if the bounding box is out of the image;
2. data points with faulty bbox names, i.e. if the bbox name is not one of the 29 anatomical regions;

It also finds the following data relationships that we should pay attention to:
1. the matching between bbox and phrases, i.e. if a bbox has a phrase, multiple phrases, or no phrase at all;
2. the matching between bbox and abnormality, i.e. if a bbox is normal or abnormal.
"""
# output the statistics into a txt file
txt_file_to_log_stats = "./dataset_stats.txt"


def print_stats_counter_dicts(counter_dict):
    """Print the counts in descending order"""
    with open(txt_file_to_log_stats, "a") as f:
        total_count = sum(value for value in counter_dict.values())
        for bbox_name, count in sorted(counter_dict.items(), key=lambda k_v: k_v[1], reverse=True):
            f.write(f"\n\t\t{bbox_name}: {count:,} ({(count/total_count) * 100:.2f}%)")


def log_stats_to_txt_file(dataset: str, stats: dict) -> None:
    if dataset != "Total":
        num_images = stats["num_images"]
        num_ignored_images = stats["num_ignored_images"]
        num_bboxes = stats["num_bboxes"]
        num_normal_bboxes = stats["num_normal_bboxes"]
        num_abnormal_bboxes = stats["num_abnormal_bboxes"]
        num_bboxes_with_phrases = stats["num_bboxes_with_phrases"]
        num_outlier_bboxes = stats["num_outlier_bboxes"]
        bbox_with_phrases_counter_dict = stats["bbox_with_phrases_counter_dict"]
        outlier_bbox_counter_dict = stats["outlier_bbox_counter_dict"]
    else:
        num_images = stats["total_num_images"]
        num_ignored_images = stats["total_num_ignored_images"]
        num_bboxes = stats["total_num_bboxes"]
        num_normal_bboxes = stats["total_num_normal_bboxes"]
        num_abnormal_bboxes = stats["total_num_abnormal_bboxes"]
        num_bboxes_with_phrases = stats["total_num_bboxes_with_phrases"]
        num_outlier_bboxes = stats["total_num_outlier_bboxes"]
        bbox_with_phrases_counter_dict = stats["total_bbox_with_phrases_counter_dict"]
        outlier_bbox_counter_dict = stats["total_outlier_bbox_counter_dict"]

    with open(txt_file_to_log_stats, "a") as f:
        f.write(f"\n\n{dataset}:")
        f.write(f"\n\t{num_images:,} images in total")
        f.write(f"\n\t{num_ignored_images} images were ignored (due to faulty x-rays etc.)")

        f.write(f"\n\n\t{num_bboxes:,} bboxes in total")
        f.write(f"\n\t{num_normal_bboxes:,} normal bboxes in total")
        f.write(f"\n\t{num_abnormal_bboxes:,} abnormal bboxes in total")
        f.write(f"\n\t{num_bboxes_with_phrases:,} bboxes have corresponding phrases")
        f.write(f"\n\t-> {(num_bboxes_with_phrases/num_bboxes) * 100:.2f}% of bboxes have corresponding phrases")

        f.write(f"\n\n\t{num_outlier_bboxes:,} 'outlier' regions that don't have bboxes but have phrases:")
        f.write(f"\n\t-> {(num_outlier_bboxes/num_bboxes_with_phrases) * 100:.2f}% of overall bboxes with phrases")

        f.write("\n\n\tCounts and percentages of 'outlier' regions without bboxes:")
        print_stats_counter_dicts(outlier_bbox_counter_dict)

        f.write("\n\n\tCounts and percentages of normal bboxes with phrases:")
        print_stats_counter_dicts(bbox_with_phrases_counter_dict)


def coordinates_faulty(height, width, x1, y1, x2, y2) -> bool:
    """
    Checks if bbox coordinates are faulty:
    1. if the area of the bbox is zero, i.e. if x1 == x2 or y1 == y2
    2. if the bounding box is within the image, i.e. if the bottom right corner ((x2, y2)) and the top left corner ((x1, y1)) are within the image
    Returns True if coordinates are faulty, False otherwise
    """
    if x1==x2 or y1==y2:
        return True
    elif x2 <= 0 or y2 <= 0:
        return True
    elif x1 >= width or y1 >= height:
        return True
    return False


def determine_if_abnormal(attributes_list: list[list]) -> bool:
    """
    attributes in ChestIma is a list of dictionaries, where each dictionary represents a region, and in each dictionary there are lists of attributes, attributes_ids, phrases, phrases_ids.
    The basic data format for attributes is:
        "attributes": [
        {
            "right apical zone": true,
            "bbox_name": "right apical zone",
            "synsets": [
                "C0929167"
            ],
            "name": "Apical zone of right lung",
            "attributes": [
                [
                    "anatomicalfinding|yes|pneumothorax",
                    "nlp|yes|abnormal"
                ]
            ],
            "attributes_ids": [
                [
                    "C1963215;;C0032326",
                    "C0205161"
                ]
            ],
            "phrases": [
                "However, a small medial pneumothorax\n has newly occurred."
            ],
            "phrase_IDs": [
                "55916528|6"
            ],
            "sections": [
                "finalreport"
            ],
            "comparison_cues": [
                []
            ],
            "temporal_cues": [
                [
                    "temporal|yes|acute"
                ]
            ],
            "severity_cues": [
                [
                    "severity|yes|mild"
                ]
            ],
            "texture_cues": [
                [
                    "texture|yes|lucency"
                ]
            ],
            "object_id": "d7bef063-28053f7a-f27dae40-4035348b-21a36d32_right apical zone"
        },
        ...
    The "attributes" list is extracted from the phrases in the "phrases" list. As the phrase is "However, a small medial pneumothorax\n has newly occurred.", and the attributes is "anatomicalfinding|yes|pneumothorax","nlp|yes|abnormal"
    One region can have multiple phrases and cooresponding attributes.
    We determine if a region is abnormal by checking if the attributes list contains "nlp|yes|abnormal".
    """
    for attributes in attributes_list:
        for attribute in attributes:
            if attribute == "nlp|yes|abnormal":
                return True
    # no abnormality could be detected
    return False


def update_stats_for_image(image_scene_graph: dict, stats: dict) -> None:
    is_abnormal_dict = {}
    for attribute in image_scene_graph["attributes"]:
        bbox_name = attribute["bbox_name"]
        # there are bbox_names such as "left chest wall" or "right breast" that are not part of the 29 anatomical regions
        # they are considered outliers
        if bbox_name not in ANATOMICAL_REGIONS:
            stats["num_outlier_bboxes"] += 1
            stats["outlier_bbox_counter_dict"][bbox_name] += 1
        else:
            stats["num_bboxes_with_phrases"] += 1
            stats["bbox_with_phrases_counter_dict"][bbox_name] += 1

        is_abnormal = determine_if_abnormal(attribute["attributes"])
        is_abnormal_dict[bbox_name] = is_abnormal

    return is_abnormal_dict


def get_num_rows(path_csv_file: str) -> int:
    with open(path_csv_file) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=",")
        # skip the first line (i.e. the header line)
        next(csv_reader)
        return sum(1 for row in csv_reader)


def compute_stats_for_csv_file(dataset: str, path_csv_file: str, image_ids_to_avoid: set) -> dict:
    stats = {
        stat: 0
        for stat in [
            "num_images",
            "num_ignored_images",
            "num_bboxes",
            "num_normal_bboxes",
            "num_abnormal_bboxes",
            "num_bboxes_with_phrases",
            "num_outlier_bboxes",
        ]
    }
    stats["bbox_with_phrases_counter_dict"] = defaultdict(int)
    stats["outlier_bbox_counter_dict"] = defaultdict(int)
    stats["num_images"] += get_num_rows(path_csv_file)

    with open(path_csv_file) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=",")

        # skip the first line (i.e. the header line)
        next(csv_reader)

        # iterate over all rows of the given csv file (i.e. over all images)
        for row in tqdm(csv_reader, total=stats["num_images"]):
            image_id = row[3]

            # all images in set IMAGE_IDS_TO_IGNORE seem to be failed x-rays and thus have to be discarded
            # (they also don't have corresponding scene graph json files anyway)
            if image_id in IMAGE_IDS_TO_IGNORE:
                stats["num_ignored_images"] += 1
                continue

            # all images in set image_ids_to_avoid are image IDs for images in the gold standard dataset,
            # which should all be excluded from model training and validation
            if image_id in image_ids_to_avoid:
                continue

            chest_imagenome_scene_graph_file_path = (
                os.path.join(path_chest_imagenome, "silver_dataset", "scene_graph", image_id) + "_SceneGraph.json"
            )

            with open(chest_imagenome_scene_graph_file_path) as fp:
                image_scene_graph = json.load(fp)

            # update num_bboxes_with_phrases and num_outlier_bboxes based on current image
            # also update the bbox_with_phrases and outlier_bbox counter dicts
            # returns a is_abnormal_dict that specifies if bboxes mentioned in report are normal or abnormal
            is_abnormal_dict = update_stats_for_image(image_scene_graph, stats)

            # for each image, there are normally 29 bboxes for 29 anatomical regions
            for anatomical_region in image_scene_graph["objects"]:
                bbox_name = anatomical_region["bbox_name"]

                if bbox_name not in ANATOMICAL_REGIONS:
                    continue

                stats["num_bboxes"] += 1

                if is_abnormal_dict.get(bbox_name, False):
                    stats["num_abnormal_bboxes"] += 1
                else:
                    stats["num_normal_bboxes"] += 1

    log_stats_to_txt_file(dataset=dataset, stats=stats)

    return stats


def compute_and_print_stats_for_csv_files(csv_files_dict, image_ids_to_avoid):
    total_stats = {
        stat: 0
        for stat in [
            "total_num_images",
            "total_num_ignored_images",  
            "total_num_bboxes",
            "total_num_normal_bboxes",
            "total_num_abnormal_bboxes",
            "total_num_bboxes_with_phrases",
            "total_num_outlier_bboxes",  
        ]
    }
    total_stats["total_bbox_with_phrases_counter_dict"] = defaultdict(int)  # dict to count how often each of the 29 anatomical regions have phrases
    total_stats["total_outlier_bbox_counter_dict"] = defaultdict(int)  # dict to count how often each of the outlier regions have phrases

    for dataset, path_csv_file in csv_files_dict.items():
        stats = compute_stats_for_csv_file(dataset, path_csv_file, image_ids_to_avoid)

        for key, value in stats.items():
            if key not in ["bbox_with_phrases_counter_dict", "outlier_bbox_counter_dict"]:
                total_stats["total_" + key] += value
            else:
                for bbox_name, count in value.items():  # value is a counter dict in this case
                    total_stats["total_" + key][bbox_name] += count

    log_stats_to_txt_file(dataset="Total", stats=total_stats)


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
    path_to_splits_folder = os.path.join(path_chest_imagenome, "silver_dataset", "splits")
    return {dataset: os.path.join(path_to_splits_folder, dataset) + ".csv" for dataset in ["train", "valid", "test"]}


def main():
    csv_files_dict = get_train_val_test_csv_files()

    # the "splits" directory of chest-imagenome contains a csv file called "images_to_avoid.csv",
    # which contains image IDs for images in the gold standard dataset, which should all be excluded
    # from model training and validation
    image_ids_to_avoid = get_images_to_avoid()

    compute_and_print_stats_for_csv_files(csv_files_dict, image_ids_to_avoid)


if __name__ == "__main__":
    main()
