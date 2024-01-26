from collections import defaultdict
import csv
import json
import os

from tqdm import tqdm

from src.dataset.constants import ANATOMICAL_REGIONS, IMAGE_IDS_TO_IGNORE

from src.path_datasets_and_weights import path_chest_imagenome

"""
This script is to the faulty data in the ChestIma dataset. 
It finds the following data points that we should remove from the dataset:
1. data points with faulty bbox coordinates, i.e. if the area of the bbox is zero or if the bounding box is out of the image;
2. data points with faulty bbox names, i.e. if the bbox name is not one of the 29 anatomical regions;

The output has 2 csv files:
    - faulty_bbox_coordinates.csv:
      This csv file contains all data points with faulty bbox coordinates. Each row has the following information:
      subject_id, study_id, image_id, path, object, bbox_coordinates
    - faulty_bbox_names.csv:
      This csv file contains all data points with bbox names not in the 29 regions. Each row has the following information:
      subject_id, study_id, image_id, path, object, bbox_name
"""

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

def bbox_name_faulty(bbox_name: str) -> bool:
    """
    Checks if bbox name is faulty, i.e. if the bbox name is not one of the 29 anatomical regions
    Returns True if bbox name is faulty, False otherwise
    """
    return bbox_name not in ANATOMICAL_REGIONS

def find_faulty_data(path_to_json_files: str) -> list:
    faulty_coord_bbox = []
    faulty_name_bbox = []
    for filename in tqdm(os.listdir(path_to_json_files)):
        if not filename.endswith("_SceneGraph.json") or filename.replace("_SceneGraph.json", "") in IMAGE_IDS_TO_IGNORE:
            continue

        image_id = filename.replace("_SceneGraph.json", "")
        json_file_path = os.path.join(path_to_json_files, filename)

        with open(json_file_path) as fp:
            image_scene_graph = json.load(fp)
            for obj in image_scene_graph["objects"]:
                obj_id = obj["object_id"]
                bbox_name_faulty_flag = bbox_name_faulty(obj["bbox_name"])
                bbox_coordinates_faulty_flag = coordinates_faulty(obj["x1"], obj["y1"], obj["x2"], obj["y2"])

                if bbox_coordinates_faulty_flag:
                    faulty_coord_bbox.append([image_scene_graph["patient_id"], image_scene_graph["study_id"], image_id, json_file_path, obj_id, [obj["x1"], obj["y1"], obj["x2"], obj["y2"]]])

                if bbox_name_faulty_flag:
                    faulty_name_bbox.append([image_scene_graph["patient_id"], image_scene_graph["study_id"], image_id, json_file_path, obj_id, obj["bbox_name"]])

    return faulty_coord_bbox, faulty_name_bbox

def write_rows_in_csv_file(rows: list, path_csv_file: str, header: list) -> None:
    with open(path_csv_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(header)
        writer.writerows(rows)

def main() -> None:
    path_to_json_files = os.path.join(path_chest_imagenome, "silver_dataset", "scene_graph")
    faulty_coord_data, faulty_name_data = find_faulty_data(path_to_json_files)

    path_faulty_coord_csv = os.path.join(path_chest_imagenome, "silver_dataset", "splits", "faulty_bbox_coordinates.csv")
    write_rows_in_csv_file(faulty_coord_data, path_faulty_coord_csv, ["subject_id", "study_id", "image_id", "path", "object", "bbox_coordinates"])

    path_faulty_name_csv = os.path.join(path_chest_imagenome, "silver_dataset", "splits", "faulty_bbox_names.csv")
    write_rows_in_csv_file(faulty_name_data, path_faulty_name_csv, ["subject_id", "study_id", "image_id", "path", "object", "bbox_name"])

if __name__ == '__main__':
    main()