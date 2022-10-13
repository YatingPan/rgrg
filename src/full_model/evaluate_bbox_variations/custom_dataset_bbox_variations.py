import cv2
import torch
from torch.utils.data import Dataset


class CustomDatasetBboxVariations(Dataset):
    def __init__(self, dataset_as_df, transforms, log):
        super().__init__()
        self.dataset_as_df = dataset_as_df
        self.transforms = transforms
        self.log = log

    def __len__(self):
        return len(self.dataset_as_df)

    def __getitem__(self, index):
        # get the image_path for potential logging in except block
        image_path = self.dataset_as_df.iloc[index]["mimic_image_file_path"]

        # if something in __get__item fails, then return None
        # collate_fn in dataloader filters out None values
        try:
            bbox_coordinates_varied = self.dataset_as_df.iloc[index]["bbox_coordinates_varied"]  # List[List[int]] of shape 29 x 4
            bbox_labels = self.dataset_as_df.iloc[index]["bbox_labels"]  # List[int] of len 29
            bbox_phrases = self.dataset_as_df.iloc[index]["bbox_phrases"]  # List[str] of len 29

            # cv2.imread by default loads an image with 3 channels
            # since we have grayscale images, we only have 1 channel and thus use cv2.IMREAD_UNCHANGED to read in the 1 channel
            image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

            # apply transformations to image, bbox_coordinates and bbox_labels
            transformed = self.transforms(image=image, bboxes=bbox_coordinates_varied, class_labels=bbox_labels)

            transformed_image = transformed["image"]  # torch.tensor of shape 1 x 512 x 512 and dtype=float
            transformed_bbox_coordinates = transformed["bboxes"]  # List[List[float]]  of shape 29 x 4

            sample = {
                "image": transformed_image,
                "bbox_coordinates": torch.tensor(transformed_bbox_coordinates),
                "bbox_reference_sentences": bbox_phrases,  # List[str]
            }

        except Exception as e:
            self.log.error(f"__getitem__ failed for: {image_path}")
            self.log.error(f"Reason: {e}")
            return None

        return sample
