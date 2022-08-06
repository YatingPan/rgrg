from typing import List, Dict, Optional, Tuple

import torch
import torch.nn as nn

from binary_classifier.binary_classifier import BinaryClassifier
from src.object_detector.object_detector import ObjectDetector
from src.decoder.gpt2 import DecoderModel


class ReportGenerationModel(nn.Module):
    """
    Full model consisting of object detector encoder, binary classifier and language model decoder.
    """

    def __init__(self):
        super().__init__()
        self.object_detector = ObjectDetector(return_feature_vectors=True)
        path_to_best_object_detector_weights = "..."
        self.object_detector.load_state_dict(torch.load(path_to_best_object_detector_weights))

        self.binary_classifier = BinaryClassifier()

        self.language_model = DecoderModel()
        path_to_best_detector_weights = "..."
        self.language_model.load_state_dict(torch.load(path_to_best_detector_weights))

    def forward(
        self,
        images: torch.FloatTensor,  # images is of shape [batch_size, 1, 512, 512] (whole gray-scale images of size 512 x 512)
        image_targets: List[Dict],  # contains a dict for every image with keys "boxes" and "labels"
        input_ids: torch.LongTensor,  # shape [batch_size x 36 x seq_len], 1 sentence for every region for every image (sentence can be empty, i.e. "")
        attention_mask: torch.FloatTensor,  # shape [batch_size x 36 x seq_len]
        region_has_sentence: torch.BoolTensor,  # shape [batch_size x 36], ground truth boolean mask that indicates if a region has a sentence or not
        return_loss: bool = True,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        position_ids: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = False,
    ):
        """
        Forward method is used for training and evaluation of model.
        Generate method is used for inference.
        """
        # top_region_features of shape [batch_size, 36, 1024] (i.e. 1 feature vector for every region for every image in batch)
        # class_detected is a boolean tensor of shape [batch_size, 36]. Its value is True for a class if the object detector detected the class/region in the image

        if self.training:
            obj_detector_loss_dict, top_region_features, class_detected = self.object_detector(images, image_targets)

            # during training, only get the binary classifier loss
            binary_classifier_loss = self.binary_classifier(
                top_region_features, class_detected, return_loss=True, region_has_sentence=region_has_sentence
            )
            # to train the decoder, we want to use only the top region features (and corresponding input_ids, attention_mask)
            # of regions that were both detected by the object detector and have a sentence as the ground truth
            # this is done under the assumption that at inference time, the binary classifier will do an adequate job
            # at selecting those regions that need a sentence to be generated by itself
            valid_input_ids, valid_attention_mask, valid_region_features = self.get_valid_decoder_input_for_training(
                class_detected, region_has_sentence, input_ids, attention_mask, top_region_features
            )
        else:
            # during evaluation, also return detections (i.e. detected bboxes)
            obj_detector_loss_dict, detections, top_region_features, class_detected = self.object_detector(images, image_targets)

            # during evaluation, get the binary classifier loss, regions that were selected by the binary classifier (and that were also detected)
            # and the corresponding region features (selected_region_features)
            # this is done to evaluate the decoder under "real-word" conditions, i.e. the binary classifier decides which regions get a sentence
            binary_classifier_loss, selected_regions, selected_region_features = self.binary_classifier(
                top_region_features, class_detected, return_loss=True, region_has_sentence=region_has_sentence
            )

            # use the selected_regions mask to filter the inputs_ids and attention_mask to those that correspond to regions that were selected
            valid_input_ids, valid_attention_mask = self.get_valid_decoder_input_for_evaluation(selected_regions, input_ids, attention_mask)
            valid_region_features = selected_region_features

        language_model_loss = self.language_model(
            valid_input_ids,
            valid_attention_mask,
            valid_region_features,
            return_loss,
            past_key_values,
            position_ids,
            use_cache,
        )

        if self.training:
            return obj_detector_loss_dict, binary_classifier_loss, language_model_loss
        else:
            # class_detected needed to evaluate how good the object detector is at detecting the different regions during evaluation
            # detections and class_detected needed to compute IoU of object detector during evaluation
            # selected_regions needed to evaluate binary classifier during evaluation and to map each generated sentence to its corresponding region (for example for plotting)
            return (
                obj_detector_loss_dict,
                binary_classifier_loss,
                language_model_loss,
                detections,
                class_detected,
                selected_regions,
            )

    def get_valid_decoder_input_for_training(
        self,
        class_detected,  # shape [batch_size x 36]
        region_has_sentence,  # shape [batch_size x 36]
        input_ids,  # shape [batch_size x 36 x seq_len]
        attention_mask,  # shape [batch_size x 36 x seq_len]
        region_features,  # shape [batch_size x 36 x 1024]
    ):
        """
        We want to train the decoder only on region features (and corresponding input_ids/attention_mask) whose corresponding sentences are non-empty and
        that were detected by the object detector.

        Example:
            Let's assume region_has_sentence has shape [batch_size x 36] with batch_size = 2, so shape [2 x 36].
            This means we have boolean values for all 36 regions of the 2 images in the batch, that indicate if the
            regions have a corresponding sentence in the reference report or not.

            Now, let's assume region_has_sentence is True for the first 3 regions of each image. This means only the first
            3 regions of each image are described with sentences in the reference report.

            input_ids has shape [batch_size x 36 x seq_len].

            If we run valid_input_ids = input_ids[region_has_sentence], then we get valid_input_ids of shape [6 x seq_len].
            We thus get the first 3 rows of the first image, and the first 3 rows of the second image concatenated into 1 matrix.

            But we don't only select the input_ids/attention_mask/region_features via region_has_sentence, but also combine it (via logical and)
            with class_detected to only get the valid inputs to train the decoder.
        """
        valid = torch.logical_and(class_detected, region_has_sentence)

        valid_input_ids = input_ids[valid]  # of shape [num_detected_regions_with_non_empty_gt_phrase_in_batch x seq_len]
        valid_attention_mask = attention_mask[valid]  # of shape [num_detected_regions_with_non_empty_gt_phrase_in_batch x seq_len]
        valid_region_features = region_features[valid]  # of shape [num_detected_regions_with_non_empty_gt_phrase_in_batch x 1024]

        return valid_input_ids, valid_attention_mask, valid_region_features

    def get_valid_decoder_input_for_evaluation(
        self,
        selected_regions,
        input_ids,
        attention_mask
    ):
        valid_input_ids = input_ids[selected_regions]  # of shape [num_regions_selected_in_batch x seq_len]
        valid_attention_mask = attention_mask[selected_regions]  # of shape [num_regions_selected_in_batch x seq_len]

        return valid_input_ids, valid_attention_mask

    @torch.no_grad()
    def generate(
        self,
        images: torch.FloatTensor,  # images is of shape [batch_size, 1, 512, 512] (whole gray-scale images of size 512 x 512)
        max_length: int = None,
        num_beams: int = 1,
        num_beam_groups: int = 1,
        do_sample: bool = False,
        num_return_sequences: int = 1,
        early_stopping: bool = False,
    ):
        """
        In inference mode, we usually input 1 image (with 36 regions) at a time.

        The object detector first find the region features for all 36 regions.

        The binary classifier takes the region_features of shape [batch_size=1, 36, 1024] and returns:
            - selected_region_features: shape [num_regions_selected_in_batch, 1024],
            all region_features which were selected by the classifier to get a sentence generated (and which were also detected by the object detector)

            - selected_regions: shape [batch_size x 36], boolean matrix that indicates which regions were selected to get a sentences generated
            (these regions must also have been detected by the object detector).
            This is needed in case we want to find the corresponding reference sentences to compute scores for metrics such as BertScore or BLEU.

        The decoder then takes the selected_region_features and generates output ids for the batch.
        These output ids can then be decoded by the tokenizer to get the generated sentences.

        We also return selected_regions, such that we can map each generated sentence to a selected region.
        We also return detections, such that we can map each generated sentence to a bounding box.
        We also return class_detected to know which regions were not detected by the object detector (can be plotted).
        """
        # top_region_features of shape [batch_size, 36, 1024]
        _, detections, top_region_features, class_detected = self.object_detector(images)

        # selected_region_features is of shape [num_regions_selected_in_batch, 1024]
        # selected_regions is of shape [batch_size x 36] and is True for regions that should get a sentence
        # (it has exactly num_regions_selected_in_batch True values)
        selected_regions, selected_region_features = self.binary_classifier(
            top_region_features, class_detected, return_loss=False
        )

        # output_ids of shape (num_regions_selected_in_batch x longest_generated_sequence_length)
        output_ids = self.language_model.generate(
            selected_region_features,
            max_length,
            num_beams,
            num_beam_groups,
            do_sample,
            num_return_sequences,
            early_stopping,
        )

        return output_ids, selected_regions, detections, class_detected
