"""
This module contains all functions used to evaluate the full model.

The (main) function evaluate_model of this module is called by the function train_model in train_full_model.py
every K steps and also at the end of every epoch.

The K is specified by the EVALUATE_EVERY_K_STEPS variable in run_configurations.py

evaluate_model and its sub-functions evaluate among other things:

    - total val loss as well as the val losses of each individual module (i.e. model component)
    - object detector:
        - average IoU of region (ideally 1.0 for every region)
        - average num detected regions per image (ideally 36.0)
        - average num each region is detected in an image (ideally 1.0 for every region)
    - binary classifier region selection:
        - precision and recall for all regions, regions that have gt = normal (i.e. the region was considered normal by the radiologist),
        regions that have gt = abnormal (i.e. the region was considered abnormal by the radiologist)
    - binary classifier region abnormal detection:
        - precision and recall for all regions
    - language model (is evaluated in separate evaluate_language_model.py module):
        - BLEU 1-4 and BertScore for all generated sentences, generated sentences with gt = normal,
        generated sentences with gt = abnormal
        - NUM_BATCHES_OF_GENERATED_SENTENCES_TO_SAVE_TO_FILE (see run_configurations.py) batches of generated sentences
        are saved as a txt file (for manual verification what the model generates)
        - NUM_IMAGES_TO_PLOT images are saved to tensorboard where gt and predicted bboxes for every region
        are depicted, as well as the generated sentences (if they exist) and reference sentences for every region
"""

from copy import deepcopy
import os

import torch
import torchmetrics
from tqdm import tqdm

from src.dataset.constants import ANATOMICAL_REGIONS
from src.full_model.evaluate_full_model.evaluate_language_model import evaluate_language_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def log_stats_to_console(
    log,
    train_loss,
    val_loss,
    epoch,
):
    log.info(f"Epoch: {epoch}:")
    log.info(f"\tTrain loss: {train_loss:.3f}")
    log.info(f"\tVal loss: {val_loss:.3f}")


def write_all_losses_and_scores_to_tensorboard(
    writer,
    overall_steps_taken,
    train_losses_dict,
    val_losses_dict,
    obj_detector_scores,
    region_selection_scores,
    region_abnormal_scores,
    language_model_scores,
    current_lr,
):
    def write_losses(writer, overall_steps_taken, train_losses_dict, val_losses_dict):
        for loss_type in train_losses_dict:
            writer.add_scalars(
                "_loss",
                {f"{loss_type}_train": train_losses_dict[loss_type], f"{loss_type}_val": val_losses_dict[loss_type]},
                overall_steps_taken,
            )

    def write_obj_detector_scores(writer, overall_steps_taken, obj_detector_scores):
        writer.add_scalar(
            "avg_num_detected_regions_per_image",
            obj_detector_scores["avg_num_detected_regions_per_image"],
            overall_steps_taken,
        )

        # replace white space by underscore for each region name (i.e. "right upper lung" -> "right_upper_lung")
        anatomical_regions = ["_".join(region.split()) for region in ANATOMICAL_REGIONS]
        avg_detections_per_region = obj_detector_scores["avg_detections_per_region"]
        avg_iou_per_region = obj_detector_scores["avg_iou_per_region"]

        for region_, avg_detections_region in zip(anatomical_regions, avg_detections_per_region):
            writer.add_scalar(f"num_detected_{region_}", avg_detections_region, overall_steps_taken)

        for region_, avg_iou_region in zip(anatomical_regions, avg_iou_per_region):
            writer.add_scalar(f"iou_{region_}", avg_iou_region, overall_steps_taken)

    def write_region_selection_scores(writer, overall_steps_taken, region_selection_scores):
        for subset in region_selection_scores:
            for metric, score in region_selection_scores[subset].items():
                writer.add_scalar(f"region_select_{subset}_{metric}", score, overall_steps_taken)

    def write_region_abnormal_scores(writer, overall_steps_taken, region_abnormal_scores):
        for metric, score in region_abnormal_scores.items():
            writer.add_scalar(f"region_abnormal_{metric}", score, overall_steps_taken)

    def write_language_model_scores(writer, overall_steps_taken, language_model_scores):
        for subset in language_model_scores:
            for metric, score in language_model_scores[subset].items():
                writer.add_scalar(f"language_model_{subset}_{metric}", score, overall_steps_taken)

    write_losses(writer, overall_steps_taken, train_losses_dict, val_losses_dict)
    write_obj_detector_scores(writer, overall_steps_taken, obj_detector_scores)
    write_region_selection_scores(writer, overall_steps_taken, region_selection_scores)
    write_region_abnormal_scores(writer, overall_steps_taken, region_abnormal_scores)
    write_language_model_scores(writer, overall_steps_taken, language_model_scores)

    writer.add_scalar("lr", current_lr, overall_steps_taken)


def update_region_abnormal_metrics(region_abnormal_scores, predicted_abnormal_regions, region_is_abnormal, class_detected):
    """
    Args:
        region_abnormal_scores (Dict)
        predicted_abnormal_regions (Tensor[bool]): shape [batch_size x 36]
        region_is_abnormal (Tensor[bool]): shape [batch_size x 36]
        class_detected (Tensor[bool]): shape [batch_size x 36]

    We only update/compute the scores for regions that were actually detected by the object detector (specified by class_detected).
    """
    detected_predicted_abnormal_regions = predicted_abnormal_regions[class_detected]
    detected_region_is_abnormal = region_is_abnormal[class_detected]

    region_abnormal_scores["precision"](detected_predicted_abnormal_regions, detected_region_is_abnormal)
    region_abnormal_scores["recall"](detected_predicted_abnormal_regions, detected_region_is_abnormal)


def update_region_selection_metrics(region_selection_scores, selected_regions, region_has_sentence, region_is_abnormal):
    """
    Args:
        region_selection_scores (Dict[str, Dict])
        selected_regions (Tensor[bool]): shape [batch_size x 36]
        region_has_sentence (Tensor[bool]): shape [batch_size x 36]
        region_is_abnormal (Tensor[bool]): shape [batch_size x 36]
    """
    normal_selected_regions = selected_regions[~region_is_abnormal]
    normal_region_has_sentence = region_has_sentence[~region_is_abnormal]

    abnormal_selected_regions = selected_regions[region_is_abnormal]
    abnormal_region_has_sentence = region_has_sentence[region_is_abnormal]

    region_selection_scores["all"]["precision"](selected_regions.reshape(-1), region_has_sentence.reshape(-1))
    region_selection_scores["all"]["recall"](selected_regions.reshape(-1), region_has_sentence.reshape(-1))

    region_selection_scores["normal"]["precision"](normal_selected_regions, normal_region_has_sentence)
    region_selection_scores["normal"]["recall"](normal_selected_regions, normal_region_has_sentence)

    region_selection_scores["abnormal"]["precision"](abnormal_selected_regions, abnormal_region_has_sentence)
    region_selection_scores["abnormal"]["recall"](abnormal_selected_regions, abnormal_region_has_sentence)


def update_object_detector_metrics(obj_detector_scores, detections, image_targets, class_detected):
    def compute_box_area(box):
        """
        Calculate the area of a box given the 4 corner values.

        Args:
            box (Tensor[batch_size x 36 x 4])

        Returns:
            area (Tensor[batch_size x 36])
        """
        x0 = box[..., 0]
        y0 = box[..., 1]
        x1 = box[..., 2]
        y1 = box[..., 3]

        return (x1 - x0) * (y1 - y0)

    def compute_intersection_and_union_area_per_region(detections, targets, class_detected):
        # pred_boxes is of shape [batch_size x 36 x 4] and contains the predicted region boxes with the highest score (i.e. top-1)
        # they are sorted in the 2nd dimension, meaning the 1st of the 36 boxes corresponds to the 1st region/class,
        # the 2nd to the 2nd class and so on
        pred_boxes = detections["top_region_boxes"]

        # targets is a list of dicts, with each dict containing the key "boxes" that contain the gt boxes of a single image
        # gt_boxes is of shape [batch_size x 36 x 4]
        gt_boxes = torch.stack([t["boxes"] for t in targets], dim=0)

        # below tensors are of shape [batch_size x 36]
        x0_max = torch.maximum(pred_boxes[..., 0], gt_boxes[..., 0])
        y0_max = torch.maximum(pred_boxes[..., 1], gt_boxes[..., 1])
        x1_min = torch.minimum(pred_boxes[..., 2], gt_boxes[..., 2])
        y1_min = torch.minimum(pred_boxes[..., 3], gt_boxes[..., 3])

        # intersection_boxes is of shape [batch_size x 36 x 4]
        intersection_boxes = torch.stack([x0_max, y0_max, x1_min, y1_min], dim=-1)

        # below tensors are of shape [batch_size x 36]
        intersection_area = compute_box_area(intersection_boxes)
        pred_area = compute_box_area(pred_boxes)
        gt_area = compute_box_area(gt_boxes)

        union_area = (pred_area + gt_area) - intersection_area

        # if x0_max >= x1_min or y0_max >= y1_min, then there is no intersection
        valid_intersection = torch.logical_and(x0_max < x1_min, y0_max < y1_min)

        # also there is no intersection if the class was not detected by object detector
        valid_intersection = torch.logical_and(valid_intersection, class_detected)

        # set all non-valid intersection areas to 0
        intersection_area = torch.where(
            valid_intersection,
            intersection_area,
            torch.tensor(0, dtype=intersection_area.dtype, device=intersection_area.device),
        )

        # sum up the values along the batch dimension (the values will divided by each other later to get the averages)
        intersection_area = torch.sum(intersection_area, dim=0)
        union_area = torch.sum(union_area, dim=0)

        return intersection_area, union_area

    # sum up detections for each region
    region_detected_batch = torch.sum(class_detected, dim=0)

    intersection_area_per_region_batch, union_area_per_region_batch = compute_intersection_and_union_area_per_region(detections, image_targets, class_detected)

    obj_detector_scores["sum_region_detected"] += region_detected_batch
    obj_detector_scores["sum_intersection_area_per_region"] += intersection_area_per_region_batch
    obj_detector_scores["sum_union_area_per_region"] += union_area_per_region_batch


def get_val_losses_and_other_metrics(model, val_dl):
    """
    Args:
        model (nn.Module): The input model to be evaluated.
        val_dl (torch.utils.data.Dataloder): The val dataloader to evaluate on.

    Returns:
        val_losses_dict (Dict): holds different val losses of the different modules as well as the total val loss
        obj_detector_scores (Dict): holds scores of the average IoU per Region, average number of detected regions per image,
        average number each region is detected in an image
        region_selection_scores (Dict): holds precision and recall scores for all, normal and abnormal sentences
        region_abnormal_scores (Dict): holds precision and recall scores for all sentences
    """
    val_losses_dict = {
        "total_loss": 0.0,
        "obj_detector_loss": 0.0,
        "region_selection_loss": 0.0,
        "region_abnormal_loss": 0.0,
        "language_model_loss": 0.0,
    }

    num_images = 0

    """
    For the object detector, besides the obj_detector_val_loss, we also want to compute:
      - the average IoU for each region,
      - average number of detected regions per image (ideally 36.0)
      - average number each region is detected in an image (ideally 1.0 for all regions)

    To compute these metrics, we allocate several tensors:

    sum_intersection_area_per_region: for accumulating the intersection area of each region
    (will be divided by union area of each region at the end of get the IoU for each region)

    sum_union_area_per_region: for accumulating the union area of each region
    (will divide the intersection area of each region at the end of get the IoU for each region)

    sum_region_detected: for accumulating the number of times a region is detected over all images
    (this 1D array will be divided by num_images to get the average number each region is detected in an image,
    and these averages will be summed up to get the average number of detected regions in an image)
    """
    obj_detector_scores = {}
    obj_detector_scores["sum_intersection_area_per_region"] = torch.zeros(36, device=device)
    obj_detector_scores["sum_union_area_per_region"] = torch.zeros(36, device=device)
    obj_detector_scores["sum_region_detected"] = torch.zeros(36, device=device)

    """
    For the binary classifier for region selection, we want to compute the precision and recall for:
      - all regions
      - normal regions
      - abnormal regions

    Evaluation according to:
      TP: (normal/abnormal) region has sentence (gt), and is selected by classifier to get sentence (pred)
      FP: (normal/abnormal) region does not have sentence (gt), but is selected by classifier to get sentence (pred)
      TN: (normal/abnormal) region does not have sentence (gt), and is not selected by classifier to get sentence (pred)
      FN: (normal/abnormal) region has sentence (gt), but is not selected by classifier to get sentence (pred)
    """
    region_selection_scores = {}
    region_selection_scores["all"] = {
        "precision": torchmetrics.Precision(num_classes=2, average=None).to(device),
        "recall": torchmetrics.Recall(num_classes=2, average=None).to(device),
    }
    region_selection_scores["normal"] = {
        "precision": torchmetrics.Precision(num_classes=2, average=None).to(device),
        "recall": torchmetrics.Recall(num_classes=2, average=None).to(device),
    }
    region_selection_scores["abnormal"] = {
        "precision": torchmetrics.Precision(num_classes=2, average=None).to(device),
        "recall": torchmetrics.Recall(num_classes=2, average=None).to(device),
    }

    """
    For the binary classifier for region normal/abnormal detection, we want to compute the precision and recall for:
      - all regions

    Evaluation according to:
      TP: region is abnormal (gt), and is predicted as abnormal by classifier (pred)
      FP: region is normal (gt), but is predicted as abnormal by classifier (pred)
      TN: region is normal (gt), and is predicted as normal by classifier (pred)
      FN: region is abnormal (gt), but is predicted as normal by classifier (pred)
    """
    region_abnormal_scores = {
        "precision": torchmetrics.Precision(num_classes=2, average=None).to(device),
        "recall": torchmetrics.Recall(num_classes=2, average=None).to(device),
    }

    with torch.no_grad():
        for batch in tqdm(val_dl):
            # "image_targets" maps to a list of dicts, where each dict has the keys "boxes" and "labels" and corresponds to a single image
            # "boxes" maps to a tensor of shape [36 x 4] and "labels" maps to a tensor of shape [36]
            # note that the "labels" tensor is always sorted, i.e. it is of the form [1, 2, 3, ..., 36] (starting at 1, since 0 is background)
            images = batch["images"]
            image_targets = batch["image_targets"]
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            region_has_sentence = batch["region_has_sentence"]
            region_is_abnormal = batch["region_is_abnormal"]

            batch_size = images.size(0)
            num_images += batch_size

            # put all tensors on the GPU
            images = images.to(device, non_blocking=True)
            image_targets = [{k: v.to(device, non_blocking=True) for k, v in t.items()} for t in image_targets]
            input_ids = input_ids.to(device, non_blocking=True)
            attention_mask = attention_mask.to(device, non_blocking=True)
            region_has_sentence = region_has_sentence.to(device, non_blocking=True)
            region_is_abnormal = region_is_abnormal.to(device, non_blocking=True)

            # detections is a dict with keys "top_region_boxes" and "top_scores"
            # "top_region_boxes" maps to a tensor of shape [batch_size x 36 x 4]
            # "top_scores" maps to a tensor of shape [batch_size x 36]
            #
            # class_detected is a tensor of shape [batch_size x 36]
            # selected_regions is a tensor of shape [batch_size x 36]
            # predicted_abnormal_regions is a tensor of shape [batch_size x 36]
            (
                obj_detector_loss_dict,
                classifier_loss_region_selection,
                classifier_loss_region_abnormal,
                language_model_loss,
                detections,
                class_detected,
                selected_regions,
                predicted_abnormal_regions,
            ) = model(images, image_targets, input_ids, attention_mask, region_has_sentence, region_is_abnormal)

            # sum up all 4 losses from the object detector
            obj_detector_losses = sum(loss for loss in obj_detector_loss_dict.values())

            # sum up the rest of the losses
            total_loss = obj_detector_losses + classifier_loss_region_selection + classifier_loss_region_abnormal + language_model_loss

            list_of_losses = [
                total_loss,
                obj_detector_losses,
                classifier_loss_region_selection,
                classifier_loss_region_abnormal,
                language_model_loss,
            ]

            # dicts are insertion ordered since Python 3.7
            for loss_type, loss in zip(val_losses_dict, list_of_losses):
                val_losses_dict[loss_type] += loss.item() * batch_size

            # update scores for object detector metrics
            update_object_detector_metrics(obj_detector_scores, detections, image_targets, class_detected)

            # update scores for region selection metrics
            update_region_selection_metrics(region_selection_scores, selected_regions, region_has_sentence, region_is_abnormal)

            # update scores for region abnormal detection metrics
            update_region_abnormal_metrics(region_abnormal_scores, predicted_abnormal_regions, region_is_abnormal, class_detected)

    # normalize the val losses by steps_taken (i.e. len(val_dl))
    for loss_type in val_losses_dict:
        val_losses_dict[loss_type] /= len(val_dl)

    # compute object detector scores
    sum_intersection = obj_detector_scores["sum_intersection_area_per_region"]
    sum_union = obj_detector_scores["sum_union_area_per_region"]
    obj_detector_scores["avg_iou_per_region"] = (sum_intersection / sum_union).tolist()

    sum_region_detected = obj_detector_scores["sum_region_detected"]
    obj_detector_scores["avg_num_detected_regions_per_image"] = torch.sum(sum_region_detected / num_images).item()
    obj_detector_scores["avg_detections_per_region"] = (sum_region_detected / num_images).tolist()

    # compute the "micro" average scores for region_selection_scores
    for subset in region_selection_scores:
        for metric, score in region_selection_scores[subset].items():
            region_selection_scores[subset][metric] = score.compute()[1].item()  # only report results for the positive class (hence [1])

    # compute the "micro" average scores for region_abnormal_scores
    for metric, score in region_abnormal_scores.items():
        region_abnormal_scores[metric] = score.compute()[1].item()

    return val_losses_dict, obj_detector_scores, region_selection_scores, region_abnormal_scores


def evaluate_model(model, train_losses_dict, val_dl, lr_scheduler, optimizer, writer, tokenizer, run_params, is_epoch_end, generated_sentences_folder_path, log):
    # set the model to evaluation mode
    model.eval()

    overall_steps_taken = run_params["overall_steps_taken"]

    # normalize all train losses by steps_taken
    for loss_type in train_losses_dict:
        train_losses_dict[loss_type] /= run_params["steps_taken"]

    (
        val_losses_dict,
        obj_detector_scores,
        region_selection_scores,
        region_abnormal_scores,
    ) = get_val_losses_and_other_metrics(model, val_dl)

    language_model_scores = evaluate_language_model(model, val_dl, tokenizer, writer, overall_steps_taken, generated_sentences_folder_path)

    current_lr = float(optimizer.param_groups[0]["lr"])

    write_all_losses_and_scores_to_tensorboard(
        writer,
        overall_steps_taken,
        train_losses_dict,
        val_losses_dict,
        obj_detector_scores,
        region_selection_scores,
        region_abnormal_scores,
        language_model_scores,
        current_lr,
    )

    train_total_loss = train_losses_dict["total_loss"]
    total_val_loss = val_losses_dict["total_loss"]

    # decrease lr by 1e-1 if total_val_loss has not decreased after certain number of evaluations
    lr_scheduler.step(total_val_loss)

    if total_val_loss < run_params["lowest_val_loss"]:
        run_params["lowest_val_loss"] = total_val_loss
        run_params["best_epoch"] = run_params["epoch"]
        run_params["best_model_save_path"] = os.path.join(
            run_params["weights_folder_path"],
            f"val_loss_{run_params['lowest_val_loss']:.3f}_epoch_{run_params['best_epoch']}.pth",
        )
        run_params["best_model_state"] = deepcopy(model.state_dict())

    if is_epoch_end:
        log_stats_to_console(log, train_total_loss, total_val_loss, run_params["epoch"])
