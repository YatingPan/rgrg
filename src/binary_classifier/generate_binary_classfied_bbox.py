import torch
import torch.nn as nn
import cv2
import os
import matplotlib.pyplot as plt
import csv
import albumentations as A
from albumentations.pytorch import ToTensorV2
from src.object_detector.object_detector import ObjectDetector
from src.binary_classifier.binary_classifier_region_abnormal import BinaryClassifierRegionAbnormal
from src.binary_classifier.binary_classifier_region_selection import BinaryClassifierRegionSelection
from src.dataset.constants import ANATOMICAL_REGIONS

"""
Detection Output:
-loss: empty dict when in evaluation
-top_region_boxes with a list of lists for at most 29 bboxes, [ 62.1847,  37.5724, 246.2823, 357.1617] is one box coordinates
-top_region_scores with a list of scores for at most 29 bboxes, 0.9993 is one score
-top_region_tensors with a tensor of shape (1, 29, 1024) for the feature vectors of the detected regions
-a boolean list which is (always) in 29 length, indicating whether each anatomical region is detected

{}, {'top_region_boxes': tensor([[[ 84.3232,  88.6287, 246.2666, 441.0349],
         [106.9789,  99.2323, 244.3961, 211.3714],
         [ 90.6608, 212.5578, 240.2658, 300.7414],
         [ 86.4279, 297.9957, 242.1113, 438.2645],
         [164.3338, 193.1795, 241.7714, 314.6854],
         [128.8230,  88.6495, 243.8136, 162.6473],
         [166.4985, 135.0031, 314.1602, 401.2585],
         [ 84.7913, 366.4947, 279.1947, 440.1894],
         [269.0054,  88.9714, 441.7512, 436.3864],
         [271.0857,  97.5679, 410.2545, 217.2875],
         [277.2947, 215.7286, 429.9984, 297.4494],
         [285.8763, 295.1308, 442.9730, 433.7875],
         [273.6597, 196.6490, 353.2260, 308.6743],
         [273.0342,  89.8908, 384.4020, 163.7727],
         [235.9407, 140.7058, 367.1772, 458.3211],
         [279.2821, 371.1555, 441.2620, 431.5907],
         [213.6422,  78.8229, 288.8950, 269.1517],
         [233.2762,  38.9306, 296.9648, 490.3981],
         [ 43.1253, 123.4607, 239.0140, 158.3665],
         [275.4500, 121.6743, 462.4796, 158.3427],
         [259.1046, 175.6498, 292.8362, 212.0878],
         [212.6673, 130.2854, 362.3648, 392.1364],
         [224.7701, 141.0317, 317.9717, 258.0857],
         [222.2860, 176.2373, 260.6783, 260.1266],
         [212.7280, 254.8286, 363.1716, 389.1239],
         [210.9230, 256.0848, 262.2525, 298.5945],
         [209.8049, 299.0849, 263.1266, 383.3849],
         [226.2536, 207.0565, 346.4468, 359.7882],
         [ 94.5071, 377.6455, 441.9211, 492.8282]]], device='cuda:0'), 'top_scores': tensor([[0.9991, 0.9994, 0.9954, 0.9969, 0.9966, 0.9982, 0.0000, 0.9965, 0.9990,
         0.9993, 0.9823, 0.9950, 0.9865, 0.9986, 0.0000, 0.9965, 0.9980, 0.9952,
         0.9789, 0.9919, 0.9817, 0.9980, 0.9965, 0.9672, 0.9963, 0.9638, 0.9817,
         0.0000, 0.9979]], device='cuda:0')}, tensor([[[ 0.1847, -0.0856,  0.0071,  ...,  0.0146, -0.1496, -0.1539],
         [ 0.1291, -0.0175, -0.0643,  ...,  0.0582, -0.0683,  0.0739],
         [ 0.3225, -0.2175, -0.0239,  ...,  0.0693, -0.2023, -0.0393],
         ...,
         [ 0.0980, -0.0648,  0.0819,  ...,  0.0094, -0.5081, -0.0258],
         [ 0.2268, -0.1105, -0.1142,  ...,  0.0111, -0.2298, -0.1269],
         [-0.0335,  0.0936,  0.0172,  ...,  0.0361, -0.0635, -0.3579]]],
       device='cuda:0'), tensor([[ True,  True,  True,  True,  True,  True, False,  True,  True,  True,
          True,  True,  True,  True, False,  True,  True,  True,  True,  True,
          True,  True,  True,  True,  True,  True,  True, False,  True]],
       device='cuda:0'))
"""

class UnifiedModel(nn.Module):
    def __init__(self):
        super(UnifiedModel, self).__init__()
        # set self.return_feature_vectors to True to return the feature vectors
        self.object_detector = ObjectDetector(return_feature_vectors=True) # to return the top_region_tensor to binary classifers
        self.binary_classifier_region_selection = BinaryClassifierRegionSelection()
        self.binary_classifier_region_abnormal = BinaryClassifierRegionAbnormal()
        self.training = False

    def forward(self, image):
        # Object detection
        detection_output = self.object_detector(image)
        print (detection_output)
        losses, detections, top_region_features, class_detected = detection_output
        if not self.training:
            with torch.no_grad():
                predicted_abnormal_regions = self.binary_classifier_region_abnormal(top_region_features, class_detected, region_is_abnormal=None)
                print (predicted_abnormal_regions)
                predicted_selection_regions = self.binary_classifier_region_selection(top_region_features, class_detected, return_loss=False)
                print (predicted_selection_regions)
        else:
            predicted_abnormal_regions = None
            predicted_selection_regions = None
        
        return detections, class_detected, predicted_abnormal_regions, predicted_selection_regions


def load_model(checkpoint_path, device):

    model = UnifiedModel()
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_state_dict = checkpoint["model"]
    filtered_model_state_dict = {k: v for k, v in model_state_dict.items() if "language_model" not in k}
    model.load_state_dict(filtered_model_state_dict, strict=False)  # Using strict=False to ignore non-matching keys
    model.to(device).eval()
    
    return model

def get_inference_transforms():
    mean = 0.471
    std = 0.302
    IMAGE_INPUT_SIZE = 512

    return A.Compose([
        A.LongestMaxSize(max_size=IMAGE_INPUT_SIZE, interpolation=cv2.INTER_AREA),
        A.PadIfNeeded(min_height=IMAGE_INPUT_SIZE, min_width=IMAGE_INPUT_SIZE, border_mode=cv2.BORDER_CONSTANT),
        A.Normalize(mean=mean, std=std),
        ToTensorV2()
    ])

def process_image(image_path, transform):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Unable to read the image at path: {image_path}")
    
    transformed = transform(image=image)
    transformed_image = transformed["image"]
    resized_image = cv2.resize(image, (512, 512))

    return transformed_image.unsqueeze(0), resized_image

def plot_box(box, ax, clr, linestyle, class_label):
    x0, y0, x1, y1 = box
    ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor=clr, linewidth=2, linestyle=linestyle))
    # Annotation for class label
    ax.text(x0, y0, class_label, bbox=dict(facecolor=clr, alpha=0.5), fontsize=8, color='white')

def plot_and_save_bboxes(image, pred_boxes, class_detected, output_path, region_sets):
    for i, region_set in enumerate(region_sets):
        fig, ax = plt.subplots(1, figsize=(8, 8))
        ax.imshow(image, cmap='gray')
        ax.axis('off')

        region_colors = ["b", "g", "r", "c", "m", "y"]
        if len(region_set) < len(region_colors):
            region_colors = region_colors[:len(region_set)]

        for region, color in zip(region_set, region_colors):
            if region in ANATOMICAL_REGIONS:
                region_index = ANATOMICAL_REGIONS[region]
                box = pred_boxes[region_index]
                is_detected = class_detected[region_index]
                plot_box(box, ax, clr=color, linestyle='dashed', class_detected=is_detected)

        plt.title(f"Region Set {i + 1}")
        plt.savefig(os.path.join(output_path, f"bbox_set_{i + 1}.png"), bbox_inches='tight')
        plt.close()

def draw_bboxes_on_resized_image(resized_image, pred_boxes, class_detected, output_path, region_set):
    # Convert the grayscale image to color format for drawing colored bboxes
    color_resized_image = cv2.cvtColor(resized_image, cv2.COLOR_GRAY2BGR)

    # Define colors in BGR format for drawing colored bboxes
    region_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]

    for idx, region in enumerate(region_set):
        if region in ANATOMICAL_REGIONS:
            region_index = ANATOMICAL_REGIONS[region]
            box = pred_boxes[region_index]
            color = region_colors[idx % len(region_colors)]

            if class_detected[region_index]:
                cv2.rectangle(color_resized_image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color, 2)
                cv2.putText(color_resized_image, region, (int(box[0]), int(box[1] - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    cv2.imwrite(output_path, color_resized_image)

def draw_all_bboxes_on_image(image, pred_boxes, class_detected, output_path):
    fig, ax = plt.subplots(1, figsize=(8, 8))
    ax.imshow(image, cmap='gray')
    ax.axis('off')

    region_colors = ["b", "g", "r", "c", "m", "y"]
    for region in ANATOMICAL_REGIONS:
        idx = ANATOMICAL_REGIONS[region]
        color = region_colors[idx % len(region_colors)]
        box = pred_boxes[idx]
        is_detected = class_detected[idx]
        
        # Updated to use region name instead of boolean
        plot_box(box, ax, clr=color, linestyle='dashed', class_label=region if is_detected else "Not detected")

    plt.savefig(output_path, bbox_inches='tight')
    plt.close()

def draw_abnormal_bboxes(image, pred_boxes, abnormal_regions, output_path, color_normal="blue", color_abnormal="red"):
    fig, ax = plt.subplots(1, figsize=(8, 8))
    ax.imshow(image, cmap='gray')
    ax.axis('off')

    print("abnormal regions: ", abnormal_regions)

    abnormal_bboxes = []
    abnormal_classes = []

    for region, abnormal in zip(ANATOMICAL_REGIONS, abnormal_regions):
        idx = ANATOMICAL_REGIONS[region]
        if abnormal_regions[idx]:
            print (f"Abnormal region: {region}")
            color = color_abnormal
            abnormal_bboxes.append(pred_boxes[idx])
            abnormal_classes.append(region)

    # draw abnormal bboxes with region names
    for box, region in zip(abnormal_bboxes, abnormal_classes):
        plot_box(box, ax, clr=color, linestyle='dashed', class_label=region)

    plt.savefig(output_path, bbox_inches='tight')
    plt.close()

def draw_important_bboxes(image, pred_boxes, important_regions, output_path, color_normal="blue", color_important="yellow"):
    fig, ax = plt.subplots(1, figsize=(8, 8))
    ax.imshow(image, cmap='gray')
    ax.axis('off')

    print (f"important regions:" , important_regions)
    # here the output important regions is not [], but [[T, F....]]

    important_regions = important_regions[0]
    important_bboxes = []
    important_classes = []

    for region, important in zip(ANATOMICAL_REGIONS, important_regions):
        idx = ANATOMICAL_REGIONS[region]
        if important_regions[idx]:
            print (f"Important region: {region}")
            color = color_important
            important_bboxes.append(pred_boxes[idx])
            important_classes.append(region)

    # draw important bboxes with region names
    for box, region in zip(important_bboxes, important_classes):
        plot_box(box, ax, clr=color, linestyle='dashed', class_label=region)

    plt.savefig(output_path, bbox_inches='tight')
    plt.close()

def draw_classified_bboxes(image, pred_boxes, abnormal_regions, important_regions, output_path, color_normal="blue", color_abnormal="red", color_important="yellow"):
    fig, ax = plt.subplots(1, figsize=(8, 8))
    ax.imshow(image, cmap='gray')
    ax.axis('off')

    important_regions = important_regions[0]

    # this is the combination of the two previous functions
    for region, abnormal, important in zip(ANATOMICAL_REGIONS, abnormal_regions, important_regions):
        idx = ANATOMICAL_REGIONS[region]
        if abnormal_regions[idx]:
            color = color_abnormal
        elif important_regions[idx]:
            color = color_important
        else:
            color = color_normal

        box = pred_boxes[idx]
        plot_box(box, ax, clr=color, linestyle='dashed', class_label=region)

    plt.savefig(output_path, bbox_inches='tight')
    plt.close()

def main(model_path, image_folder, output_folder, device):
    transform = get_inference_transforms()
    model = load_model(model_path, device)

    os.makedirs(output_folder, exist_ok=True)

    # Prepare to collect CSV data for all images
    csv_header = ["Image File", "Detected Class"] + ["Box " + str(i) for i in range(1, 30)] + ["Abnormal Region", "Important Region"]
    csv_data = [csv_header]

    csv_data = []

    for image_file in os.listdir(image_folder):
        image_path = os.path.join(image_folder, image_file)
        image_tensor, resized_image = process_image(image_path, transform)
        image_tensor = image_tensor.to(device)

        with torch.no_grad():
            outputs = model(image_tensor)
            print(f"Outputs: {outputs}")
            pred_boxes = outputs[0]["top_region_boxes"][0].cpu().numpy()
            class_detected = outputs[1][0].cpu().numpy()
            abnormal_regions = outputs[2][0].cpu().numpy()
            important_regions = outputs[3][0].cpu().numpy()

            base_file_name = os.path.splitext(image_file)[0]
            output_image_path_all_bboxes = os.path.join(output_folder, f"{base_file_name}_all_bboxes.png")
            output_image_path_abnormal_bboxes = os.path.join(output_folder, f"{base_file_name}_abnormal_bboxes.png")
            output_image_path_important_bboxes = os.path.join(output_folder, f"{base_file_name}_important_bboxes.png")
            output_image_path_classified_bboxes = os.path.join(output_folder, f"{base_file_name}_classified_bboxes.png")

            draw_all_bboxes_on_image(resized_image, pred_boxes, class_detected, output_image_path_all_bboxes)
            draw_abnormal_bboxes(resized_image, pred_boxes, abnormal_regions, output_image_path_abnormal_bboxes)
            draw_important_bboxes(resized_image, pred_boxes, important_regions, output_image_path_important_bboxes)
            draw_classified_bboxes(resized_image, pred_boxes, abnormal_regions, important_regions, output_image_path_classified_bboxes)
            print(f"Processed and saved output for image: {image_file}")

            row_data = [image_file] + list(class_detected) + [str(list(pred_boxes)), str(list(abnormal_regions)), str(list(important_regions))]
            csv_data.append(row_data)
        
    # Write CSV data to file
    output_csv_path = os.path.join(output_folder, "detection_results.csv")
    with open(output_csv_path, "w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerows(csv_data)
    print("Detection results saved to CSV file.")

if __name__ == "__main__":
    checkpoint_path = "/home/user/yatpan/yatpan/rgrg/runs/binary_classifers/obj_detect_binary_classifiers_checkpoint_val_loss_106.983_overall_steps_40821.pt"
    image_folder = "/home/user/yatpan/yatpan/rgrg/datasets/mimic-cxr-jpg/files/p10/p10000032/s50414267/"
    output_folder = "/home/user/yatpan/yatpan/rgrg/datasets/output_images_binary_bbox"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    main(checkpoint_path, image_folder, output_folder, device)
