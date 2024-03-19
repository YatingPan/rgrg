import torch
import cv2
import os
from albumentations.pytorch import ToTensorV2
import albumentations as A
from src.object_detector.object_detector import ObjectDetector
import matplotlib.pyplot as plt
import csv
from torchvision.transforms import Compose, ToTensor, Normalize
from src.dataset.constants import ANATOMICAL_REGIONS
"""
The output of forward function in object_detector.py includes:
top_region_boxes with a list of lists for at most 29 bboxes, [ 62.1847,  37.5724, 246.2823, 357.1617] is one box coordinates
top_region_scores with a list of scores for at most 29 bboxes, 0.9993 is one score
a boolean list which is (always) in 29 length, indicating whether each anatomical region is detected

Output example:
{'top_region_boxes': 
tensor([[[ 62.1847,  37.5724, 246.2823, 357.1617],
         [ 94.9973,  47.4827, 243.8695, 166.4112],
         [ 85.5067, 162.3012, 237.3160, 234.2783],
         [ 63.9153, 232.5567, 228.6178, 357.1140],
         [159.6048, 147.3925, 240.1432, 246.7487],
         [108.5755,  36.3669, 246.1896, 116.7793],
         [ 41.8676, 325.4930,  87.3756, 370.8314],
         [ 64.4204, 304.1375, 260.7064, 358.9724],
         [272.1625,  37.7189, 445.2013, 369.3154],
         [276.6585,  48.2947, 426.9522, 164.7577],
         [277.0860, 163.4965, 429.9567, 239.4509],
         [276.5676, 238.5851, 442.3969, 371.3423],
         [275.7169, 147.1160, 353.1674, 253.0452],
         [277.7853,  35.3258, 414.9539, 118.3021],
         [225.3808,   1.7663, 288.2535, 415.2270],
         [276.1733, 315.9911, 441.5049, 373.5011],
         [204.3724,  26.5507, 292.5441, 227.8762],
         [215.9922,   6.0345, 291.0137, 477.0546],
         [ 52.2446,  59.4918, 236.4050, 116.7713],
         [287.1554,  59.1837, 477.0437, 114.1213],
         [262.0039, 127.2791, 301.1245, 164.1391],
         [197.5078,  83.4909, 366.3718, 343.3380],
         [223.5683,  90.1908, 309.1987, 212.3859],
         [207.8217,  37.7248, 282.2265, 400.6718],
         [198.3218, 210.2058, 365.8741, 343.5663],
         [198.8682, 216.2078, 249.4941, 257.1916],
         [198.2528, 251.4716, 253.4841, 336.4872],
         [214.9772,  52.5690, 283.4844, 474.5016],
         [ 65.9281, 308.1884, 447.7742, 482.7435]]]), 
'top_scores': tensor([[0.9993, 0.9996, 0.9971, 0.9942, 0.9991, 0.9979, 0.8709, 0.9937, 0.9991,
         0.9990, 0.9943, 0.9868, 0.9971, 0.9984, 0.0000, 0.9964, 0.9996, 0.9978,
         0.9936, 0.9990, 0.9995, 0.9988, 0.9998, 0.0000, 0.9996, 0.9909, 0.9985,
         0.0000, 0.9789]])}, 
tensor([[ True,  True,  True,  True,  True,  True,  True,  True,  True,  True,
          True,  True,  True,  True, False,  True,  True,  True,  True,  True,
          True,  True,  True, False,  True,  True,  True, False,  True]]))

We store the output into a csv file, with each row:
image_id, image_path, right lung box, right lung score, right upper lung zone box, right upper lung zone score, ..., abdomen box, abdomen score

Main steps:
1. preprocess the input images, resize to 512×512 and gray scale it
2. run object detector to get the outputs
3. store the outputs into csv file
4. draw bbox on the resized images
5. output the resized images with bbox

We use the same strategies in RGRG's object detector training script: generate 6 bbox per time in different colors to avoid overlap

"""

def load_model(model_path, device):
    model = ObjectDetector(return_feature_vectors=False)
    model.load_state_dict(torch.load(model_path, map_location=device))
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

    # Resize and transform the image
    transformed = transform(image=image)
    transformed_image = transformed["image"]

    # Resize the original image for drawing bounding boxes later
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


def main(model_path, image_folder, output_folder, device):
    transform = get_inference_transforms()
    model = load_model(model_path, device)

    # Define region sets as in the training script
    region_sets = [
        ["right lung", "right costophrenic angle", "left lung", "left costophrenic angle", "cardiac silhouette", "spine"],
        ["right upper lung zone", "right mid lung zone", "right lower lung zone", "left upper lung zone", "left mid lung zone", "left lower lung zone"],
        ["right hilar structures", "right apical zone", "left hilar structures", "left apical zone", "right hemidiaphragm", "left hemidiaphragm"],
        ["trachea", "right clavicle", "left clavicle", "aortic arch", "abdomen", "right atrium"],
        ["mediastinum", "svc", "cavoatrial junction", "carina", "upper mediastinum"]
    ]

    csv_data = []

    for image_file in os.listdir(image_folder):
        image_path = os.path.join(image_folder, image_file)
        image_tensor, resized_image = process_image(image_path, transform)
        image_tensor = image_tensor.to(device)

        with torch.no_grad():
            outputs = model(image_tensor)
            pred_boxes = outputs[1]["top_region_boxes"][0].cpu().numpy()
            class_detected = outputs[2][0].cpu().numpy()

        output_image_path = os.path.join(output_folder, f"bbox_{image_file}")
        draw_all_bboxes_on_image(resized_image, pred_boxes, class_detected, output_image_path)
        print(f"Processed and saved all bboxes for the image to {output_image_path}")

        # Iterate through each region set
        for i, region_set in enumerate(region_sets):
            output_image_path = os.path.join(output_folder, f"bbox_set_{i+1}_{image_file}")
            draw_bboxes_on_resized_image(resized_image, pred_boxes, class_detected, output_image_path, region_set)
            print(f"Processed and saved bboxes for different region sets to {output_image_path}")

        row_data = [image_file, image_path]
        for box, score in zip(pred_boxes, outputs[1]["top_scores"][0].cpu().numpy()):
            row_data.extend([' '.join(map(str, box)), score])
        csv_data.append(row_data)

    csv_output_path = os.path.join(output_folder, "output_data.csv")
    write_to_csv(csv_data, csv_output_path)
    print(f"CSV data saved to {csv_output_path}")

def write_to_csv(data, output_csv_path):
    with open(output_csv_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['image_id', 'image_path', *['{}_box'.format(region) for region in ANATOMICAL_REGIONS], *['{}_score'.format(region) for region in ANATOMICAL_REGIONS]])
        for row in data:
            writer.writerow(row)

if __name__ == "__main__":
    model_path = "/home/user/yatpan/yatpan/rgrg/runs/object_detector/object_detector_weights_val_loss_13.042_epoch_8.pth"
    image_folder = "/home/user/yatpan/yatpan/rgrg/datasets/mimic-cxr-jpg/files/p10/p10000032/s50414267/"
    output_folder = "/home/user/yatpan/yatpan/rgrg/datasets/output_images_bbox"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    main(model_path, image_folder, output_folder, device)
