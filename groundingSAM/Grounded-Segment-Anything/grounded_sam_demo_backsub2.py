import argparse
import os
import copy
import glob
from tqdm import tqdm
import numpy as np
import json
import torch
from PIL import Image, ImageDraw, ImageFont

os.environ['CURL_CA_BUNDLE'] = ''

# Grounding DINO
import GroundingDINO.groundingdino.datasets.transforms as T
from GroundingDINO.groundingdino.models import build_model
from GroundingDINO.groundingdino.util import box_ops
from GroundingDINO.groundingdino.util.slconfig import SLConfig
from GroundingDINO.groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap

# segment anything
from segment_anything import build_sam, SamPredictor 
import cv2
import numpy as np
import matplotlib.pyplot as plt


def load_image(image_path):
    # load image
    image_pil = Image.open(image_path).convert("RGB")  # load image

    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image, _ = transform(image_pil, None)  # 3, h, w
    return image_pil, image


def load_model(model_config_path, model_checkpoint_path, device):
    args = SLConfig.fromfile(model_config_path)
    args.device = device
    model = build_model(args)
    checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
    load_res = model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    print(load_res)
    _ = model.eval()
    return model


def get_grounding_output(model, image, caption, box_threshold, text_threshold, with_logits=True, device="cpu"):
    caption = caption.lower()
    caption = caption.strip()
    if not caption.endswith("."):
        caption = caption + "."
    model = model.to(device)
    image = image.to(device)
    with torch.no_grad():
        outputs = model(image[None], captions=[caption])
    logits = outputs["pred_logits"].cpu().sigmoid()[0]  # (nq, 256)
    boxes = outputs["pred_boxes"].cpu()[0]  # (nq, 4)
    logits.shape[0]

    # filter output
    logits_filt = logits.clone()
    boxes_filt = boxes.clone()
    filt_mask = logits_filt.max(dim=1)[0] > box_threshold
    logits_filt = logits_filt[filt_mask]  # num_filt, 256
    boxes_filt = boxes_filt[filt_mask]  # num_filt, 4
    logits_filt.shape[0]

    # get phrase
    tokenlizer = model.tokenizer
    tokenized = tokenlizer(caption)
    # build pred
    pred_phrases = []
    for logit, box in zip(logits_filt, boxes_filt):
        pred_phrase = get_phrases_from_posmap(logit > text_threshold, tokenized, tokenlizer)
        if with_logits:
            pred_phrases.append(pred_phrase + f"({str(logit.max().item())[:4]})")
        else:
            pred_phrases.append(pred_phrase)

    return boxes_filt, pred_phrases

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        # color = np.array([30/255, 144/255, 255/255, 0.6])
        color = np.array([1, 1, 1, 1])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_box(box, ax, label):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2)) 
    ax.text(x0, y0, label)


def save_mask_data(output_dir, mask_list, box_list, label_list):
    value = 0  # 0 for background

    mask_img = torch.zeros(mask_list.shape[-2:])
    for idx, mask in enumerate(mask_list):
        mask_img[mask.cpu().numpy()[0] == True] = value + idx + 1
    plt.figure(figsize=(10, 10))
    plt.imshow(mask_img.numpy())
    plt.axis('off')
    plt.savefig(os.path.join(output_dir, 'mask.jpg'), bbox_inches="tight", dpi=300, pad_inches=0.0)

    json_data = [{
        'value': value,
        'label': 'background'
    }]
    for label, box in zip(label_list, box_list):
        value += 1
        name, logit = label.split('(')
        logit = logit[:-1] # the last is ')'
        json_data.append({
            'value': value,
            'label': name,
            'logit': float(logit),
            'box': box.numpy().tolist(),
        })
    with open(os.path.join(output_dir, 'mask.json'), 'w') as f:
        json.dump(json_data, f)
    

if __name__ == "__main__":

    parser = argparse.ArgumentParser("Grounded-Segment-Anything Demo", add_help=True)
    parser.add_argument("--config", type=str, required=True, help="path to config file")
    parser.add_argument(
        "--grounded_checkpoint", type=str, required=True, help="path to checkpoint file"
    )
    parser.add_argument(
        "--sam_checkpoint", type=str, required=True, help="path to checkpoint file"
    )
    parser.add_argument("--input_image", type=str, required=False, help="path to image file")
    parser.add_argument("--input_image_dir", type=str, default="", required=False, help="path to image files dir")
    parser.add_argument("--text_prompt", type=str, required=True, help="text prompt")
    parser.add_argument(
        "--output_dir", "-o", type=str, default="outputs", required=True, help="output directory"
    )

    parser.add_argument("--box_threshold", type=float, default=0.3, help="box threshold")
    parser.add_argument("--text_threshold", type=float, default=0.25, help="text threshold")
    parser.add_argument('--no-morph', dest='morph', action='store_false')
    parser.set_defaults(morph=True)


    parser.add_argument("--device", type=str, default="cpu", help="running on cpu only!, default=False")
    args = parser.parse_args()

    # cfg
    config_file = args.config  # change the path of the model config file
    grounded_checkpoint = args.grounded_checkpoint  # change the path of the model
    sam_checkpoint = args.sam_checkpoint
    image_path = args.input_image
    image_dir = args.input_image_dir
    text_prompt = args.text_prompt
    out_dir = args.output_dir
    mask_dir = out_dir + '/mask'
    mask_diff_dir = out_dir + '/mask_added'
    box_threshold = args.box_threshold
    text_threshold = args.text_threshold
    device = args.device
    image_paths = []
    if image_dir == "":
        image_paths.append(image_path)
    else:
        image_paths = glob.glob(os.path.join(image_dir, "*"))
        image_paths = [i for i in image_paths if i.endswith('.png')]

    # make dir
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)
    os.makedirs(mask_diff_dir, exist_ok=True)
    model = load_model(config_file, grounded_checkpoint, device=device)
    # initialize SAM
    predictor = SamPredictor(build_sam(checkpoint=sam_checkpoint).to(device))
    print(len(image_paths))
    for image_path in tqdm(image_paths):
        print(f"Processing {image_path} ...")
        # load image
        image_pil, image = load_image(image_path)

        # visualize raw image
        # image_pil.save(os.path.join(output_dir, "raw_image.jpg"))

        import time
        t1 = time.time()

        # run grounding dino model
        boxes_filt, pred_phrases = get_grounding_output(
            model, image, text_prompt, box_threshold, text_threshold, device=device
        )

        image = cv2.imread(image_path)
        og_image = image.copy()
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        predictor.set_image(image)

        size = image_pil.size
        H, W = size[1], size[0]
        for i in range(boxes_filt.size(0)):
            boxes_filt[i] = boxes_filt[i] * torch.Tensor([W, H, W, H])
            boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
            boxes_filt[i][2:] += boxes_filt[i][:2]

        boxes_filt = boxes_filt.cpu()

        transformed_boxes = predictor.transform.apply_boxes_torch(boxes_filt[0], image.shape[:2]).to(device)
        print(predictor.model.mask_threshold)
        masks, _, _ = predictor.predict_torch(
            point_coords = None,
            point_labels = None,
            boxes = transformed_boxes.to(device),
            multimask_output = False,
        )

        t2 = time.time()
        print(f"total time: {t2 - t1}")

        mask_old = masks[0].cpu().numpy().squeeze()
        mask = mask_old.copy()
        if args.morph:
            # add 200 pixel padding on mask_old
            mask_old = np.pad(mask_old, 200, mode='constant', constant_values=0)
            kernel = np.ones((5,5),np.uint8)
            mask = cv2.morphologyEx(mask_old.astype(float), cv2.MORPH_CLOSE, kernel, iterations=20)
            # remove 200 pixels from boundary
            mask = mask[200:-200, 200:-200]
            mask_old = mask_old[200:-200, 200:-200]

        mask_added =  mask * (1 - mask_old)
        mask = mask.astype(bool)

        new_img = np.zeros_like(image)
        new_img[mask] = og_image[mask]
        
        part_image = new_img.copy()
        part_image[mask == 0] = 255
        new_img = cv2.merge([*cv2.split(part_image), mask.astype(np.uint8) * 255], 4)

        out_path = os.path.join(out_dir, image_path.split('/')[-1])
        mask_out_path = os.path.join(mask_dir, image_path.split('/')[-1])
        mask_diff_path = os.path.join(mask_diff_dir, image_path.split('/')[-1])
        print(out_path)
        print(mask_out_path)
        cv2.imwrite(out_path, new_img)
        cv2.imwrite(mask_out_path, mask.astype(np.uint8) * 255)
        cv2.imwrite(mask_diff_path, mask_added.astype(np.uint8) * 255)

        # # draw output image
        # plt.figure(figsize=(10, 10))
        # plt.imshow(image)
        # for mask in masks:
        #     show_mask(mask.cpu().numpy(), plt.gca(), random_color=True)
        # for box, label in zip(boxes_filt, pred_phrases):
        #     show_box(box.numpy(), plt.gca(), label)

        # plt.axis('off')
        # plt.savefig(
        #     os.path.join(output_dir, "grounded_sam_output.jpg"), 
        #     bbox_inches="tight", dpi=300, pad_inches=0.0
        # )

        # save_mask_data(output_dir, masks, boxes_filt, pred_phrases)

