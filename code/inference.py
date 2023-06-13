import os

import albumentations as A
import numpy as np
import pandas as pd

# torch
import torch
import torch.nn.functional as F
from dataset import XRayInferenceDataset
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

CLASSES = [
    "finger-1",
    "finger-2",
    "finger-3",
    "finger-4",
    "finger-5",
    "finger-6",
    "finger-7",
    "finger-8",
    "finger-9",
    "finger-10",
    "finger-11",
    "finger-12",
    "finger-13",
    "finger-14",
    "finger-15",
    "finger-16",
    "finger-17",
    "finger-18",
    "finger-19",
    "Trapezium",
    "Trapezoid",
    "Capitate",
    "Hamate",
    "Scaphoid",
    "Lunate",
    "Triquetrum",
    "Pisiform",
    "Radius",
    "Ulna",
]

wandb_name = "grid_mask"

SAVED_DIR = "/opt/ml/results_model/" + wandb_name

save_csv_dir = "/opt/ml/results_csv/" + wandb_name

model = torch.load(os.path.join(SAVED_DIR, "best_dice.pt"))

CLASS2IND = {v: i for i, v in enumerate(CLASSES)}

IND2CLASS = {v: k for k, v in CLASS2IND.items()}

if not os.path.isdir(save_csv_dir):
    os.mkdir(save_csv_dir)


# mask map으로 나오는 인퍼런스 결과를 RLE로 인코딩 합니다.
def encode_mask_to_rle(mask):
    """
    mask: numpy array binary mask
    1 - mask
    0 - background
    Returns encoded run length
    """
    pixels = mask.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return " ".join(str(x) for x in runs)


def test(model, data_loader, thr=0.5):
    model = model.cuda()
    model.eval()

    rles = []
    filename_and_class = []
    with torch.no_grad():
        for step, (images, image_names) in tqdm(
            enumerate(data_loader), total=len(data_loader)
        ):
            images = images.cuda()
            outputs = model(images)["out"]

            # restore original size
            outputs = F.interpolate(outputs, size=(2048, 2048), mode="bilinear")
            outputs = torch.sigmoid(outputs)
            outputs = (outputs > thr).detach().cpu().numpy()

            for output, image_name in zip(outputs, image_names):
                for c, segm in enumerate(output):
                    rle = encode_mask_to_rle(segm)
                    rles.append(rle)
                    filename_and_class.append(f"{IND2CLASS[c]}_{image_name}")

    return rles, filename_and_class


tf = A.Resize(512, 512)

test_dataset = XRayInferenceDataset(transforms=tf)

test_loader = DataLoader(
    dataset=test_dataset, batch_size=2, shuffle=False, num_workers=2, drop_last=False
)

rles, filename_and_class = test(model, test_loader)

classes, filename = zip(*[x.split("_") for x in filename_and_class])
image_name = [os.path.basename(f) for f in filename]

df = pd.DataFrame(
    {
        "image_name": image_name,
        "class": classes,
        "rle": rles,
    }
)

df.to_csv(save_csv_dir + "/output.csv", index=False)
