# python native
import datetime

# 모듈 가져오기
import importlib
import os
import random

# parsar
from argparse import ArgumentParser

# external library
import numpy as np

# torch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import wandb

# dataset
from dataset import XRayDataset
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

# augmenation
from transforms import base_pixeldropout_augmentation, baseaugmentation

# loss
# from loss import FocalLoss, DiceLoss


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--num_class", type=int, default=29)
    parser.add_argument("--val_every", type=int, default=1)
    parser.add_argument("--saved_dir", type=str, default="/opt/ml/results_model/")
    parser.add_argument("--wandb_project", type=str, default="project")
    parser.add_argument("--wandb_name", type=str, default="name")
    parser.add_argument("--models", type=str, default="FCN_RESNET50")
    parser.add_argument("--is_smp", type=int, default=0)
    args = parser.parse_args()

    return args


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


def dice_coef(y_true, y_pred):
    y_true_f = y_true.flatten(2)
    y_pred_f = y_pred.flatten(2)
    intersection = torch.sum(y_true_f * y_pred_f, -1)

    eps = 0.0001
    return (2.0 * intersection + eps) / (
        torch.sum(y_true_f, -1) + torch.sum(y_pred_f, -1) + eps
    )


def save_model(model, saved_dir, file_name="best_dice.pt"):
    output_path = os.path.join(saved_dir, file_name)
    torch.save(model, output_path)


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def validation(epoch, model, is_smp, data_loader, criterion, thr=0.5):
    print(f"Start validation #{epoch:2d}")
    model.eval()

    dices = []
    with torch.no_grad():
        total_loss = 0
        cnt = 0

        for step, (images, masks) in tqdm(
            enumerate(data_loader), total=len(data_loader)
        ):
            images, masks = images.cuda(), masks.cuda()
            model = model.cuda()

            if is_smp:
                outputs = model(images)
            else:
                outputs = model(images)["out"]

            output_h, output_w = outputs.size(-2), outputs.size(-1)
            mask_h, mask_w = masks.size(-2), masks.size(-1)

            # restore original size
            if output_h != mask_h or output_w != mask_w:
                outputs = F.interpolate(outputs, size=(mask_h, mask_w), mode="bilinear")

            loss = criterion(outputs, masks)
            total_loss += loss
            cnt += 1

            outputs = torch.sigmoid(outputs)
            outputs = (outputs > thr).detach().cpu()
            masks = masks.detach().cpu()

            dice = dice_coef(outputs, masks)
            dices.append(dice)

    dices = torch.cat(dices, 0)
    dices_per_class = torch.mean(dices, 0)
    dice_str = [f"{c:<12}: {d.item():.4f}" for c, d in zip(CLASSES, dices_per_class)]
    dice_str = "\n".join(dice_str)
    print(dice_str)

    avg_dice = torch.mean(dices_per_class).item()

    return avg_dice


def train(
    seed,
    wandb_project,
    wandb_name,
    batch_size,
    models,
    lr,
    epochs,
    is_smp,
    num_class,
    saved_dir,
    val_every,
):
    # seed
    set_seed(seed)

    # wandb
    wandb.init(project=wandb_project, entity="boostcamp-cv5-dc", name=wandb_name)
    wandb.config.update(args)

    # augmenation
    train_tf = base_pixeldropout_augmentation()
    valid_tf = baseaugmentation()

    # dataset
    train_dataset = XRayDataset(is_train=True, transforms=train_tf)
    valid_dataset = XRayDataset(is_train=False, transforms=valid_tf)

    # dataloader
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        drop_last=True,
        # pin_memory =True
    )

    valid_loader = DataLoader(
        dataset=valid_dataset,
        batch_size=2,
        shuffle=False,
        num_workers=1,
        drop_last=False,
        # pin_memory =True
    )

    # init metric score
    best_dice = 0.0

    # 모듈 가져오기

    # 속성 가져오기
    model_module = getattr(importlib.import_module("my_model"), models)
    model = model_module()

    # # model = getattr(model)
    # if model=='fcn_resnet50':
    #     model = FCN_RESNET50()
    # if model=='fcn_resnet101':
    #     model = FCN_RESNET101()

    # Loss function
    criterion = nn.BCEWithLogitsLoss()

    # Optimizer
    optimizer = optim.Adam(params=model.parameters(), lr=lr, weight_decay=1e-6)

    print("Start training..")
    # train
    for epoch in range(epochs):
        model.train()
        for step, (images, masks) in enumerate(train_loader):
            # gpu 연산을 위해 device 할당
            images, masks = images.cuda(), masks.cuda()
            model = model.cuda()

            if is_smp:
                outputs = model(images)
            else:
                outputs = model(images)["out"]

            # loss 계산
            loss = criterion(outputs, masks)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # step 주기에 따른 loss 출력
            if (step + 1) % 25 == 0:
                print(
                    f'{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")} | '
                    f"Epoch [{epoch+1}/{epoch}], "
                    f"Step [{step+1}/{len(train_loader)}], "
                    f"Loss: {round(loss.item(),4)}"
                )

            wandb.log({"train_Loss": round(loss.item(), 4)})
        # validation 주기에 따른 loss 출력 및 best model 저장
        if (epoch + 1) % val_every == 0:
            dice = validation(epoch + 1, model, is_smp, valid_loader, criterion)
            wandb.log({"Dice": dice})
            if not os.path.exists(saved_dir + wandb_name + "/"):
                os.makedirs(saved_dir + wandb_name + "/")

            if best_dice < dice:
                print(
                    f"Best performance at epoch: {epoch + 1}, {best_dice:.4f} -> {dice:.4f}"
                )
                print(f"Save model in {saved_dir}")
                best_dice = dice
                save_model(model, saved_dir + wandb_name + "/")


def main(args):
    train(**args.__dict__)


if __name__ == "__main__":
    args = parse_args()
    main(args)
