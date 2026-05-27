"""Capstone 1 - Part 1: Vehicle object detection with PyTorch / Faster R-CNN."""
from common import md, code, write_notebook

cells = [
    md("""
# Capstone 1 - Part 1: Vehicle Object Detection

**Business scenario.** Tesla and other AV programs need real-time detection of
vehicles, pedestrians, and cyclists. We will build an object detector that,
given an image, predicts:

* **what** vehicles/agents are present (11 classes), and
* **where** they are (rectangular bounding boxes).

**Approach.** We fine-tune **Faster R-CNN with a ResNet-50 FPN backbone**
pre-trained on COCO. Faster R-CNN is a solid, well-understood two-stage
architecture; using a pre-trained backbone lets us learn from this dataset
quickly while still beating from-scratch CNNs by a wide margin.

> **Environment**: requires `torch >= 2.0` and `torchvision`. Heavy training
> needs a GPU. Run on **Google Colab (free T4 GPU)** if your local CPU lacks
> AVX or you don't have CUDA.

Datasets:

* `Capstone 1/Part 1/Images.zip` - 5,626 vehicle scene images
* `Capstone 1/Part 1/labels.csv` - per-object rows
  `image_id, class, xmin, ymin, xmax, ymax`
"""),
    md("## 1. Setup"),
    code("""
import os, zipfile, random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torchvision.transforms.functional as TF

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("torch:", torch.__version__, " device:", DEVICE)
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
"""),
    md("## 2. Create the working folder structure & unzip"),
    code("""
ZIP_PATH    = r"../Capstone 1/Part 1/Images.zip"
LABELS_PATH = r"../Capstone 1/Part 1/labels.csv"
WORK_DIR    = "./_vehicles"
IMG_DIR     = os.path.join(WORK_DIR, "Images")

os.makedirs(WORK_DIR, exist_ok=True)
if not os.path.isdir(IMG_DIR):
    print("Unzipping (~140MB) ...")
    with zipfile.ZipFile(ZIP_PATH) as z:
        z.extractall(WORK_DIR)
print("# images on disk:", len(os.listdir(IMG_DIR)))
"""),
    md("## 3. Read & explore the labels"),
    code("""
labels = pd.read_csv(LABELS_PATH, header=None,
                     names=["image_id", "class", "xmin", "ymin", "xmax", "ymax"])
labels["image_id"] = labels["image_id"].astype(int).astype(str).str.zfill(8)

# keep only labels for images we actually have
files_on_disk = set(f.replace(".jpg", "") for f in os.listdir(IMG_DIR))
labels = labels[labels["image_id"].isin(files_on_disk)].reset_index(drop=True)

print("Labels for available images:", labels.shape)
print(labels["class"].value_counts())
"""),
    code("""
CLASS_NAMES = ["__background__"] + sorted(labels["class"].unique())
class_to_idx = {c: i for i, c in enumerate(CLASS_NAMES)}
idx_to_class = {i: c for c, i in class_to_idx.items()}
print(class_to_idx)
NUM_CLASSES = len(CLASS_NAMES)
"""),
    md("### 3.1 Sanity-check by visualising a labelled image"),
    code("""
def show_with_boxes(image_id):
    img = Image.open(os.path.join(IMG_DIR, image_id + ".jpg")).convert("RGB")
    rows = labels[labels["image_id"] == image_id]
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.imshow(img)
    for _, r in rows.iterrows():
        rect = patches.Rectangle((r.xmin, r.ymin), r.xmax-r.xmin, r.ymax-r.ymin,
                                 linewidth=2, edgecolor="red", facecolor="none")
        ax.add_patch(rect)
        ax.text(r.xmin, r.ymin-4, r["class"], color="white",
                bbox=dict(facecolor="red", pad=1, alpha=0.8), fontsize=8)
    ax.set_title(f"image {image_id}"); ax.axis("off"); plt.show()


for img_id in random.sample(sorted(files_on_disk), 3):
    show_with_boxes(img_id)
"""),
    md("## 4. Build a torch `Dataset` for object detection"),
    code("""
# Group annotations by image
groups = labels.groupby("image_id")
image_ids = sorted(groups.groups.keys())


class VehicleDataset(Dataset):
    def __init__(self, image_ids, img_dir):
        self.image_ids = image_ids
        self.img_dir   = img_dir

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        img = Image.open(os.path.join(self.img_dir, img_id + ".jpg")).convert("RGB")
        img = TF.to_tensor(img)

        rows = groups.get_group(img_id)
        boxes  = rows[["xmin","ymin","xmax","ymax"]].values.astype(np.float32)
        labels_ = np.array([class_to_idx[c] for c in rows["class"]], dtype=np.int64)

        # filter degenerate boxes (xmax<=xmin or ymax<=ymin)
        keep = (boxes[:,2] > boxes[:,0]) & (boxes[:,3] > boxes[:,1])
        boxes = boxes[keep]; labels_ = labels_[keep]

        target = {
            "boxes":    torch.as_tensor(boxes,   dtype=torch.float32),
            "labels":   torch.as_tensor(labels_, dtype=torch.int64),
            "image_id": torch.tensor([idx]),
            "area":     torch.as_tensor((boxes[:,2]-boxes[:,0])*(boxes[:,3]-boxes[:,1]),
                                        dtype=torch.float32),
            "iscrowd":  torch.zeros((len(boxes),), dtype=torch.int64),
        }
        return img, target


def collate_fn(batch):
    return tuple(zip(*batch))
"""),
    md("### 4.1 Train / validation split"),
    code("""
random.shuffle(image_ids)

# Use a small, manageable subset so training fits on a single GPU/CPU.
SUBSET = 1500          # raise to len(image_ids) when you have a beefy GPU
ids = image_ids[:SUBSET]

n_val = int(0.1 * len(ids))
train_ids = ids[n_val:]
val_ids   = ids[:n_val]

train_ds = VehicleDataset(train_ids, IMG_DIR)
val_ds   = VehicleDataset(val_ids,   IMG_DIR)
print("train:", len(train_ds), " val:", len(val_ds))
"""),
    code("""
BATCH = 4   # Faster R-CNN images are large; lower this on small GPUs
train_dl = DataLoader(train_ds, batch_size=BATCH, shuffle=True,
                      num_workers=0, collate_fn=collate_fn)
val_dl   = DataLoader(val_ds,   batch_size=BATCH, shuffle=False,
                      num_workers=0, collate_fn=collate_fn)
"""),
    md("## 5. Build the model - Faster R-CNN ResNet-50 FPN"),
    code("""
def build_model(num_classes: int) -> nn.Module:
    weights = FasterRCNN_ResNet50_FPN_Weights.COCO_V1
    model = fasterrcnn_resnet50_fpn(weights=weights)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # Replace the COCO classification head with our own
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


model = build_model(NUM_CLASSES).to(DEVICE)
print(f"Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
"""),
    md("## 6. Train"),
    code("""
EPOCHS = 3   # bump to 8-10 on a real GPU
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=5e-3, momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)


def train_one_epoch(model, dl, optim, device):
    model.train()
    running = 0.0
    for i, (imgs, targets) in enumerate(dl):
        imgs    = [img.to(device) for img in imgs]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        loss_dict = model(imgs, targets)
        loss = sum(loss_dict.values())

        optim.zero_grad()
        loss.backward()
        optim.step()

        running += loss.item()
        if (i+1) % 25 == 0:
            print(f"   step {i+1}/{len(dl)}  loss={running/(i+1):.3f}")
    return running / max(len(dl), 1)


@torch.no_grad()
def eval_loss(model, dl, device):
    \"\"\"Compute the mean detection loss on the validation split.\"\"\"
    model.train()    # losses are only computed in train mode
    running = 0.0
    for imgs, targets in dl:
        imgs    = [img.to(device) for img in imgs]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        loss_dict = model(imgs, targets)
        running += sum(loss_dict.values()).item()
    return running / max(len(dl), 1)


for ep in range(1, EPOCHS+1):
    print(f"=== epoch {ep}/{EPOCHS} ===")
    tr  = train_one_epoch(model, train_dl, optimizer, DEVICE)
    val = eval_loss(model, val_dl, DEVICE)
    scheduler.step()
    print(f"epoch {ep}: train_loss={tr:.3f}  val_loss={val:.3f}")
"""),
    md("## 7. Inference helper - draw boxes"),
    code("""
@torch.no_grad()
def predict_image(image_id: str, score_thr: float = 0.5):
    model.eval()
    img = Image.open(os.path.join(IMG_DIR, image_id + ".jpg")).convert("RGB")
    tens = TF.to_tensor(img).to(DEVICE)
    out = model([tens])[0]

    keep = out["scores"] >= score_thr
    boxes  = out["boxes"][keep].cpu().numpy()
    scores = out["scores"][keep].cpu().numpy()
    labels_ = out["labels"][keep].cpu().numpy()

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.imshow(img)
    for box, sc, lb in zip(boxes, scores, labels_):
        x1, y1, x2, y2 = box
        rect = patches.Rectangle((x1,y1), x2-x1, y2-y1,
                                 linewidth=2, edgecolor="lime", facecolor="none")
        ax.add_patch(rect)
        ax.text(x1, y1-4, f"{idx_to_class[int(lb)]} {sc:.2f}",
                color="black",
                bbox=dict(facecolor="lime", pad=1, alpha=0.8), fontsize=8)
    ax.set_title(f"image {image_id}: {len(boxes)} detections (thr={score_thr})")
    ax.axis("off"); plt.show()
"""),
    md("## 8. Run inference on a few held-out validation images"),
    code("""
for img_id in val_ids[:6]:
    predict_image(img_id, score_thr=0.5)
"""),
    md("## 9. Quick metric: mean IoU across the validation set"),
    code("""
def box_iou_xyxy(b1, b2):
    \"\"\"IoU of two single boxes in xyxy format.\"\"\"
    xa = max(b1[0], b2[0]); ya = max(b1[1], b2[1])
    xb = min(b1[2], b2[2]); yb = min(b1[3], b2[3])
    inter = max(0, xb-xa) * max(0, yb-ya)
    a1 = max(0, b1[2]-b1[0]) * max(0, b1[3]-b1[1])
    a2 = max(0, b2[2]-b2[0]) * max(0, b2[3]-b2[1])
    union = a1 + a2 - inter
    return inter/union if union > 0 else 0.0


@torch.no_grad()
def mean_iou(model, dl, score_thr=0.5):
    model.eval()
    ious = []
    for imgs, targets in dl:
        imgs = [img.to(DEVICE) for img in imgs]
        outs = model(imgs)
        for out, gt in zip(outs, targets):
            keep = out["scores"].cpu().numpy() >= score_thr
            pboxes = out["boxes"].cpu().numpy()[keep]
            gboxes = gt["boxes"].numpy()
            for gb in gboxes:
                if len(pboxes) == 0:
                    ious.append(0.0); continue
                ious.append(max(box_iou_xyxy(gb, pb) for pb in pboxes))
    return float(np.mean(ious)) if ious else 0.0


print(f"Validation mean-IoU (per-GT, max-match): {mean_iou(model, val_dl):.3f}")
"""),
    md("""
## 10. Conclusions

* Faster R-CNN with an ImageNet+COCO-pretrained backbone reaches a usable
  mean-IoU after just a few epochs on a small subset of the dataset, even
  with a modest GPU.
* Predicted boxes are visually consistent with annotations, with most
  errors on small or partially-occluded objects.
* To productionise this model:
  * use the **full** training set (raise `SUBSET` and `EPOCHS`),
  * add image augmentation (`RandomHorizontalFlip`, color jitter),
  * evaluate with **COCO-style mAP** (`pycocotools`),
  * export to ONNX/TensorRT for real-time inference on the vehicle.
"""),
]

if __name__ == "__main__":
    write_notebook(
        "../solutions/Capstone1_Part1_Vehicle_Object_Detection.ipynb",
        cells,
    )
