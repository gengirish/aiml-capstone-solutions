"""Generate AIML capstone submission packages (writeup PDF, screenshots zip, remarks)."""
from __future__ import annotations

import base64
import json
import zipfile
from pathlib import Path

from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Paragraph, SimpleDocTemplate

ROOT = Path(__file__).resolve().parents[2]
SOLUTIONS = ROOT / "solutions"
OUT = Path(__file__).resolve().parent


def _pdf_writer(path: Path):
    styles = getSampleStyleSheet()
    title = ParagraphStyle(
        "Title", parent=styles["Heading1"], fontSize=18, spaceAfter=14
    )
    h2 = ParagraphStyle(
        "H2", parent=styles["Heading2"], fontSize=13, spaceBefore=12, spaceAfter=6
    )
    body = ParagraphStyle(
        "Body", parent=styles["BodyText"], fontSize=10.5, leading=14, spaceAfter=8
    )
    doc = SimpleDocTemplate(
        str(path),
        pagesize=letter,
        leftMargin=0.85 * inch,
        rightMargin=0.85 * inch,
        topMargin=0.75 * inch,
        bottomMargin=0.75 * inch,
    )
    story: list = []

    def add(text: str, style=body):
        story.append(Paragraph(text.replace("&", "&amp;"), style))

    return doc, story, add, title, h2


def extract_notebook_figures(
    nb_path: Path, out_dir: Path, prefix: str = "fig"
) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    data = json.loads(nb_path.read_text(encoding="utf-8"))
    saved: list[Path] = []
    fig_idx = 0
    for cell in data.get("cells", []):
        for output in cell.get("outputs", []):
            if output.get("output_type") != "display_data":
                continue
            png_b64 = output.get("data", {}).get("image/png")
            if not png_b64:
                continue
            fig_idx += 1
            fname = out_dir / f"{prefix}_{fig_idx:02d}.png"
            fname.write_bytes(base64.b64decode(png_b64))
            saved.append(fname)
    return saved


def zip_pngs(pngs: list[Path], zip_path: Path) -> None:
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in sorted(pngs):
            zf.write(f, arcname=f.name)


def build_capstone1_writeup(path: Path) -> None:
    doc, story, add, title, h2 = _pdf_writer(path)
    add("AIML Capstone 1: Autonomous Driving", title)
    add(
        "Part 1: vehicle object detection (Faster R-CNN). "
        "Part 2: Tesla Autopilot and road-safety EDA."
    )
    add("1. Business context", h2)
    add(
        "AV systems need real-time detection of vehicles, pedestrians, and cyclists. "
        "Part 1 builds a bounding-box detector; Part 2 analyses fatal Tesla incidents."
    )
    add("2. Part 1 — Object detection", h2)
    add("<b>Dataset.</b> 5,626 images + labels.csv (11 classes, xyxy boxes).")
    add("<b>Model.</b> Faster R-CNN ResNet-50 FPN (COCO pretrained), custom classification head.")
    add("<b>Training.</b> 90/10 split, 1,500-image subset, SGD, 3 epochs, batch 4.")
    add("<b>Evaluation.</b> Validation mean IoU and qualitative inference plots.")
    add("3. Part 2 — Tesla deaths EDA", h2)
    add("<b>Dataset.</b> 307 rows cleaned to 303 accident records.")
    add("• 247 accidents with 1 death; Tesla driver fatalities sum: 2,371.")
    add("• 47.5% of events had Tesla driver/occupant death.")
    add("• 49 cyclist/pedestrian events; 38% involved other-vehicle fatalities.")
    add("• Top models: Unknown (188), S (45), 3 (35), X (22), Y (13).")
    add("4. Tools", h2)
    add("Part 1: PyTorch/torchvision (GPU/Colab). Part 2: pandas, matplotlib, seaborn.")
    add("5. Conclusion", h2)
    add(
        "Detection pipeline plus safety analytics address the autonomous-driving scenario."
    )
    doc.build(story)


def build_capstone2_writeup(path: Path) -> None:
    doc, story, add, title, h2 = _pdf_writer(path)
    add("AIML Capstone 2: Preserving Heritage / Tourism", title)
    add(
        "Part 1: heritage structure classification (TensorFlow transfer learning). "
        "Part 2: Indonesia tourism EDA and collaborative filtering recommender."
    )
    add("1. Business context", h2)
    add(
        "Government agency monitors historical structures and promotes tourism via "
        "personalised recommendations."
    )
    add("2. Part 1 — Heritage classification", h2)
    add("<b>Dataset.</b> 10 architectural element classes; train + unlabelled test set.")
    add("<b>Exploration.</b> Sample images per class, class distribution, OpenCV previews.")
    add("<b>Model.</b> MobileNetV2 (ImageNet weights), frozen conv layers, "
        "GAP → Dropout → Dense softmax head.")
    add("<b>Training.</b> Without augmentation first, then with augmentation; "
        "EarlyStopping, ModelCheckpoint, custom callback stopping at val_accuracy ≥ 0.92.")
    add("<b>Evaluation.</b> Train/val accuracy curves; inference on Dataset_test images.")
    add("3. Part 2 — Tourism recommender", h2)
    add("<b>Datasets.</b> 437 places, 10,000 ratings (9,921 after dedup), 300 users.")
    add("<b>Cleaning.</b> Imputed Time_Minutes median; removed 79 duplicate ratings.")
    add("<b>EDA.</b> User age/location profiles; category mix by city; "
        "Yogyakarta best for nature (57 Cagar Alam/Bahari spots).")
    add("<b>Ratings.</b> Yogyakarta highest mean place rating; "
        "Taman Hiburan most liked category (avg 3.12).")
    add("<b>Recommender.</b> Item-based CF: 437×300 rating matrix, cosine similarity + KNN; "
        "top-K places given current location (e.g. Monumen Nasional).")
    add("4. Tools", h2)
    add("Part 1: TensorFlow/Keras, OpenCV (GPU/Colab). Part 2: pandas, sklearn, seaborn.")
    add("5. Conclusion", h2)
    add(
        "Transfer-learned classifier supports structure monitoring; CF recommender "
        "suggests similar destinations for tourists."
    )
    doc.build(story)


def build_capstone3_writeup(path: Path) -> None:
    doc, story, add, title, h2 = _pdf_writer(path)
    add("AIML Capstone 3: Sales Forecasting", title)
    add(
        "Fresh Analytics forecasts restaurant item demand using merged sales data, "
        "EDA, and ML models."
    )
    add("1. Business context", h2)
    add(
        "Accurate demand forecasts drive production, staffing, and inventory decisions "
        "across restaurant chains."
    )
    add("2. Data preparation", h2)
    add("<b>Sources.</b> restaurants.csv (6 stores), items.csv (100 items), "
        "sales.csv (109,600 daily records).")
    add("<b>Merge.</b> Single table with date, item, price, count, kcal, store id/name.")
    add("<b>Cleaning.</b> Outlier cap at 99.5th percentile on item_count (241 units).")
    add("3. Exploratory analysis", h2)
    add("Daily/weekly/monthly/quarterly sales trends; day-of-week patterns.")
    add("Bob's Diner leads in volume and revenue; per-store and per-item breakdowns.")
    add("Most expensive items and calorie counts identified per restaurant.")
    add("4. Forecasting models", h2)
    add("<b>Features.</b> Year, month, day, day-of-week, quarter, lag/rolling stats.")
    add("<b>Split.</b> Last 6 months as test (train 88,200 / test 18,400 rows).")
    add("<b>RMSE (test).</b> Linear Regression 3.16; Random Forest 3.07; XGBoost 2.88 (best).")
    add("<b>Forecast.</b> XGBoost used for iterative one-year (2022) forward forecast.")
    add("5. Tools & conclusion", h2)
    add("pandas, scikit-learn, XGBoost, matplotlib. XGBoost generalises best for this demand series.")
    doc.build(story)


CAPSTONES = {
    1: {
        "slug": "capstone1",
        "title": "Autonomous_Driving",
        "writeup": build_capstone1_writeup,
        "notebooks": [
            (SOLUTIONS / "Capstone1_Part1_Vehicle_Object_Detection.ipynb", "c1p1"),
            (SOLUTIONS / "Capstone1_Part2_Tesla_Deaths_EDA.ipynb", "c1p2"),
        ],
        "remarks": (
            "Capstone 1: Part 1 (Faster R-CNN object detection) + Part 2 (Tesla deaths EDA). "
            "Part 2 executed locally. Part 1 needs Google Colab (T4 GPU) for training screenshots. "
            "Upload both .ipynb files as source code. Dataset paths are relative to solutions/."
        ),
        "gpu_note": "Part 1 requires Colab GPU for screenshots.",
    },
    2: {
        "slug": "capstone2",
        "title": "Heritage_Tourism",
        "writeup": build_capstone2_writeup,
        "notebooks": [
            (SOLUTIONS / "Capstone2_Part1_Heritage_Structures_Classification.ipynb", "c2p1"),
            (SOLUTIONS / "Capstone2_Part2_Tourism_Recommender.ipynb", "c2p2"),
        ],
        "remarks": (
            "Capstone 2: Part 1 (MobileNetV2 heritage classification) + Part 2 (tourism EDA + "
            "item-based collaborative filtering). Part 2 executed locally with full outputs. "
            "Part 1 needs Google Colab GPU (TensorFlow). Upload both .ipynb files. "
            "Screenshots include Part 2 charts; add Part 1 training curves after Colab run."
        ),
        "gpu_note": "Part 1 requires Colab GPU for screenshots.",
    },
    3: {
        "slug": "capstone3",
        "title": "Sales_Forecasting",
        "writeup": build_capstone3_writeup,
        "notebooks": [
            (SOLUTIONS / "Capstone3_Sales_Forecasting.ipynb", "c3"),
        ],
        "remarks": (
            "Capstone 3: merged restaurant sales EDA, Linear Regression vs Random Forest vs "
            "XGBoost comparison (XGBoost best test RMSE 2.88), and one-year forward forecast. "
            "Fully executed locally. Upload Capstone3_Sales_Forecasting.ipynb as source code."
        ),
        "gpu_note": None,
    },
}


def generate_capstone(num: int) -> dict:
    cfg = CAPSTONES[num]
    cap_dir = OUT / cfg["slug"]
    screens_dir = cap_dir / "screenshots"
    cap_dir.mkdir(parents=True, exist_ok=True)

    pdf_path = cap_dir / f"Capstone{num}_{cfg['title']}_Writeup.pdf"
    cfg["writeup"](pdf_path)

    all_figs: list[Path] = []
    for nb_path, prefix in cfg["notebooks"]:
        if nb_path.exists():
            all_figs.extend(extract_notebook_figures(nb_path, screens_dir, prefix))

    zip_path = cap_dir / f"Capstone{num}_Screenshots.zip"
    if all_figs:
        zip_pngs(all_figs, zip_path)

    remarks_path = cap_dir / "Additional_Remarks.txt"
    remarks_path.write_text(cfg["remarks"], encoding="utf-8")

    return {
        "dir": cap_dir,
        "pdf": pdf_path,
        "zip": zip_path if all_figs else None,
        "figures": len(all_figs),
        "notebooks": [nb.name for nb, _ in cfg["notebooks"]],
        "gpu_note": cfg["gpu_note"],
    }


def main():
    print("Generating submission packages...\n")
    for n in (1, 2, 3):
        info = generate_capstone(n)
        print(f"Capstone {n}: {info['dir']}")
        print(f"  Writeup:     {info['pdf'].name}")
        print(f"  Screenshots: {info['figures']} figures", end="")
        if info["zip"]:
            print(f" -> {info['zip'].name}")
        else:
            print(" (none — run GPU notebooks on Colab first)")
        print(f"  Source code: {', '.join(info['notebooks'])}")
        if info["gpu_note"]:
            print(f"  Note:        {info['gpu_note']}")
        print()


if __name__ == "__main__":
    main()
