# AIML Capstone Solutions

Five Jupyter notebooks (one per problem statement / part) solving the three
capstones. The notebooks are placed in this `solutions/` folder and read the
provided datasets via **relative paths** (`../Capstone X/...`), so just open
them from this folder and they work end-to-end.

| # | Notebook | Stack | Runs locally? |
|---|----------|-------|---------------|
| 1 | `Capstone1_Part1_Vehicle_Object_Detection.ipynb`         | PyTorch + torchvision Faster R-CNN  | needs PyTorch (Colab T4 GPU recommended) |
| 2 | `Capstone1_Part2_Tesla_Deaths_EDA.ipynb`                 | pandas / matplotlib / seaborn       | **yes (already executed)** |
| 3 | `Capstone2_Part1_Heritage_Structures_Classification.ipynb` | TensorFlow / Keras (MobileNetV2) | needs TF (Colab) |
| 4 | `Capstone2_Part2_Tourism_Recommender.ipynb`              | pandas + scikit-learn               | **yes (already executed)** |
| 5 | `Capstone3_Sales_Forecasting.ipynb`                      | pandas + scikit-learn + XGBoost     | **yes (already executed)** |

## Running locally

```powershell
pip install pandas numpy matplotlib seaborn scikit-learn xgboost openpyxl jupyterlab opencv-python
jupyter lab
```

## Running the deep-learning notebooks (#1 and #3)

These require **TensorFlow** (notebook #3) or **PyTorch** (notebook #1). If
your CPU does not support AVX (you'll see "DLL initialization failed" on
import), upload the notebook + the relevant dataset folder to **Google
Colab** - it has both libraries and a free T4 GPU.

```python
# In a Colab cell:
from google.colab import drive
drive.mount('/content/drive')
# Upload Capstone 1/Part 1 (or Capstone 2/Part 1) into your Drive and
# update the dataset paths near the top of the notebook accordingly.
```

## What each notebook contains

### Capstone 1 - Autonomous Driving
- **Part 1 - Object Detection**: unzip Images.zip, build a `VehicleDataset`,
  fine-tune Faster R-CNN ResNet-50 FPN on 11 vehicle/agent classes, evaluate
  with mean-IoU, and visualise predicted bounding boxes.
- **Part 2 - Tesla Deaths EDA**: full data cleaning, events per
  year/month/day-of-week/state/country, victim profiles, model breakdown and
  verified Autopilot-deaths analysis.

### Capstone 2 - Preserving Heritage / Tourism
- **Part 1 - Heritage Classification**: transfer learning with MobileNetV2,
  augmentation comparison, custom callback that stops at `val_acc >= 0.92`,
  fine-tuning, and inference on the test split.
- **Part 2 - Tourism Recommender**: cleaning, demographic + category EDA,
  identification of best city for nature lovers, and an item-based
  collaborative-filtering recommender (cosine similarity + KNN) that returns
  the top-K places for any queried place.

### Capstone 3 - Sales Forecasting
- Tidy merge of `restaurants.csv`, `items.csv` and `sales.csv`, full sales
  EDA (daily / weekly / monthly / quarterly trends, per-store and per-item
  analysis), and a Linear-Regression vs Random-Forest vs XGBoost comparison
  with proper time-series train/test split (last 6 months as test) plus a
  full one-year (2022) forward forecast generated iteratively.
