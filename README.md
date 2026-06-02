# 🗑️ Brussels Waste Classifier

A computer vision model that recognizes household waste from a photo and recommends the correct bin according to **Brussels waste sorting rules** (Bruxelles Propreté).

---

## 🎯 Approach

Rather than training the model directly on bin categories (PMC, paper, glass...), a two-step approach is used:

1. **The ML model** learns to recognize the object type (plastic bottle, can, cardboard...)
2. **A business rule** (Python dictionary) maps each object to the correct Brussels bin

This approach yields better accuracy because categories like PMC group visually very different objects (plastic, metal, drink cartons). A model trained on visually coherent classes generalizes much better.

---

## 🗂️ Dataset

- **Source:** [Recyclable and Household Waste Classification](https://www.kaggle.com/datasets/alistairking/recyclable-and-household-waste-classification) — Kaggle
- **~7,500 images** (real_world only) across **29 object classes**
- Each class contains two subfolders: `default/` (studio images) and `real_world/` (real conditions)
- Only `real_world/` images are used for training to avoid bias from black backgrounds

---

## ♻️ Brussels Sorting Rules

| Bin | Color | Contents |
|-----|-------|----------|
| PMC | Blue bag | Plastics, metals, drink cartons |
| Paper/Cardboard | Yellow bag | Paper, cardboard, newspapers |
| Glass | Green container | Bottles and glass jars |
| Organic | Orange bag | Food waste |
| Garden | Green bag | Garden waste *(no data yet)* |
| Residual | Black bag | Everything else |

---

## 🏗️ Model

- **Architecture:** ResNet50 pre-trained on ImageNet (transfer learning)
- **Strategy:** All layers frozen except the final `fc` layer, replaced with a 29-class output
- **Optimizer:** Adam (lr=1e-3)
- **Loss:** CrossEntropyLoss
- **Batch size:** 32
- **Image size:** 224×224
- **Normalization:** ImageNet mean/std

---

## 📊 Results

Three model versions were trained and compared:

| Version | Dataset | Split | Test Accuracy |
|---------|---------|-------|---------------|
| v1 | default + real_world | 80/20 | ~83.3% |
| v2 | real_world only | 80/20 | ~79.6% |
| v3 | real_world only | 70/15/15 | **79.1%** |

> ⚠️ v1 and v2 accuracy scores are not directly comparable to v3 — they used a non-rigorous test set. Only **v3** has a truly unseen test set (used once at the very end).

### Best recognized classes (>95%)
| Class | Accuracy |
|-------|----------|
| `disposable_plastic_cutlery` | 97.4% |
| `eggshells` | 97.1% |
| `coffee_grounds` | 96.0% |
| `aerosol_cans` | 95.8% |
| `shoes` | 95.8% |

### Most difficult classes (<55%)
| Class | Accuracy | Reason |
|-------|----------|--------|
| `aluminum_food_cans` | 34.9% | Confused with `steel_food_cans` |
| `plastic_soda_bottles` | 48.7% | Confused with `plastic_water_bottles` |
| `steel_food_cans` | 51.4% | Confused with `aluminum_food_cans` |

> **Key insight:** most confusions occur between classes that belong to the **same Brussels bin** — the waste sorting mapping remains reliable despite these classification errors.

---

## 📁 Project Structure

```
brussels_waste_classifier/
├── data/                          # Dataset (not versioned)
│   ├── aerosol_cans/
│   │   ├── default/
│   │   └── real_world/
│   └── ...
├── models/                        # Saved model weights
│   ├── waste_classifier_resnet50_v1.pth
│   ├── waste_classifier_resnet50_v2_realworld.pth
│   └── waste_classifier_resnet50_v3.pth
├── 04_Brussels_Waste_Classifier.ipynb
└── README.md
```

---

## 🚀 Getting Started

### Requirements

```bash
pip install torch torchvision matplotlib Pillow numpy jupyter ipykernel
```

### Run the notebook

```bash
jupyter notebook 04_Brussels_Waste_Classifier.ipynb
```

### Predict on a new image

```python
from pathlib import Path
predict(model_v3, Path('your_image.jpg'), test_transform, full_dataset_v2.classes, device)
```

---

## 🔮 Future Work

- **v4 — ImageFolder structure:** reorganize images into `train/val/test/` folders to use `torchvision.ImageFolder` and remove the custom `WasteDataset` class
- **v4 — Class merging:** merge visually similar classes (`aluminum_food_cans` + `steel_food_cans` → `metal_food_cans`) to improve accuracy
- **v5 — Fine-tuning:** unfreeze the last ResNet50 layers and retrain with a very low learning rate
- **v6 — TACO dataset:** integrate the [TACO dataset](http://tacodataset.org/) for more realistic waste images
- **App:** develop a web/mobile application with FastAPI backend that takes a photo and returns the recommended Brussels bin

---

## 📚 References

- [Zero to Mastery PyTorch Course](https://www.learnpytorch.io/) — Daniel Bourke
- [Kaggle Dataset](https://www.kaggle.com/datasets/alistairking/recyclable-and-household-waste-classification) — Alistair King
- [Bruxelles Propreté](https://www.bruxelles-proprete.be/) — Brussels waste sorting rules
- [TACO Dataset](http://tacodataset.org/) — Trash Annotations in Context

---

*Project built as a practical application of the [Zero to Mastery PyTorch](https://www.learnpytorch.io/) course by Daniel Bourke.*
