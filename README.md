# ⚛️ Nano-AI-Materials-Predictor

![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red.svg)
![Status](https://img.shields.io/badge/Status-Active-green.svg)

> **ML-Driven Property Prediction for Nanoscale Materials using Graph Neural Networks and the Materials Project API**

Built by **UMBA YANGA IVON EXAUCE** — Deep-Tech Systems Architect & Innovation Strategist | UMBA Consulting Engineers

*Connecting MTech Nanoscience & Nanotechnology with PhD-level AI research.*

---

## 🎯 Overview

This project bridges **computational nanoscience** and **modern deep learning** to predict material properties at the atomic level — without running expensive molecular dynamics simulations for every candidate material.

Key capabilities:
- **Graph Neural Networks (GNNs)** treating crystal structures as graphs of atoms and bonds
- **Materials Project API** integration for real nanoscale material datasets
- **Multi-target prediction**: bandgap, formation energy, bulk modulus, electrical conductivity
- **Active learning loop** to prioritize which materials to simulate next
- **Interactive visualization** of crystal structures and property landscapes

---

## 🏗️ Architecture

```
┌──────────────────────────────────────────────────────────────┐
│              DATA LAYER — Materials Project API               │
│   Crystal Structures (CIF) │ DFT Properties │ Composition    │
└──────────────────────────────┬───────────────────────────────┘
                               │
┌──────────────────────────────▼───────────────────────────────┐
│              GRAPH CONSTRUCTION LAYER                         │
│   Atoms = Nodes │ Bonds = Edges │ Features = Atomic Props    │
└──────────────────────────────┬───────────────────────────────┘
                               │
┌──────────────────────────────▼───────────────────────────────┐
│              GNN MODEL LAYER                                  │
│   Message Passing │ Graph Attention │ Global Pooling         │
│   MEGNet │ SchNet │ DimeNet++ │ Custom CGCNN                 │
└──────────────────────────────┬───────────────────────────────┘
                               │
          ┌────────────────────┼─────────────────────┐
          ▼                    ▼                      ▼
  ┌──────────────┐   ┌──────────────────┐   ┌────────────────┐
  │  PREDICTION  │   │  ACTIVE LEARNING │   │  VISUALIZATION │
  │  Bandgap     │   │  Uncertainty     │   │  3D Structure  │
  │  Formation E │   │  Sampling        │   │  Property Maps │
  │  Bulk Modulus│   │  Next Best Exp.  │   │  Dashboards    │
  └──────────────┘   └──────────────────┘   └────────────────┘
```

---

## 📁 Project Structure

```
Nano-AI-Materials-Predictor/
│
├── data/
│   ├── fetcher.py                   # Materials Project API client
│   ├── preprocessor.py              # CIF → graph conversion
│   ├── dataset.py                   # PyTorch Geometric dataset
│   └── sample_materials.json        # 50 sample materials (no API key needed)
│
├── models/
│   ├── cgcnn.py                     # Crystal Graph Convolutional Neural Network
│   ├── megnet.py                    # Multi-layer Edge Graph Network
│   ├── schnet.py                    # SchNet (distance-based convolution)
│   └── model_factory.py             # Model selection and initialization
│
├── gnn/
│   ├── graph_builder.py             # Atom/bond graph construction
│   ├── message_passing.py           # Custom message passing layer
│   ├── attention.py                 # Graph attention mechanism
│   └── pooling.py                   # Global graph pooling strategies
│
├── prediction/
│   ├── trainer.py                   # Training loop with early stopping
│   ├── evaluator.py                 # MAE, RMSE, R² metrics
│   ├── predictor.py                 # Inference on new materials
│   └── active_learner.py            # Uncertainty-based active learning
│
├── visualization/
│   ├── structure_viewer.py          # 3D crystal structure renderer
│   ├── property_plotter.py          # Property distribution plots
│   ├── parity_plot.py               # Predicted vs actual plots
│   └── dashboard.py                 # Streamlit interactive dashboard
│
├── utils/
│   ├── periodic_table.py            # Atomic feature vectors
│   ├── crystal_utils.py             # Symmetry and lattice utilities
│   └── metrics.py                   # Custom evaluation metrics
│
├── notebooks/
│   ├── 01_data_exploration.ipynb    # EDA on Materials Project data
│   ├── 02_graph_construction.ipynb  # Visual walkthrough of GNN input
│   ├── 03_model_training.ipynb      # Step-by-step training notebook
│   └── 04_active_learning.ipynb     # Active learning experiment
│
├── tests/
│   ├── test_graph_builder.py
│   ├── test_model.py
│   └── test_predictor.py
│
├── requirements.txt
├── Dockerfile
└── README.md
```

---

## 🚀 Quick Start

### Option 1: Without Materials Project API Key (sample data)

```bash
git clone https://github.com/ivonexauce/Nano-AI-Materials-Predictor.git
cd Nano-AI-Materials-Predictor
pip install -r requirements.txt

# Train on sample materials (no API key needed)
python prediction/trainer.py --data data/sample_materials.json --target bandgap

# Predict for a new composition
python prediction/predictor.py --formula "TiO2"

# Launch dashboard
streamlit run visualization/dashboard.py
```

### Option 2: Full Materials Project Integration

```bash
cp .env.example .env
# Add your Materials Project API key to .env:
# MP_API_KEY=your_key_here

# Fetch 10,000 materials
python data/fetcher.py --n-materials 10000 --properties bandgap,formation_energy

# Build graph dataset
python data/preprocessor.py

# Train GNN
python prediction/trainer.py --model cgcnn --target bandgap --epochs 200
```

---

## 🧠 Models

### CGCNN (Crystal Graph Convolutional Neural Network)
The baseline model. Treats crystals as periodic graphs with atoms as nodes and bonds as edges. Fast to train, strong baseline for most property prediction tasks.

### MEGNet (Multi-layer Edge Graph Network)
Extends CGCNN by also updating edge features during message passing — capturing bond length and angle changes under different conditions (temperature, pressure).

### SchNet
Distance-aware continuous-filter convolution. Particularly effective for predicting quantum-mechanical properties like HOMO-LUMO gaps and electron density.

---

## 🎯 Target Properties

| Property | Unit | MAE (CGCNN) | Use Case |
|---|---|---|---|
| Bandgap | eV | ~0.25 eV | Semiconductor design |
| Formation Energy | eV/atom | ~0.08 eV | Stability screening |
| Bulk Modulus | GPa | ~15 GPa | Mechanical hardness |
| Shear Modulus | GPa | ~12 GPa | Material toughness |

*MAE values based on the original CGCNN paper benchmarks.*

---

## 🔬 Active Learning

The active learning module identifies which unmeasured materials the model is **most uncertain** about — prioritizing them for expensive DFT simulations. This accelerates materials discovery by up to 10× compared to random sampling.

```
Unlabeled Pool → Uncertainty Estimation → Select Top-K → DFT/Experiment → Retrain
```

---

## 🔗 Research Connection

This project is directly connected to the intersection of:
- **MTech Nanoscience & Nanotechnology** — crystal structure understanding, DFT baselines
- **PhD in Computer Studies** — AI/ML architecture, graph learning
- **Second PhD in Electronics & ECE/IoT** — material-informed sensor design

*Target publication venue: npj Computational Materials, Journal of Chemical Information and Modeling*

---

## 📊 Visualization

```bash
# 3D crystal structure viewer
python visualization/structure_viewer.py --material "mp-149"  # Silicon

# Property heatmap across composition space
python visualization/property_plotter.py --property bandgap

# Parity plot (predicted vs DFT)
python visualization/parity_plot.py --model cgcnn --target formation_energy
```

---

## 📜 License
MIT License — free to use, distribute, and modify.

---

## 🙌 Author

**UMBA YANGA IVON EXAUCE (Ebb)**  
Deep-Tech Systems Architect & Innovation Strategist  
Founder & CEO — UMBA Consulting Engineers

🎓 AI • Computational Nanoscience • Blockchain Security • Smart Enterprise Systems  
🌐 [umbaconsulting.com](https://umbaconsulting.com) | 📧 umbayanga6bio@gmail.com

> *"Intelligence begins where simulation ends — let the machine learn what atoms already know."*
