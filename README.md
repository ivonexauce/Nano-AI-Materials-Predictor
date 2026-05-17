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

This project bridges **computational nanoscience** and **modern deep learning** to predict material properties at the atomic level — without running expensive DFT simulations for every candidate material.

Key capabilities:
- **Graph Neural Networks (GNNs)** treating crystal structures as graphs of atoms and bonds
- **Materials Project API** integration for real nanoscale material datasets
- **Single & multi-target prediction**: bandgap, formation energy, bulk modulus
- **Active learning loop** to prioritize which materials to simulate next
- **Interactive visualization** of property distributions and data

---

## 🏗️ Architecture

```
┌──────────────────────────────────────────────────────────────┐
│              DATA LAYER — Materials Project API               │
│   Crystal Structures │ DFT Properties │ Composition           │
└──────────────────────────────┬───────────────────────────────┘
                               │
┌──────────────────────────────▼───────────────────────────────┐
│              CGCNN MODEL                                      │
│   Crystal Graph Convolution │ Global Mean Pooling             │
└──────────────────────────────┬───────────────────────────────┘
                               │
          ┌────────────────────┼─────────────────────┐
          ▼                    ▼                      ▼
  ┌──────────────┐   ┌──────────────────┐   ┌────────────────┐
  │  PREDICTION  │   │  ACTIVE LEARNING │   │  VISUALIZATION │
  │  Bandgap     │   │  Uncertainty     │   │  Property      │
  │  Formation E │   │  MC Dropout      │   │  Distributions │
  │  Bulk Modulus│   │  Query by Batch  │   │  Parity Plot   │
  └──────────────┘   └──────────────────┘   └────────────────┘
```

---

## 📁 Project Structure

```
Nano-AI-Materials-Predictor/
│
├── data/
│   ├── fetcher.py                   # Materials Project API client + synthetic data generator
│   └── sample_materials.json        # Generated sample materials (no API key needed)
│
├── models/
│   └── cgcnn.py                     # Crystal Graph Convolutional Neural Network
│
├── prediction/
│   ├── trainer.py                   # Training loop with early stopping, batching, checkpointing
│   └── active_learner.py            # Uncertainty-based active learning (MC Dropout)
│
├── visualization/
│   └── dashboard.py                 # Streamlit interactive dashboard
│
├── checkpoints/                     # Saved model weights and training history
│
├── .env.example
├── .gitignore
├── LICENSE
├── requirements.txt
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
python prediction/trainer.py --data data/sample_materials.json --target bandgap_ev

# Run active learning demo
python prediction/active_learner.py

# Launch dashboard
streamlit run visualization/dashboard.py
```

### Option 2: Full Materials Project Integration

```bash
cp .env.example .env
# Add your Materials Project API key to .env:
# MP_API_KEY=your_key_here

# Fetch materials
python data/fetcher.py --n-materials 1000

# Train GNN
python prediction/trainer.py --target bandgap_ev --epochs 200
```

---

## 🧠 Model

### CGCNN (Crystal Graph Convolutional Neural Network)
Treats crystals as periodic graphs with atoms as nodes and bonds as edges. Uses gated residual message passing with global mean pooling. Implements the architecture from Xie & Grossman (PRL 2018).

---

## 🎯 Target Properties

| Property | Unit | Use Case |
|---|---|---|
| Bandgap | eV | Semiconductor design |
| Formation Energy | eV/atom | Stability screening |
| Bulk Modulus | GPa | Mechanical hardness |

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
# Launch interactive dashboard
streamlit run visualization/dashboard.py
```

The Streamlit dashboard includes: bandgap distribution histogram, element frequency chart, formation energy vs bandgap scatter plot, mock parity plot, and an interactive data table with filters.

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
