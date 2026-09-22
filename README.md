# Quantum-DL-MNIST

**A Comparative Study of Classical and Hybrid Quantum-Classical Deep Learning for Binary MNIST Classification**

<p align="center">

<img src="https://img.shields.io/badge/Python-3.13-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python"/>
<img src="https://img.shields.io/badge/PyTorch-2.11-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch"/>
<img src="https://img.shields.io/badge/PennyLane-0.45.1-6C63FF?style=for-the-badge&logo=pennylane&logoColor=white" alt="PennyLane"/>
<img src="https://img.shields.io/badge/NumPy-2.1.3-013243?style=for-the-badge&logo=numpy&logoColor=white" alt="NumPy"/>

<br/>

<img src="https://img.shields.io/badge/Pandas-2.2.3-150458?style=for-the-badge&logo=pandas&logoColor=white" alt="Pandas"/>
<img src="https://img.shields.io/badge/scikit--learn-1.6.1-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white" alt="scikit-learn"/>
<img src="https://img.shields.io/badge/Matplotlib-3.10.0-11557C?style=for-the-badge&logo=matplotlib&logoColor=white" alt="Matplotlib"/>
<img src="https://img.shields.io/badge/Google%20Colab-CPU%20Only-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white" alt="Google Colab"/>

</p>

---

## Overview

This project investigates a pre-specified research question:

> Does replacing a classical neural network head with a quantum head of approximately matched parameter count produce a detectable performance difference on binary MNIST classification (digits 0 vs 1)?

Rather than simply "trying quantum ML," the project is designed as a controlled ablation study. Four architectures share the same CNN backbone wherever possible, so that any observed difference can be attributed to the specific component under test rather than to confounding factors such as additional capacity, different data, or different training conditions.

## Models Compared

| Model                       | Description                                                                       |
| --------------------------- | --------------------------------------------------------------------------------- |
| **Majority Baseline**       | Predicts the more frequent class; performance floor                               |
| **NN**                      | Fully connected baseline with no convolution                                      |
| **CNN**                     | Convolutional baseline with a shared backbone                                     |
| **CNN + Classical Control** | Shared CNN backbone + small classical head, parameter-matched to the quantum head |
| **CNN + Quantum**           | Shared CNN backbone + 4-qubit variational quantum circuit head using PennyLane    |

The **CNN + Classical vs. CNN + Quantum** comparison is the central research question. The classical control exists specifically to isolate whether quantum processing provides a measurable difference from adding a comparable processing block.

## Dataset

* **MNIST**, restricted to digits **0 and 1** for binary classification
* Official MNIST train/test separation preserved; the datasets were not pooled and re-split
* Stratified train/validation split created from the official training subset
* Official test subset reserved untouched for final evaluation
* Normalization statistics computed from the training split only

## Methodology Highlights

* **Shared CNN backbone** across CNN, CNN + Classical, and CNN + Quantum; the backbone architecture was kept identical except for the head under test
* **Approximate parameter matching** between the classical and quantum heads to reduce model-capacity differences as a confounding factor
* **Pilot phase** conducted before the official experiments to identify implementation issues without contaminating the final runs
* The pilot caught two real implementation issues: a device-compatibility problem in batched quantum execution and a feature-indexing error
* **5 independent runs per model** using fixed shared seeds
* Results reported as **Mean ± SD** with confidence intervals
* **Pre-specified statistical test:** Wilcoxon signed-rank test, paired by seed, comparing CNN + Classical vs. CNN + Quantum
* Statistical results are treated as **exploratory evidence** given the small sample size rather than definitive proof
* Strict data-leakage discipline: the official test set was used only for final evaluation and never for tuning

## Key Findings

* All trained models substantially exceeded the majority-class baseline
* **No statistically distinguishable difference** was found between the classical control and quantum models across accuracy, F1, MCC, or recall (Wilcoxon signed-rank tests, all *p* > 0.05; confidence intervals included zero)
* The quantum model was the **only model to produce false negatives**
* The classical control showed occasional training instability; the quantum model did not show the same behavior
* The quantum model trained **roughly an order of magnitude slower**, despite having substantially fewer trainable parameters. This reflects the computational cost of classical quantum simulation and should not be interpreted as evidence about the speed of real quantum hardware

### Interpretation

On this small binary classification task, the quantum head neither improved nor degraded performance in a statistically detectable way relative to the parameter-matched classical control, while incurring substantially higher computational cost under classical simulation.

The result is therefore a **null finding under the tested conditions**, rather than evidence that quantum models are universally equivalent to or inferior to classical models.

## Repository Structure

```text
Quantum-DL-MNIST/
│
├── docs/
│   └── Quantum_MNIST_Research_Report.pdf
│
├── figures/
│   ├── training and validation curves
│   ├── confusion matrices
│   ├── ROC curves
│   └── computational comparison figures
│
├── notebooks/
│   ├── 00_Data_Preparation.ipynb
│   ├── 01_NN_Binary_MNIST.ipynb
│   ├── 02_CNN_Binary_MNIST.ipynb
│   ├── 03_CNN_Classical_Control.ipynb
│   ├── 04_CNN_Quantum.ipynb
│   ├── 05_Final_Analysis.ipynb
│   ├── Pilot_Phase.ipynb
│   ├── Quantum_Fundamentals_Primer.ipynb
│   └── shared_backbone.py
│
├── results/
│   ├── model metrics
│   ├── training histories
│   ├── summary statistics
│   └── statistical analysis
│
├── splits/
│   ├── train_indices.npy
│   ├── val_indices.npy
│   ├── test_indices.npy
│   └── normalization_stats.json
│
├── .gitignore
├── requirements.txt
├── README.md
└── LICENSE
```

The notebooks contain the experimental workflow and recorded outputs, while `shared_backbone.py` contains shared model logic used across the relevant experiments.

## Experimental Environment

The experiments were conducted in **Google Colab** using CPU-only computation.

### Quantum Simulation

* **4-qubit** variational quantum circuit
* PennyLane
* `lightning.qubit` backend where available
* `default.qubit` fallback documented in the experimental workflow

See [`requirements.txt`](requirements.txt) for the pinned Python dependencies.

## Reproducibility Notes

The repository intentionally does **not** include the raw MNIST dataset or trained model checkpoints. The notebooks, preprocessing split information, figures, recorded results, and research report are included.

The experiments were originally conducted in Google Colab, and the pinned dependencies reflect the core package versions used in that environment.

Because the raw dataset and trained checkpoints are excluded from the repository, reproducing the complete experiment requires obtaining MNIST separately and rerunning the relevant notebooks.

## Research Report

The complete research report, including methodology, experimental results, figures, statistical analysis, and appendices, is available at:

**[`docs/Quantum_MNIST_Research_Report.pdf`](docs/Quantum_MNIST_Research_Report.pdf)**

## Notes on the Research Process

The experimental protocol went through multiple rounds of critical review before implementation began, specifically to identify design issues that could undermine the validity of the comparison, including data leakage risks, parameter-matching gaps, asymmetric fallbacks between models, and reproducibility concerns.

Development also involved Claude as a research and coding collaborator. Design decisions, corrections, implementation details, and interpretation were actively driven and verified by the author rather than accepted without verification. The pilot phase, for example, identified real implementation issues that were subsequently corrected before the official experiments.

---

*This project was completed as part of an independent deep learning study assigned through coursework, using a beginner-accessible but methodologically rigorous experimental design.*
