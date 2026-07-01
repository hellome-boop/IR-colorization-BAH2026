# IR-colorization-BAH2026
# Geo-Resolve AI: Automated Infrared Image Super-Resolution & Colorization

Developed for the **Bharatiya Antariksh Hackathon 2026 (Problem Statement #10)** under the guidance of ISRO / Space Applications Centre (SAC).

## 🚀 Project Overview
Geo-Resolve AI is a zero-terminal, browser-driven data pipeline designed to enhance and colorize low-resolution thermal infrared (TIR) satellite imagery. The system ingests unoptimized **200m spatial resolution TIR data** and reconstructs it into high-fidelity **100m super-resolved grayscale and colorized layers** using an optimized symmetric U-Net backbone architecture.

## 📦 Mandatory Deliverables Links
* **Trained Model Weights (.pth):** [Download Pre-trained Weights Here](https://drive.google.com/file/d/1hmEVQzt35ROfE7wlRZ1Jz_-GCnwL54X3/view?usp=sharing
)
* **Technical Evaluation Report (PDF):** [View Technical Report Here](./docs/Technical_Report.pdf)

## 📁 Repository Structure
```text
├── models/
│   └── unet_backbone.py       # PyTorch definition of symmetric U-Net architecture
├── app/
│   └── inference_dashboard.py # Gradio zero-terminal browser application layout
├── notebooks/
│   └── Geo_Resolve_Inference.ipynb # Fully executable cloud-based notebook
├── requirements.txt           # Environment library configuration profile
└── README.md                  # System instruction and setup documentation
```

## 🛠️ Installation & Local Execution
Ensure you have Python 3.10+ installed. Execute the following terminal commands to prepare your workspace environment:

```bash
# Clone the codebase repository
git clone https://github.com
cd IR-colorization-BAH2026

# Install foundational geospatial and deep learning libraries
pip install -r requirements.txt

# Run the asynchronous Gradio web dashboard panel
python app/inference_dashboard.py
```

## 🛰️ Production Data Flow Sequence
The operational pipeline executes an automated sequence satisfying institutional specifications:
`Raw 200m TIR Input (.npy/.tif)` ➔ `Dynamic Min-Max 8-bit Normalization` ➔ `U-Net Feature Convolution` ➔ `100m Resolved Grayscale Output` ➔ `Jet Pseudocolor Thermal Indexing`.
