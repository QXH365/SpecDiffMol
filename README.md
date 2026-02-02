# SpecDiffMol

SpecDiffMol is an open-source model that predicts molecular SMILES from spectra (IR and Raman) using a Transformer, and then generates 3D molecular conformations based on the predicted SMILES and spectral data using a Diffusion model.

## Requirements

The project requires **Python 3.8+** and a CUDA-capable GPU. Key dependencies include:
- PyTorch
- PyTorch Geometric (PyG)
- RDKit
- NumPy, Pandas, SciPy

See [requirements.txt](requirements.txt) for the full list.

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/SpecDiffMol.git
   cd SpecDiffMol
   ```

2. **Create a virtual environment (Recommended):**
   ```bash
   conda create -n specdiffmol python=3.9
   conda activate specdiffmol
   ```

3. **Install dependencies:**
   First, install PyTorch and PyTorch Geometric according to your CUDA version (see [PyG Installation](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html)).
   
   Then install the rest of the requirements:
   ```bash
   pip install -r requirements.txt
   ```
   
   *Optional:* Install `rxn-chemutils` for improved SMILES tokenization:
   ```bash
   pip install rxn-chemutils
   ```

## Dataset (qme14s)

This project uses the **qme14s** dataset, consisting of paired IR and Raman spectra for molecules.

1. **Download**: Obtain the dataset (IR and Raman CSV files).
   
2. **Structure**: Place the data in a `dataset/qme14s` directory, organizing it as follows:
   ```
   dataset/
   └── qme14s/
       ├── IR_broaden/       # Contains IR_*.csv
       └── Raman_broaden/    # Contains Raman_*.csv
   ```
   
   Each CSV file should contain the SMILES string in the first line, followed by coordinate or spectral data.

## Usage

### Step 1: Spectrum-to-SMILES (Transformer)

1. **Preprocessing**:
   Extract features (peaks) from the raw spectra and prepare training data.
   ```bash
   cd src/step1
   python pre.py --ir_data_dir ../../../dataset/qme14s/IR_broaden --raman_data_dir ../../../dataset/qme14s/Raman_broaden --out_path ./processed_data
   ```

2. **Train Transformer**:
   Train the model to translate spectral peaks to SMILES.
   ```bash
   python train_transformer.py --output_path ./processed_data 
   ```

3. **Filtering & Evaluation**:
   Filter predictions and calculate metrics.
   ```bash
   # Filter top-10 predictions
   python filter10.py \
       --input_file result10_without_fzs.txt \
       --output_file result10_filtered_without_fzs.txt \
       --label_file path/to/tgt-test.txt

   # Calculate metrics
   python check_results.py \
       --label_file path/to/tgt-test.txt \
       --top10_file result10.txt \
       --top1_file result10_filtered.txt \
       --output_csv qme14s_all_results.csv \
       --report_file qme14s_all_metrics_report.txt
   ```

### Step 2: Conformation Generation (Diffusion)

1. **Preprocessing**:
   Prepare the dataset for the diffusion model.
   ```bash
   cd src/step2
   python preprocess.py \
       --ir_dir ../../../dataset/qme14s/IR_broaden \
       --raman_dir ../../../dataset/qme14s/Raman_broaden \
       --test_id_file test_indices.txt \
       --pred_smiles_file result10_filtered_without_fzs.txt \
       --output_dir ./qme14s_processed
   ```

2. **Training**:
   Train the diffusion model.
   ```bash
   python model/train.py --config ./model/configs/finetune.yml --logdir ./logs
   ```

3. **Evaluation**:
   Calculate RMSD for generated conformations.
   ```bash
   # First run inference
   # Then calculate RMSD:
   python model/cal_rmsd.py --result_pkl ./inference_results.pkl --output_csv analysis_report.csv
   
   # Analyze the report:
   python model/test.py analysis_report.csv
   ```

## Directory Structure

```
specdiffmol_code/
├── data/               # Data storage (gitignored)
├── src/
│   ├── step1/          # Transformer (Spectrum -> SMILES)
│   │   ├── pre.py      # Data preprocessing
│   │   └── train_transformer.py
│   └── step2/          # Diffusion (SMILES + Spectrum -> Conformation)
│       └── model/      # Model definitions and training script
├── requirements.txt    # Python dependencies
└── README.md           # This file
```

## Contributing

Contributions are welcome! Please submit a Pull Request or open an Issue.
