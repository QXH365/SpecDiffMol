# SpecDiffMol

SpecDiffMol is a multimodal spectrum-to-3D molecular conformation generation model that employs a two-stage strategy. It first predicts molecular SMILES from multimodal spectra via a Transformer-based model, followed by generating 3D conformations based on the predicted SMILES and spectral data using a diffusion-based model.

## Requirements

The project requires **Python 3.8+** and a CUDA-capable GPU. Key dependencies include:
- PyTorch
- PyTorch Geometric (PyG)
- RDKit
- NumPy, Pandas, SciPy


## Installation

1. **Create an environment:**
   ```bash
   conda create -n specdiffmol python=3.9
   conda activate specdiffmol
   ```

2. **Install dependencies:**
   First, install PyTorch and PyTorch Geometric according to your CUDA version (see [PyG Installation](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html)).
   
   Then install the rest of the requirements:
   ```bash
   pip install -r requirements.txt
   ```

## Dataset

This project uses the QMe14s dataset, consisting of paired IR and Raman spectra for 3D molecules.

1. **Download**: Obtain the [dataset](https://figshare.com/s/889262a4e999b5c9a5b3)(IR and Raman CSV files).
   
2. **Structure**: Place the data in a `dataset/qme14s` directory, organizing it as follows:
   ```
   dataset/
   └── qme14s/
       ├── IR_broaden/       # Contains IR_*.csv
       └── Raman_broaden/    # Contains Raman_*.csv
   ```

## Usage

### Step 1: Spectrum-to-SMILES

1. **Preprocessing**:
   Extract features (peaks) from the raw spectra and prepare training data.
   ```bash
   cd src/step1
   python preprocess.py --ir_data_dir ../../dataset/qme14s/IR_broaden --raman_data_dir ../../dataset/qme14s/Raman_broaden --out_path ./processed_data
   ```

2. **Train Transformer**:
   Train the model to translate spectral peaks to SMILES.
   ```bash
   python train_transformer.py --output_path ./processed_data --template_path ./transformer_template.yaml
   ```

3. **Prediction**:
   ```bash
   onmt_translate -model path_to_the_checkpoint -src ./processed_data/src-test.txt -n_best 10 -output ./pred.txt -gpu 0 -verbose
   ```

4. **Filtering & Evaluation**:
   Filter predictions and calculate metrics.
   ```bash
   # Filter top-10 predictions
   python filter_data.py \
       --input_file pred.txt \
       --output_file result.txt \
       --label_file ./processed_data/tgt-test.txt  # Used strictly for molecular formula loading

   # Calculate metrics
   python check_results.py \
       --label_file ./processed_data/tgt-test.txt \
       --top10_file pred.txt \
       --top1_file result.txt \
       --output_csv results.csv \
       --report_file metrics_report.txt
   ```

### Step 2: Conformation Generation

1. **Preprocessing**:
   Build the diffusion-model dataset from the QMe14s spectra, the SMILES predicted in Step 1, and the true 3D coordinates. 
   ```bash
   cd src/step2/preprocess
   # Copy/symlink the two Step 1 outputs into this directory first:
   #   test_indices.txt   (from src/step1/processed_data/data/)
   #   result.txt         (Step 1's filtered top-1 predictions)
   python preprocess.py
   cd ..
   ```

2. **Training**:
   Train the diffusion model.
   ```bash
   python model/train.py --config ./model/config.yml --logdir ./logs
   ```

3. **Evaluation**:
   Calculate RMSD for generated conformations.
   ```bash
   # First run inference
   python model/test.py path_to_the_checkpoint --config ./model/config.yml
   
   # Then calculate RMSD:
   python model/cal_rmsd.py --result_pkl ./inference_results.pkl --output_csv analysis_report.csv
   ```

## Directory Structure

```
specdiffmol_code/
├── data/               # Data storage (gitignored)
├── src/
│   ├── step1/          # Transformer (Spectrum -> SMILES)    
│   └── step2/          # Diffusion (SMILES + Spectrum -> Conformation)
│       ├── model/      # Model definitions and training script
│   │   └── preprocess/ # Data preprocessing  
├── requirements.txt    # Python dependencies
└── README.md           # This file
```
## License

SpecDiffMol is released under the [MIT](LICENSE.txt) license.

