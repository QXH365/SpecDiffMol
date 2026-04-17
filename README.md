# SpecDiffMol

SpecDiffMol is a multimodal spectrum-to-3D molecular conformation generation model that employs a two-stage strategy. It first predicts molecular SMILES from multimodal spectra via a Transformer-based model, followed by generating 3D conformations based on the predicted SMILES and spectral data using a diffusion-based model.

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
   git clone https://github.com/QXH365/SpecDiffMol
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

1. **Download**: Obtain the [dataset](https://figshare.com/s/889262a4e999b5c9a5b3)(IR and Raman CSV files).
   
2. **Structure**: Place the data in a `dataset/qme14s` directory, organizing it as follows:
   ```
   dataset/
   └── qme14s/
       ├── IR_broaden/       # Contains IR_*.csv
       └── Raman_broaden/    # Contains Raman_*.csv
   ```

## Usage

### Step 1: Spectrum-to-SMILES (Transformer)

1. **Preprocessing**:
   Extract features (peaks) from the raw spectra and prepare training data.
   ```bash
   cd src/step1
   python preprocess.py --ir_data_dir ../../dataset/qme14s/IR_broaden --raman_data_dir ../../dataset/qme14s/Raman_broaden --out_path ./processed_data
   ```

2. **Train Transformer**:
   Train the model to translate spectral peaks to SMILES.
   ```bash
   python train_transformer.py --output_path ./processed_data 
   ```

3. **Prediction**:
   ```bash
   onmt_translate -model path_to_the_checkpoint -src ./processed_data/src-test.txt -n_best 10 -output ./pred.txt -gpu 0 -verbose
   ```

4. **Filtering & Evaluation**:
   Filter predictions and calculate metrics.
   ```bash
   # Filter top-10 predictions
   python filter10.py \
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

### Step 2: Conformation Generation (Diffusion)

1. **Preprocessing**:
   Prepare the dataset for the diffusion model.
   ```bash
   cd src/step2
   #process train dataset
   python preprocess.py \
       --ir_dir ../../dataset/qme14s/IR_broaden \
       --raman_dir ../../dataset/qme14s/Raman_broaden \
       --test_id_file train_indices.txt \
       --pred_smiles_file train_smiles.txt \
       --output_dir ./data
   #process test dataset
   python preprocess.py \
       --ir_dir ../../dataset/qme14s/IR_broaden \
       --raman_dir ../../dataset/qme14s/Raman_broaden \
       --test_id_file test_indices.txt \
       --pred_smiles_file result.txt \
       --output_dir ./data
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

## Running a Demo

We provide a demo to illustrate how to use the model. All demo data is available in the `example_data` directory.

### Step 1: Spectrum-to-SMILES (Transformer)

1. **Train Transformer**:

   ```bash
   cd src/step1

   python train_transformer.py --output_path ../../example_data/step1_data
   ```

2. **Prediction**:

   ```bash
   onmt_translate \
       -model path_to_the_checkpoint \
       -src ../../example_data/step1_data/src-test.txt \
       -n_best 10 \
       -output ./pred.txt \
       -gpu 0 \
       -verbose
   ```

3. **Filtering & Evaluation**:

   ```bash
   # Filter top-10 predictions
   python filter10.py \
       --input_file pred.txt \
       --output_file result.txt \
       --label_file ../../example_data/step1_data/tgt-test.txt

   # Calculate metrics
   python check_results.py \
       --label_file ../../example_data/step1_data/tgt-test.txt \
       --top10_file pred.txt \
       --top1_file result.txt \
       --output_csv results.csv \
       --report_file metrics_report.txt
   ```

### Step 2: Conformation Generation (Diffusion)

We provide preprocessed data for direct testing.

1. **Training**:

   ```bash
   cd src/step2

   python model/train.py \
       --config ../../../example_data/step2_data/config.yml \
       --logdir ./logs
   ```

2. **Evaluation**:

   ```bash
   # Run inference
   python model/test.py \
       path_to_the_checkpoint \
       --config ../../../example_data/step2_data/config.yml

   # Calculate RMSD
   python model/cal_rmsd.py \
       --result_pkl ./inference_results.pkl \
       --output_csv analysis_report.csv
   ```

## Directory Structure

```
specdiffmol_code/
├── data/               # Data storage (gitignored)
├── src/
│   ├── step1/          # Transformer (Spectrum -> SMILES)
│   │   ├── preprocess.py      # Data preprocessing
│   │   └── train_transformer.py       
│   └── step2/          # Diffusion (SMILES + Spectrum -> Conformation)
│       ├── model/      # Model definitions and training script
│   │   └── preprocess/ # Data preprocessing  
├── requirements.txt    # Python dependencies
└── README.md           # This file
```
## License

SpecDiffMol is released under the [MIT](LICENSE.txt) license.