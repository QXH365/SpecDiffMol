import pandas as pd
import os
import argparse

def main():
    parser = argparse.ArgumentParser(description="Analyze RMSD results from CSV report.")
    parser.add_argument("file_path", help="Path to the analysis report CSV file")
    args = parser.parse_args()

    file_path = args.file_path

    try:
        if not os.path.exists(file_path):
            print(f"Error: File {file_path} not found.")
        else:
            df = pd.read_csv(file_path)

            # Ensure RMSD column is numeric
            df['RMSD Total (Å)'] = pd.to_numeric(df['RMSD Total (Å)'], errors='coerce')

            # Filter based on SMILES Correctness
            correct_smiles = df[df['SMILES Correct'] == True]
            incorrect_smiles = df[df['SMILES Correct'] == False]

            # Calculate average RMSD
            avg_rmsd_correct = correct_smiles['RMSD Total (Å)'].mean()
            avg_rmsd_incorrect = incorrect_smiles['RMSD Total (Å)'].mean()

            # Print results
            print(f"Number of samples with correct SMILES: {len(correct_smiles)}")
            print(f"Average RMSD for correct SMILES: {avg_rmsd_correct:.4f} Å" if not pd.isna(avg_rmsd_correct) else "Average RMSD for correct SMILES: No Data")

            print(f"\nNumber of samples with incorrect SMILES: {len(incorrect_smiles)}")
            print(f"Average RMSD for incorrect SMILES: {avg_rmsd_incorrect:.4f} Å" if not pd.isna(avg_rmsd_incorrect) else "Average RMSD for incorrect SMILES: No Data")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()