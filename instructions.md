# Running Qu-Net on HPC using Slurm (CARC)

This guide outlines how to execute your QUnet code on the USC CARC high-performance cluster (HPC) using the SLURM job manager. It includes details about hyperparameters, model architecture, current job submission setup, and the ideal workflow using OnDemand with Jupyter notebooks.

---

## Project Structure

### `main_<model>.py`

This is the main Python script that orchestrates the training. It contains all the key hyperparameters and execution logic.

```python
# Example hyperparameters
IMAGE_SIZE = (192, 256)   # Input image dimensions
BATCH_SIZE = 32           # Batch size for training
EPOCHS = 20               # Total number of training epochs
k = 5                     # Fold count for cross-validation (if used)
```

### `models/`
This folder contains all model architecture definitions. For example:
qunet_4_1.py: Contains the definition for QUnet model with 4 qubits 1 filter.

Make sure the main_\<model\>.py imports from these correctly.


## Execution

- corresponding job file

```bash
#!/bin/bash

#SBATCH --job-name=qunet_4_1           # Descriptive job name
#SBATCH --output=qunet_4_1.out         # Stdout redirected to file
#SBATCH --ntasks=32                    # Total number of tasks (e.g., MPI processes)
#SBATCH --cpus-per-task=1              # Number of CPU cores per task
#SBATCH --time=2:00:00                 # Max time limit (HH:MM:SS)
#SBATCH --mem=32GB                     # Total memory allocation
#SBATCH --account=kalev_1005           # Your project account
#SBATCH --partition=epyc-64            # HPC partition (queue)
```
```
# Creating a virtual environment
mamba create --name <env_name>

# Activate virtual environment 
mamba activate <env_name>

# Install packages (see requirements.txt for the list of packages to be installed)
mamba install <pkg> (or pip install <pkg> - if <pkg> not available via conda package manager)
```    
For more details, see [USC CARC Documentation](https://www.carc.usc.edu/user-guides/hpc-systems/software/conda)
```
command line - 
module load conda

# Activate your Python virtual environment
mamba activate <env>

# Run the training script
sbatch <job_file>.job
```

## Pre-Submission Checklist
Before running your jobs, make sure:

- Paths in main_file.py are correct (especially dataset and output paths)
- Required Python packages are installed in your environment
- Output (logs, models) is saved to an accessible location (not just /tmp)
- The SLURM script requests appropriate compute and memory resources
- You’re using the correct partition, account, and optional GPU modules

---
---

## Improvements and Ideal setup

Code improvements 
- Migrate to PyTorch
- Include optimization like Jax, optax etc for quicker simulations
- organize better so that there is no need for separate main files
- use classes and objects to build abstraction layers

Ideal setup
- On Jupyter notebooks (CARC OnDemand)
- Automated execution scripts