
---

# CNN Training Framework with PyTorch

This project provides a command‑line interface (CLI) for training and evaluating a convolutional neural network model (**CNN1**) using PyTorch.

## Features
- **Flexible modes:** supports full training or evaluation‑only mode.  
- **GPU support:** automatically uses CUDA when available for faster computation.  
- **Dynamic configuration:** adjust learning rate, weight decay, and number of epochs directly from the CLI.

---

## Installation

**Prerequisite:** ensure you have Python 3.12+ installed.

Follow these steps to set up your development environment:

1. **Create a Conda environment:** this is done only once
   ```bash
   conda create --name aims_cv python=3.12
   ```

2. **Activate the environment:** this is needed whenever you need to use your environment
   ```bash
   conda activate aims_cv
   ```

3. **Install dependencies:** this is done once, unless the requirements change.

   Make sure your `requirements.txt` file is available, then run:
   ```bash
   pip install -r requirements.txt
   ```

4. **Add the environment to Jupyter:**
   To use this environment in Jupyter Notebooks:
   ```bash
   conda install -c anaconda ipykernel
   python -m ipykernel install --user --name=aims_cv
   ```

---

## Usage

The main script uses `argparse` to handle execution options.

### 1. Train the model
Run a standard training session for 10 epochs using the GPU (or CPU if GPU not available):
```bash
python main.py --mode train --epochs 10 --lr 0.001 --cuda
```

### 2. Evaluate the model

The project provides two ways to evaluate the CNN model: through the main script or via a Jupyter Notebook.

#### Using the evaluation script
Once your trained model is saved as `model.pth`, run:
```bash
python main.py --mode eval --cuda
```

---