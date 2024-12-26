# Music Generation with an LSTM RNN (PyTorch)

This repository contains code to train a recurrent neural network (RNN) in PyTorch to generate Irish folk songs in [ABC notation](https://en.wikipedia.org/wiki/ABC_notation). The dataset and supporting functions are provided via [MIT Deep Learning's `mitdeeplearning`](https://github.com/aamini/mitdeeplearning) package.

## Table of Contents
1. [Overview](#overview)
2. [Features and Objectives](#features-and-objectives)
3. [Prerequisites and Installation](#prerequisites-and-installation)
4. [Usage Instructions](#usage-instructions)
5. [Code Walkthrough](#code-walkthrough)
6. [Project Structure](#project-structure)
7. [Tips for Improvement](#tips-for-improvement)
8. [License](#license)

---

## Overview

This code:
- Downloads a dataset of Irish folk songs in ABC notation.
- Vectorizes each character, mapping them to integer indices and back.
- Defines an LSTM-based RNN model in PyTorch to predict the next character in the sequence.
- Trains the model with a cross-entropy loss and an Adam optimizer.
- Generates new melodies in ABC notation from the trained model.

## Features and Objectives

- **Data Preprocessing**: We concatenate all songs, create a character vocabulary, and convert characters to integer indices.
- **LSTM Model**: An embedding layer, followed by an LSTM layer, followed by a fully connected layer (`nn.Linear`).
- **Training Procedure**:
  - Custom training loop with batching.
  - Use of `nn.CrossEntropyLoss`.
  - Optimizer steps (`torch.optim.Adam`).
- **Sequence Generation**: Sampling from the softmax distribution of the model's last layer to predict characters one by one.

## Prerequisites and Installation

1. **Python 3.7+** (Most recent Python 3 versions should work).
2. **pip** for installing Python packages (you may also use `conda` if preferred).

### Required Python Packages

- `torch` >= 1.0
- `numpy`
- `tqdm`
- `mitdeeplearning` (for dataset loading)

You can install them with:
```bash
pip install torch tqdm numpy mitdeeplearning
