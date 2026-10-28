 <div align="center">
  <a href="https://freeimage.host/i/n8C69ZN">
    <img src="https://iili.io/n8C69ZN.md.png" alt="IronFist1 Banner" width="100%">
  </a>
</div>

<h1 align="center">IronFist1</h1>

<p align="center">
  <strong>A General-Purpose Language Model Built from Scratch</strong>
</p>

<p align="center">
  Exploring language modeling, transformer architectures, and resilient reasoning.
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-Deep%20Learning-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch">
  <img src="https://img.shields.io/badge/Parameters-500M-blueviolet?style=for-the-badge" alt="500M parameters">
  <img src="https://img.shields.io/badge/Status-Development-orange?style=for-the-badge" alt="In development">
</p>

---

## Overview

**IronFist1** is an experimental, general-purpose large language model (LLM) being developed from scratch with Python and PyTorch.

The project aims to build a complete language-modeling system, including its own tokenizer, transformer architecture, training pipeline, and experimental reasoning infrastructure.

Rather than simply fine-tuning an existing pretrained language model, IronFist1 focuses on developing the underlying components needed to train and experiment with a custom model.

The initial target is a **500-million-parameter model**, with an emphasis on modular architecture, reliable implementation, and future experimentation.

## Objectives

- Develop a general-purpose autoregressive language model.
- Implement a custom WordPiece tokenizer.
- Build transformer components using PyTorch.
- Create scalable training and data-loading infrastructure.
- Experiment with rStar-inspired reasoning and search.
- Develop policy and discriminator models for reasoning experiments.
- Maintain a comprehensive automated test suite.
- Establish a foundation for future model improvements.

## Architecture

IronFist1 is organized into several major subsystems.

### 1. Language Model

The core language model uses transformer-based components to process token sequences and predict subsequent tokens.

Key components include:

- Token embeddings.
- Positional and rotary positional representations.
- Causal self-attention.
- Feed-forward neural networks.
- Normalization layers.
- Transformer blocks.
- Language-model output projections.

The architecture is designed to support modular experimentation and future improvements.

### 2. Custom Tokenizer

IronFist1 uses a custom **WordPiece tokenizer** to convert text into token IDs suitable for model training and inference.

Tokenizer functionality includes:

- Training a vocabulary from text datasets.
- Encoding text into token sequences.
- Decoding token sequences back into text.
- Managing vocabulary and tokenizer configuration.
- Preparing data for language-model training.

The current tokenizer vocabulary target is 32,000 tokens.

### 3. rStar Reasoning

The project explores rStar-inspired reasoning through search-based approaches, including Monte Carlo Tree Search (MCTS).

The reasoning infrastructure is intended to investigate how candidate solutions can be explored, evaluated, and improved.

Experimental components include:

- Search-tree expansion.
- State evaluation.
- Candidate reasoning paths.
- Reward calculation.
- Policy-based guidance.
- Discriminator-based evaluation.
- Consistency scoring.

These components are experimental and should not be interpreted as evidence of human-level reasoning or verified improvements in model performance.

### 4. Training Infrastructure

The training system is designed to support large text datasets and GPU-accelerated model development.

Planned and developing capabilities include:

- Dataset preprocessing.
- Parquet dataset support.
- Tokenized data shards.
- Sequence-based batching.
- Training and validation loops.
- Checkpoint management.
- Loss and performance evaluation.

## Technology Stack

| Technology | Purpose |
|---|---|
| Python | Primary development language |
| PyTorch | Neural network implementation and training |
| WordPiece | Tokenization |
| NumPy | Numerical operations and token shards |
| Parquet | Dataset ingestion |
| CUDA | GPU acceleration on supported NVIDIA hardware |
| MCTS | Experimental search and reasoning |
| unittest | Automated testing |

## Project Structure

The repository is organized into modular components. The exact structure may evolve as development continues.

```text
IronFist1/
├── config/
│   └── model_config.py
├── data/
│   └── ...
├── model/
│   ├── attention.py
│   ├── transformer.py
│   ├── policy_model.py
│   └── ...
├── tokenizer/
│   ├── tokenizer.py
│   ├── train_tokenizer.py
│   └── vocab/
├── rstar/
│   └── ...
├── tests/
│   ├── test_attention.py
│   ├── test_transformer.py
│   ├── test_policy_model.py
│   ├── test_discriminator.py
│   └── ...
├── train.py
├── README.md
└── .gitignore
```

*This is an illustrative overview, not a guarantee that every listed path exists in every revision.*

## Getting Started

### Requirements

- Python 3.10 or newer.
- PyTorch.
- NumPy.
- A compatible NVIDIA GPU is recommended for training at the intended scale.
- Sufficient RAM, GPU memory, and storage for your dataset and model configuration.

### 1. Clone the repository

Replace the placeholder with the actual GitHub repository URL.

```bash
git clone https://github.com/YOUR_USERNAME/IronFist1.git
cd IronFist1
```

### 2. Create a virtual environment

**Windows PowerShell**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**Linux / macOS**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

Install the appropriate PyTorch build for your hardware from the official [PyTorch installation guide](https://pytorch.org/get-started/locally/).

Then install the project's required dependencies, if a dependency file is available:

```bash
pip install -r requirements.txt
```

If `requirements.txt` has not been created yet, install the dependencies required by the current implementation.

### 4. Run the tests

From the repository root:

```bash
python -m unittest discover -s tests -v
```

Run individual test modules when debugging a specific subsystem:

```bash
python -m unittest tests.test_attention -v
python -m unittest tests.test_transformer -v
python -m unittest tests.test_policy_model -v
```

### 5. Train the model

Training commands depend on the current training entry point and configuration. Once the training pipeline is configured, use the repository's documented training command.

**Note:** A 500M-parameter model can require substantial GPU memory, particularly during training when gradients, optimizer states, and activations are included.

## Development Roadmap

- [x] Establish the initial project structure.
- [x] Implement and test core attention components.
- [x] Develop normalization and embedding components.
- [x] Train the initial WordPiece tokenizer.
- [x] Establish experimental search and reasoning components.
- [ ] Complete integration of all model components.
- [ ] Resolve outstanding policy-model and discriminator tests.
- [ ] Finalize the training and validation pipeline.
- [ ] Establish reproducible training benchmarks.
- [ ] Evaluate language-model quality on held-out data.
- [ ] Improve reasoning evaluation and reward calibration.
- [ ] Document inference, checkpoints, and deployment.

The roadmap reflects development goals, not a claim that the model is already fully trained or production-ready.

## Testing and Reliability

IronFist1 uses automated tests to verify individual components and catch regressions during development.

Testing focuses on:

- Tensor shapes and dimensions.
- Attention masking and causal behavior.
- Gradient propagation.
- Numerical stability.
- Invalid input handling.
- Transformer integration.
- Policy and discriminator behavior.
- Search and reward calculation.

Passing unit tests establishes implementation-level correctness for the tested cases; it does not by itself establish language-model quality or reasoning capability.

## Contributing

Contributions, experiments, bug reports, and architectural suggestions are welcome.

When contributing:

1. Keep changes modular and consistent with existing code.
2. Add or update tests for modified functionality.
3. Run relevant tests before submitting changes.
4. Document new configuration options and dependencies.
5. Include reproducible examples when reporting bugs.

For substantial architectural changes, consider opening an issue or discussion before implementation.

## Current Status

IronFist1 is an **experimental research and development project**.

The project is building toward a custom 500M-parameter language model, with tokenizer development, transformer components, testing, and reasoning infrastructure forming the foundation.

Training results, benchmark scores, and model capabilities should be reported only when they have been measured and documented.

## License

No license has been specified yet. Until a license is added to the repository, the project's reuse and redistribution terms remain unspecified.

## Acknowledgments

IronFist1 builds on the broader open-source machine-learning ecosystem, particularly PyTorch and the research community studying transformer architectures, language modeling, and search-based reasoning.

---

<p align="center">
  <strong>IronFist1 — Built from the Ground Up.</strong>
</p>

<p align="center">
  Research. Experiment. Engineer.
</p>
