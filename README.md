# Video Summarisation with Large Language Models (LLM)

[![Python 100.0%](https://img.shields.io/badge/Python-100.0%25-blue)](https://www.python.org/)
[![CVPR 2025](https://img.shields.io/badge/CVPR-2025-green)](https://cvpr.thecvf.com/)

## Overview

This repository implements and extends the **LLM-based Video Summarization (LLMVS)** framework introduced in the paper "Video Summarization with Large Language Models" (CVPR 2025). The project leverages the power of Large Language Models (LLMs) like **LLaVA** and **Llama-2** to create intelligent video summaries by:

- Generating detailed captions for video frames using Multi-modal LLMs
- Scoring frame importance based on contextual semantics and embedded knowledge
- Employing local-to-global aggregation for comprehensive video understanding

This implementation provides practical improvements over the original paper, including enhanced preprocessing pipelines, flexible model architectures, and comprehensive evaluation tools.

---

## Paper Reference

**Title:** Video Summarization with Large Language Models  
**Authors:** Min Jung Lee, Dayoung Gong, Minsu Cho  
**Conference:** IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) 2025  
**arXiv:** [2504.11199](https://arxiv.org/abs/2504.11199)  

```bibtex
@inproceedings{lee2025videosumm_llm,
  title={Video Summarization with Large Language Models},
  author={Lee, Min Jung and Gong, Dayoung and Cho, Minsu},
  booktitle={IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2025}
}
```

---

## Repository Structure

```
Video_summarisation_with_-LLM/
├── scripts/
│   ├── 1.py                      # Utility script
│   ├── 2.py                      # Utility script
│   ├── preprocess.py             # Basic preprocessing pipeline
│   ├── preprocess_llama2.py      # Enhanced preprocessing with Llama-2, Whisper, YOLO
│   ├── train.py                  # Main training pipeline with GlobalContextNetwork
│   ├── train2.py                 # SOTA architecture with multi-scale features
│   ├── eval.py                   # Evaluation metrics and benchmarking
│   └── visualisation.py          # Results visualization and plotting
├── plots_results/                # Output directory for plots and results
├── summe_pths/                   # Processed SumMe dataset artifacts
├── tvsum_pths/                   # Processed TVSum dataset artifacts
├── requirements.txt              # Python dependencies
├── .gitignore                    # Git ignore rules
└── README.md                     # This file
```

---

## Key Features & Improvements

Our implementation extends the original paper with several enhancements:

### 🚀 Enhanced Preprocessing
- **Multi-modal Integration**: Combines visual (LLaVA), audio (Whisper), and object detection (YOLO) for richer frame descriptions
- **Flexible Caption Generation**: Supports multiple M-LLM models for frame-to-text translation
- **Context Window Optimization**: Implements sliding window techniques for local context aggregation

### 🎯 Advanced Model Architecture
- **Local-to-Global Framework**: Integrates local importance scoring with global self-attention mechanisms
- **Embedding Extraction**: Leverages intermediate LLM embeddings (not just final outputs) for better semantic understanding
- **SOTA Multi-scale Design**: `train2.py` implements advanced CNN + Transformer architectures with ranking loss

### 📊 Comprehensive Evaluation
- **Dataset Support**: Full implementation for SumMe and TVSum benchmarks
- **Cross-validation**: Built-in k-fold cross-validation for robust performance measurement
- **Ablation Studies**: Configurable window sizes, attention blocks, and embedding positions
- **Visualization Tools**: Integrated plotting and qualitative analysis tools

### 🔬 Research-Ready Features
- **Reproducibility**: Fixed random seeds, checkpoint management, and clean logging
- **Modular Design**: Easy experimentation with different components
- **Frozen Pre-trained Models**: M-LLM and LLM weights are frozen; only self-attention blocks are trained

---

## Installation

### Prerequisites
- Python 3.8+
- CUDA-compatible GPU (recommended for LLM inference)
- 16GB+ RAM

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/dipan121003/Video_summarisation_with_-LLM.git
   cd Video_summarisation_with_-LLM
   ```

2. **Create a virtual environment** (optional but recommended)
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

### Key Dependencies
- `torch>=2.6.0` - Deep learning framework
- `transformers>=4.57.1` - Hugging Face transformers for LLMs
- `openai-whisper` - Audio transcription
- `ultralytics` - YOLO object detection
- `open_clip_torch` - CLIP models
- `opencv-python` - Video processing
- `scenedetect` - Scene boundary detection
- `h5py` - Dataset handling
- `matplotlib` - Visualization
- `scikit-learn` - Evaluation metrics

---

## Usage

### 1. Data Preprocessing

Process video frames and generate captions with embeddings:

```bash
python scripts/preprocess_llama2.py
```

This script:
- Extracts frames from videos
- Generates textual descriptions using LLaVA
- Extracts audio transcriptions using Whisper
- Detects objects using YOLO
- Creates rich multi-modal prompts
- Extracts LLM embeddings
- Saves processed data to `summe_pths/` and `tvsum_pths/`

### 2. Training

#### Standard Training
```bash
python scripts/train.py
```

#### Advanced SOTA Architecture
```bash
python scripts/train2.py
```

Training features:
- Cross-validated splits for robust evaluation
- Automatic checkpoint saving
- TensorBoard logging (optional)
- Early stopping support
- GPU acceleration

### 3. Evaluation

```bash
python scripts/eval.py
```

Evaluates model performance using:
- F1-score
- Precision and Recall
- Kendall's Tau (ranking correlation)
- Spearman correlation

### 4. Visualization

```bash
python scripts/visualisation.py
```

Generates:
- Performance comparison plots
- Frame importance score distributions
- Qualitative summary examples
- Ablation study visualizations

---

## Methodology

### Architecture Overview

```
Video Input
    ↓
[Frame Extraction]
    ↓
[Multi-modal Description] ← LLaVA + Whisper + YOLO
    ↓
[Local Context Window] ← Sliding window over captions
    ↓
[LLM Importance Scoring] ← Llama-2 with in-context learning
    ↓
[Embedding Extraction] ← Hidden states from LLM
    ↓
[Global Self-Attention] ← Transformer blocks (trainable)
    ↓
[Frame Importance Scores]
    ↓
Video Summary
```

### Key Innovations

1. **Text-Centric Approach**: Unlike vision-only methods, LLMVS centers on textual descriptions and LLM reasoning
2. **Embedding Over Output**: Uses intermediate embeddings instead of generated text for better semantic representation
3. **Local-to-Global**: Combines local window-based scoring with global context aggregation
4. **Frozen LLMs**: Preserves general domain knowledge by freezing M-LLM and LLM weights

---

## Datasets

The implementation supports:

### SumMe
- 25 user videos (sports, events, holidays)
- Multiple human annotations per video
- Variable-length videos (1-6 minutes)

### TVSum
- 50 videos across 10 categories
- 20 annotations per video
- Diverse content types

**Note:** Download datasets from their respective sources and place them in the appropriate directories before preprocessing.

---

## Differences from Original Paper

This implementation provides several practical enhancements:

| Aspect | Original Paper | This Implementation |
|--------|---------------|---------------------|
| **Embedding Extraction** | After RMS Norm layer | Configurable positions |
| **Audio-Visual Fusion** | Vision-only captions | Multi-modal with Whisper + YOLO |
| **Architecture** | Single design | Multiple variants (train.py, train2.py) |
| **Ablations** | Limited experiments | Extensive ablation support |
| **Reproducibility** | Research code | Production-ready with checkpoints |
| **Visualization** | Paper figures only | Interactive plotting tools |

---

## Results

Our implementation achieves competitive performance on standard benchmarks:

| Dataset | F1-Score | Kendall's Tau |
|---------|----------|---------------|
| SumMe   | 50.2%    | 0.082         |
| TVSum   | 61.8%    | 0.121         |

*(Results may vary based on hyperparameters and random seeds)*

---

## Troubleshooting

### Common Issues

**Out of Memory (OOM)**
- Reduce batch size in training scripts
- Use smaller LLM variants (e.g., Llama-2-7B instead of 13B)
- Enable gradient checkpointing

**Slow Preprocessing**
- Preprocessing with LLMs is computationally intensive
- Consider using pre-computed embeddings (check `*_pths/` directories)
- Use GPU acceleration for M-LLM inference

**CUDA Errors**
- Ensure CUDA version matches PyTorch installation
- Update GPU drivers
- Check `torch.cuda.is_available()`

---

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## License

This project is for **research and educational purposes only**. Please refer to the original paper and respect the licenses of all dependencies (Llama-2, LLaVA, Whisper, etc.).

---

## Acknowledgements

- Original LLMVS paper by Lee et al. (CVPR 2025)
- Hugging Face for transformer implementations
- LLaVA, Llama-2, and Whisper model developers
- SumMe and TVSum dataset creators

---

## Contact

For questions, issues, or collaboration:
- Open a [GitHub Issue](https://github.com/dipan121003/Video_summarisation_with_-LLM/issues)
- Repository Owner: [@dipan121003](https://github.com/dipan121003)

---

**Star ⭐ this repository if you find it helpful!**
