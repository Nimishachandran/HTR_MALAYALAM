# Malayalam Handwritten Text Recognition (HTR)

A deep learning pipeline for recognizing Malayalam handwritten text, built with a custom **ResNeXt-50 + Deformable Convolutions + CTC + Attention CRNN** architecture. Trained on a cleaned and synthetically augmented Malayalam handwriting dataset.

---

## Results

| Metric | Value |
|---|---|
| Best Validation CER | **1.15%** |
| Best Validation WER | **5.30%** |
| Model Parameters | 11.22M |
| Training Epochs | 120 |
| Dataset (after cleaning) | 552 samples → 2,760 (5× augmentation) |

---

## Architecture

- **Backbone:** ResNeXt-50 with Deformable Convolutions for spatial feature extraction
- **Sequence Model:** Bidirectional LSTM (hidden size 256)
- **Decoder:** CTC loss + Attention mechanism (AttentionCRNN)
- **Vocabulary Size:** 68 Malayalam characters

---

## Data Pipeline

- Raw dataset: 999 rows from Kaggle
- Cleaning: removed multi-line (211), strikethrough (3), too-short (34), too-long (199) samples
- Final clean samples: **552**
- Augmentation (5×): TPS spatial transform, ink variation, character-aware augmentation
- Final training set: **2,484 samples** | Validation: **276 samples**

---

## Training Details

| Parameter | Value |
|---|---|
| Optimizer | Adam |
| Initial LR | 5e-4 (cosine annealing) |
| Batch Size | 32 |
| Image Size | 64 × 800 |
| Mixed Precision | Yes (AMP) |
| Framework | PyTorch |

---

## Project Structure

```
├── model7-htr-cnn-lstm-attention-predictions-ct.ipynb  # Full training + inference notebook
├── README.md
└── .gitignore
```

---

## How to Run

### 1. Install dependencies

```bash
pip install torch torchvision albumentations editdistance opencv-python pillow tqdm
```

### 2. Prepare your dataset

Place your Malayalam handwriting dataset (image + label CSV) in the expected path. The notebook uses a Kaggle dataset — update the path variables at the top of the notebook if running locally.

### 3. Train

Open the notebook and run all cells sequentially. The best model is saved automatically as `best_augmented_model.pt` whenever validation CER improves.

### 4. Inference on a single image

```python
checkpoint = torch.load('best_augmented_model.pt')
model = AttentionCRNN(vocab_size=len(checkpoint['vocab']), hidden_size=256)
model.load_state_dict(checkpoint['model'])
model.eval()
```

Pass any Malayalam handwritten image (resized to 64×800) through the model to get a transcription.

---

## Dataset

This model was trained on a Malayalam handwriting dataset from Kaggle. The dataset is **not included** in this repository. Update the data loading path in the notebook to point to your own copy.

---

## Notes

- The test-time CER on a single held-out image was ~38.9% — this is expected given the small dataset size and the significant variation in individual handwriting styles not seen during training.
- The 1.15% validation CER reflects performance on the validation split of the same distribution.
- Future improvements: larger dataset, pretrained Malayalam language model for beam search decoding.

---

## Author

Nimisha Chandran S R  
[LinkedIn](https://linkedin.com/in/YOUR-PROFILE) | [GitHub](https://github.com/Nimishachandran)
