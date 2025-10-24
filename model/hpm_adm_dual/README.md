# Dual Model Training Script

This script trains a dual-channel model that combines:
- **HPM (Human Propagation Model)**: Predicts retweet intensity based on text content and emotion features
- **ADM (AI Detection Model)**: Classifies veracity (true/false) of information

## Model Architecture

The dual model consists of:
1. **RoBERTa base model** for text encoding
2. **LSTM layer** for HPM task (propagation prediction)
3. **Linear layer** for ADM task (veracity classification)
4. **Emotion features** from emotion RoBERTa model integrated into both channels

## Dataset

- **Input**: `text_clean` column from PHEME dataset
- **HPM Label**: `retweet_intensity` column (regression task)
- **ADM Label**: `veracity_label` column (binary classification, excluding 'unverified')

## Usage

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Test the Setup

```bash
python test_train.py
```

### 3. Run Training

```bash
python train.py
```

## Training Configuration

- **Batch Size**: 16
- **Learning Rate**: 2e-5
- **Max Epochs**: 10
- **Max Text Length**: 128 tokens
- **Early Stopping**: 5 epochs patience
- **Device**: Auto-detect (CUDA/CPU)

## Outputs

The training script will:
1. Save the best model to `best_model.pth`
2. Generate training curves plot (`training_curves.png`)
3. Print evaluation metrics for validation and test sets

## Model Performance Metrics

- **HPM**: Mean Squared Error (MSE) for retweet intensity prediction
- **ADM**: Area Under Curve (AUC) for veracity classification

## File Structure

```
model/hpm_adm_dual/
├── train.py              # Main training script
├── model.py              # Dual model architecture
├── test_train.py         # Test script
├── requirements.txt      # Dependencies
├── README.md            # This file
├── best_model.pth       # Saved model (after training)
└── training_curves.png  # Training plots (after training)
```

## Notes

- The script automatically filters out 'unverified' samples for ADM task
- Emotion features are extracted using the emotion RoBERTa model
- Training uses stratified splitting to maintain label distribution
- The model combines both tasks with weighted loss functions
