"""
Main script for Bangla BERT Hate Speech Detection
Runs K-fold cross-validation training for binary classification
"""

from transformers import AutoTokenizer
import torch
from config import parse_arguments, print_config
from data import load_and_preprocess_data, prepare_kfold_splits
from train import run_kfold_training
from utils import set_seed


def main():
    # Parse arguments
    config = parse_arguments()
    print_config(config)

    # Set random seed for reproducibility
    set_seed(config.seed)

    # Load tokenizer
    print(f"Loading tokenizer from {config.model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(config.model_path)
    print("Tokenizer loaded successfully.")

    # Load and preprocess data
    print(f"Loading data from {config.dataset_path}...")
    comments, labels = load_and_preprocess_data(config.dataset_path)
    print(f"Data loaded: {len(comments)} samples.")

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")

    # Run K-fold training
    run_kfold_training(config, comments, labels, tokenizer, device)


if __name__ == "__main__":
    main()
