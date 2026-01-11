"""
Generic Transformer-based Binary Classifier for Hate Speech Detection
Supports any transformer model (BERT, RoBERTa, etc.) through AutoModel
"""

import os
import json
import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig


class TransformerBinaryClassifier(nn.Module):
    """
    Transformer-based binary classifier for hate speech detection.
    """

    def __init__(self, model_name, dropout=0.1):
        """
        Initialize the binary classifier.

        Args:
            model_name (str): Name or path of pre-trained transformer model
            dropout (float): Dropout rate for regularization
        """
        super(TransformerBinaryClassifier, self).__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        config = AutoConfig.from_pretrained(model_name)
        hidden_size = config.hidden_size

        # Classification head for binary classification
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1)  # Single output for binary classification
        )
        self.config = config
        self.model_name = model_name
        self.dropout = dropout

    def forward(self, input_ids, attention_mask=None, labels=None):
        """
        Forward pass of the model.

        Args:
            input_ids: Token IDs from tokenizer
            attention_mask: Attention mask for padding
            labels: Ground truth labels (optional, for loss calculation)

        Returns:
            dict: Dictionary containing loss (if labels provided) and logits
        """
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]
        logits = self.classifier(cls_output)  # Shape: (batch_size, 1)

        loss = None
        if labels is not None:
            labels = labels.view(-1, 1)  # Reshape to (batch_size, 1)
            loss_fct = nn.BCEWithLogitsLoss()
            loss = loss_fct(logits, labels)

        return {'loss': loss, 'logits': logits}


    def save_pretrained(self, save_directory):
        """
        Save the model and its configuration to a directory.
        """
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)

        # Save the encoder
        self.encoder.save_pretrained(save_directory)

        # Save the classifier state dict
        classifier_path = os.path.join(save_directory, "classifier.pt")
        torch.save(self.classifier.state_dict(), classifier_path)

        # Save custom configuration
        custom_config = {
            "model_name": self.model_name,
            "dropout": self.dropout,
            "hidden_size": self.config.hidden_size
        }
        config_path = os.path.join(save_directory, "classifier_config.json")
        with open(config_path, "w") as f:
            json.dump(custom_config, f, indent=4)

        print(f"Model saved to {save_directory}")


    @classmethod
    def from_pretrained(cls, load_directory):
        """
        Load the model from a directory.
        """
        config_path = os.path.join(load_directory, "classifier_config.json")
        with open(config_path, "r") as f:
            custom_config = json.load(f)

        model = cls(custom_config["model_name"], dropout=custom_config["dropout"])
        
        # Load encoder
        model.encoder = AutoModel.from_pretrained(load_directory)
        
        # Load classifier
        classifier_path = os.path.join(load_directory, "classifier.pt")
        model.classifier.load_state_dict(torch.load(classifier_path, map_location='cpu'))
        
        return model


    def freeze_base_layers(self):
        """
        Freeze encoder parameters for feature extraction.
        """
        for param in self.encoder.parameters():
            param.requires_grad = False

        frozen_params = sum(p.numel() for p in self.encoder.parameters())
        total_params = sum(p.numel() for p in self.parameters())
        print(f"Frozen {frozen_params:,} parameters out of {total_params:,} total parameters")
        print(f"Trainable parameters: {total_params - frozen_params:,}")

    def unfreeze_base_layers(self):
        """
        Unfreeze encoder parameters for fine-tuning.
        """
        for param in self.encoder.parameters():
            param.requires_grad = True

        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"All parameters unfrozen. Trainable parameters: {trainable_params:,}")
