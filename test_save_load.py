import torch
import os
import shutil
from model import TransformerBinaryClassifier
from transformers import AutoTokenizer

def test_save_load():
    model_name = "sagorsarker/bangla-bert-base"
    save_dir = "./test_model_save"
    
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
        
    print(f"Initializing model {model_name}...")
    model = TransformerBinaryClassifier(model_name)
    model.eval()
    
    # Create dummy input
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    text = "আমি তোমাকে ভালোবাসি"
    inputs = tokenizer(text, return_tensors="pt")
    
    print("Getting original model output...")
    with torch.no_grad():
        original_output = model(inputs["input_ids"], attention_mask=inputs["attention_mask"])
    
    print(f"Saving model to {save_dir}...")
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    
    print(f"Loading model from {save_dir}...")
    loaded_model = TransformerBinaryClassifier.from_pretrained(save_dir)
    loaded_model.eval()
    
    print("Getting loaded model output...")
    with torch.no_grad():
        loaded_output = loaded_model(inputs["input_ids"], attention_mask=inputs["attention_mask"])
    
    # Compare outputs
    diff = torch.abs(original_output["logits"] - loaded_output["logits"]).max().item()
    print(f"Max difference in logits: {diff}")
    
    if diff < 1e-5:
        print("✓ Verification SUCCESS: Loaded model outputs match original model outputs.")
    else:
        print("✗ Verification FAILED: Loaded model outputs do not match original model outputs.")
        
    # Clean up
    # shutil.rmtree(save_dir)

if __name__ == "__main__":
    test_save_load()
