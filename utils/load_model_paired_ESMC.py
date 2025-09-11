#!/usr/bin/env python3
"""
Load fine-tuned ESMC model for paired antibody chains from Hugging Face Hub.

This script loads a fine-tuned ESMC model specifically trained on paired 
heavy and light antibody chains. The model expects sequences in the format:
heavy_chain + separator + light_chain

Requirements:
    pip install torch safetensors huggingface_hub esm
"""

import os
import torch
from huggingface_hub import login, hf_hub_download
from esm.tokenization import get_esmc_model_tokenizers
from esm.models.esmc import ESMC
from safetensors import safe_open
from esm.sdk.api import ESMProtein, LogitsConfig

# Configuration
REPO_ID = "MahTala/antibody-esmc-paired"  # Hugging Face repo with fine-tuned weights
SEP_TOKEN = "-"  # Separator between heavy and light chains

def load_paired_esmc_model(repo_id=REPO_ID, device=None):
    """
    Load fine-tuned ESMC model for paired antibody chains.
    
    Args:
        repo_id: Hugging Face repository ID containing the fine-tuned weights
        device: Device to load model on ('cuda' or 'cpu'). Auto-detects if None.
    
    Returns:
        model: Fine-tuned ESMC model
        tokenizer: ESMC tokenizer
    """
    # Set device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load tokenizer and base model
    print("Loading base ESMC model and tokenizer...")
    tokenizer = get_esmc_model_tokenizers()
    model = ESMC.from_pretrained("esmc_600m").to(device)
    
    # Download fine-tuned weights from Hugging Face Hub
    print(f"Downloading fine-tuned weights from {repo_id}...")
    
    # Download safetensors file (will cache locally after first download)
    # Requires HF_TOKEN environment variable or login() for private repos
    local_ckpt_path = hf_hub_download(
        repo_id=repo_id,
        filename="model.safetensors",
        token=os.getenv("HF_TOKEN", None)
    )
    
    print(f"Loading weights from {local_ckpt_path}...")
    
    # Load safetensors weights
    original_state_dict = {}
    with safe_open(local_ckpt_path, framework="pt") as sf:
        for key in sf.keys():
            original_state_dict[key] = sf.get_tensor(key)
    
    # Remove "esmC_model." prefix from keys to match base model structure
    renamed_state_dict = {}
    for key, value in original_state_dict.items():
        new_key = key.replace("esmC_model.", "") if key.startswith("esmC_model.") else key
        renamed_state_dict[new_key] = value
    
    # Load fine-tuned weights into model
    print("Loading fine-tuned weights into base model...")
    missing_keys, unexpected_keys = model.load_state_dict(
        renamed_state_dict,
        strict=False
    )
    
    # Report loading status
    if missing_keys:
        print(f" ⚠️  Missing keys in checkpoint: {missing_keys}")
    if unexpected_keys:
        print(f" ⚠️  Unexpected keys in checkpoint: {unexpected_keys}")
    if not missing_keys and not unexpected_keys:
        print(" ✅ All weights loaded successfully!")
    
    print("Model ready for inference!")
    return model, tokenizer


def main():
    """
    Example usage of the paired ESMC model with antibody sequences.
    """
    # For private repos, set HF_TOKEN environment variable or use login()
    # Option 1: export HF_TOKEN="your_token_here" 
    # Option 2: Uncomment the line below for interactive login
    # login()
    
    # Load model
    model, tokenizer = load_paired_esmc_model()
    device = next(model.parameters()).device
    
    # Example paired antibody sequence
    # Heavy chain from therapeutic antibody
    h_chain = (
        "EVQLVESGGGLVQPGGSLRLSCAASGFTFSSYAMSWVRQAPGKGLEWVAVISYDGSNKYYADSVKGRF"
        "TISADTSKNTAYLQMNSLRAEDTAVYYCAREGYYGSSYWYFDYWGQGTLVTVSS"
    )
    
    # Light chain from therapeutic antibody  
    l_chain = (
        "DIQMTQSPSSLSASVGDRVTITCRASQSISSYLNWYQQKPGKAPKLLIYAASSLQSGVPSRFSGSGS"
        "GTDFTLTISSLQPEDFATYYCQQSYSTPLTFGGGTKVEIK"
    )
    
    # Combine chains with separator (model expects this format)
    paired_sequence = f"{h_chain}{SEP_TOKEN}{l_chain}"
    
    print("\n" + "="*60)
    print("PAIRED ANTIBODY SEQUENCE ANALYSIS")
    print("="*60)
    print(f"Heavy chain length: {len(h_chain)} residues")
    print(f"Light chain length: {len(l_chain)} residues")
    print(f"Total sequence length: {len(paired_sequence)} (including separator)")
    
    # =========================================================================
    # Approach 1: Using ESMProtein API (high-level interface)
    # =========================================================================
    print("\n" + "="*60)
    print("Approach 1: ESMProtein API (High-level)")
    print("="*60)
    
    # Create protein object and encode
    protein = ESMProtein(sequence=paired_sequence)
    protein_tensor = model.encode(protein)
    
    # Get logits and embeddings
    logits_output = model.logits(
        protein_tensor,
        LogitsConfig(sequence=True, return_embeddings=True)
    )
    
    # Extract tensors from output objects
    logits_api = logits_output.logits.sequence  # Extract tensor from ForwardTrackData
    embeddings_api = logits_output.embeddings
    
    print(f"Logits shape: {logits_api.shape}")
    print(f"Embeddings shape: {embeddings_api.shape}")
    print(f"Embeddings dtype: {embeddings_api.dtype}")
    
    # =========================================================================
    # Approach 2: Direct tokenization (low-level interface)
    # =========================================================================
    print("\n" + "="*60)
    print("Approach 2: Direct Tokenization (Low-level)")
    print("="*60)
    
    # Tokenize sequence
    seq_encoded = tokenizer(paired_sequence, return_tensors="pt")
    seq_input_ids = seq_encoded["input_ids"].to(device)
    
    # Forward pass
    with torch.no_grad():
        outputs = model(sequence_tokens=seq_input_ids)
    
    logits_direct = outputs.sequence_logits
    embeddings_direct = outputs.embeddings
    
    print(f"Logits shape: {logits_direct.shape}")
    print(f"Embeddings shape: {embeddings_direct.shape}")
    print(f"Embeddings dtype: {embeddings_direct.dtype}")
    
    # =========================================================================
    # Verify both approaches produce same results
    # =========================================================================
    print("\n" + "="*60)
    print("VERIFICATION: Comparing Both Approaches")
    print("="*60)
    
    # Compare logits (both should be bfloat16)
    logits_match = torch.equal(logits_api, logits_direct)
    print(f"Logits identical: {logits_match}")
    
    # Compare embeddings (API returns float32, direct returns bfloat16)
    # Convert to same dtype for comparison
    embeddings_match = torch.allclose(
        embeddings_api.to(embeddings_direct.dtype), 
        embeddings_direct, 
        atol=1e-3
    )
    print(f"Embeddings match (after dtype alignment): {embeddings_match}")
    
    # =========================================================================
    # Example: Extract features for downstream tasks
    # =========================================================================
    print("\n" + "="*60)
    print("EXTRACTED FEATURES FOR DOWNSTREAM TASKS")
    print("="*60)
    
    # Mean pooling over sequence length for fixed-size representation
    sequence_representation = embeddings_direct.mean(dim=1)  # [batch_size, hidden_dim]
    print(f"Sequence representation shape: {sequence_representation.shape}")
    
    # Get embeddings at separator position (interface between chains)
    separator_pos = len(h_chain)  # Position of separator token
    interface_embedding = embeddings_direct[0, separator_pos, :]
    print(f"Interface embedding shape: {interface_embedding.shape}")
    
    # Example: Process multiple sequences in batch
    print("\n" + "="*60)
    print("BATCH PROCESSING EXAMPLE")
    print("="*60)
    
    # Multiple paired sequences
    sequences = [
        f"{h_chain}{SEP_TOKEN}{l_chain}",  # Original
        f"{h_chain[:50]}{SEP_TOKEN}{l_chain[:50]}",  # Truncated version
    ]
    
    # Tokenize batch with padding
    batch_encoded = tokenizer(sequences, return_tensors="pt", padding=True)
    batch_input_ids = batch_encoded["input_ids"].to(device)
    batch_attention_mask = batch_encoded["attention_mask"].to(device)
    
    # Process batch
    with torch.no_grad():
        batch_outputs = model(sequence_tokens=batch_input_ids)
    
    print(f"Batch size: {len(sequences)}")
    print(f"Batch embeddings shape: {batch_outputs.embeddings.shape}")
    print(f"Batch logits shape: {batch_outputs.sequence_logits.shape}")
    
    print("\n✅ All inference examples completed successfully!")


if __name__ == "__main__":
    main()