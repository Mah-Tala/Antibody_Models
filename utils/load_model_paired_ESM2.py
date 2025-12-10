#!/usr/bin/env python3
import os, torch
from transformers import EsmModel, AutoTokenizer

# Hugging Face repo id (your private model)
REPO_ID = "MahTala/AbCDR-ESM2"

SEP_TOKEN = "-"  # Separator between heavy and light chains

# If private: either run `huggingface-cli login` once OR set HF_TOKEN in env.
hf_token = os.getenv("HF_TOKEN", None)

# 1) Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(REPO_ID, token=hf_token)
model = EsmModel.from_pretrained(REPO_ID, add_pooling_layer=False, token=hf_token).eval()

# 2) Example paired antibody sequence
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

# 3) Tokenize and forward
inputs = tokenizer(paired_sequence, return_tensors="pt", add_special_tokens=True)
with torch.no_grad():
    last_hidden = model(**inputs).last_hidden_state  # (1, L, H)

# 4) Mean pool to one embedding
mask = inputs["attention_mask"].unsqueeze(-1)       # (1, L, 1)
emb  = (last_hidden * mask).sum(1) / mask.sum(1)    # (1, H)

print("Embedding shape:", emb.shape)
print("First 5 dims:", emb[0, :5])
