# Antibody_Models

This repo shares **code and instructions** to use our antibody models.  
**Weights are hosted on the Hugging Face Hub** to keep this repo small.

- ESM2 (Unpaired) weights: **MahTala/antibody-esm2-unpaired** (private)

## Setup

```bash
# create env (Linux)
./env/create_env.sh
conda activate py310_protein_ab

# if the HF model is private, authenticate once:
huggingface-cli login
```
```bash
### Usage (Embeddings)
python utils/load_model.py
# prints: Embedding shape: (1, H)
```
> The script loads EsmModel (no pooling), tokenizes a sample heavy chain,
> runs a forward pass, and mean-pools token embeddings to one vector.

### Notes
- Weights are not stored in this GitHub repo. They are pulled from the Hub at runtime.
- To switch the model to public, toggle visibility on the Hugging Face model page later.