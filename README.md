# Antibody_Models

This repo shares **code and instructions** to use our antibody models.  
**Weights are hosted on the Hugging Face Hub** to keep this repo small.

- ESM2 (Unpaired) weights: **MahTala/antibody-esm2-unpaired** (private)
- ESM2 weights: **MahTala/antibody-esm2-paired** (private)
- ESM C weights: **MahTala/antibody-esmc-paired** (private)

## Setup

```bash
# create env (Linux)
./env/create_env.sh
conda activate py310_protein_ab

# if the HF model is private, authenticate once:
huggingface-cli login
```
```bash
### Usage (ESM2)
python utils/load_model_unpaired_ESM2.py
# prints: Embedding shape: (1, H)
```
> The script loads EsmModel (no pooling), tokenizes a sample heavy chain,
> runs a forward pass, and mean-pools token embeddings to one vector.


```bash
### Usage (ESMC)
python utils/load_model_paired_ESMC.py
```



### Notes
- Weights are not stored in this GitHub repo. They are pulled from the Hub at runtime.
