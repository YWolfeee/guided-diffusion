# FACE-id

## prepare

1. download arcface model from https://github.com/ronghuaiyang/arcface-pytorch/issues/61
2. download celeba-hq-256 from https://www.kaggle.com/datasets/badasstechie/celebahq-resized-256x256?resource=download

## util functions

1. faceid.load_arcface() : load arcface model
2. faceid.arcface_forward() : input sampled image data, output arcface feature (n, 512).
3. faceid.cosine(x, y) : faceid feature similarity between two images. Used as (classifier) guidance
4. ddpm.py now has examplar code for uncond sampling.

I only checked the data processing pipeline is correct. we also need to check arcface accuracy (maybe find some evaluation cases).

NEXT: merge the model sampling into our code base.
- create a new sampling script.
- reimplement the sampling methods.
