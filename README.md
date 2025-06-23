<img width="1200" alt="dropkan_explained" src="https://github.com/Ghaith81/dropkan/figures/DropKAN_explained.JPG">

This is the GitHub repository for the papers: ["DropKAN: Regularizing KANs by masking post-activations"](https://arxiv.org/abs/2407.13044) and ["Rethinking the Function of Neurons in KANs"](https://arxiv.org/abs/2407.20667). 

# Dropout Kolmogorov-Arnold Networks (DropKAN) 

DropKAN operates by randomly masking some of the post-activations within the KANs computation graph, while scaling-up the retained post-activations.

## How to use DropKAN

The DropKAN model can be used similarly to KAN to create a model composed of DropKANLayers. Three main parameters control dropout behavior:

- **drop_rate**: Either a single float or a list of floats specifying drop rates per layer.  
  - If a single float is provided (e.g., `0.1`), it will be applied uniformly to all layers.  
  - If a list is provided, its length must match the number of layers minus one (e.g., `[0.1, 0.2]` for a 3-layer model), specifying drop rates for each layer transition.

- **drop_mode**: Drop mask application method. Options include:  
  - `'postspline'`: Drop mask applied to the layer's postsplines  
  - `'postact'` (default): Drop mask applied to the layer's postactivations  
  - `'dropout'`: Standard dropout applied to inputs

- **drop_scale**: Boolean to scale retained activations by `1/(1 - drop_rate)`. Default: `True`.

## Citation

```python
@article{altarabichi2024dropkan,
  title={DropKAN: Regularizing KANs by masking post-activations},
  author={Altarabichi, Mohammed Ghaith},
  journal={arXiv preprint arXiv:2407.13044},
  year={2024}
}
