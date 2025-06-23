<img width="1200" alt="dropkan_explained" src="https://github.com/Ghaith81/dropkan/blob/master/DropKAN_explained.JPG">

# Dropout Kolmogorov-Arnold Networks (DropKAN)

This is the GitHub repository for the papers:  
["DropKAN: Regularizing KANs by masking post-activations"](https://arxiv.org/abs/2407.13044) and  
["Rethinking the Function of Neurons in KANs"](https://arxiv.org/abs/2407.20667).

DropKAN operates by randomly masking some of the post-activations within the KANs computation graph, while scaling-up the retained post-activations.

---

## How to use DropKAN

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
---

## Example: Training on the Wine Quality Dataset

We provide an example notebook demonstrating DropKAN and KAN training on the Wine Quality dataset from the UCI repository. Key steps include:

- Loading and splitting the dataset
- Scaling features with StandardScaler
- Converting data to PyTorch tensors
- Training KAN, KAN + Dropout, and DropKAN models with fixed dropout rate (`0.1`)
- Evaluating models with Mean Absolute Error (MAE)
- Visualizing performance using box plots comparing test MAE across models

```python
# Initialize results DataFrame and training steps
log_df = pd.DataFrame(columns=['drop_rate', 'mode', 'mae'])
epochs = 10
batch = 32
steps = int(len(X_train) / batch) * epochs

# Train plain KAN (no dropout)
for j in range(5):
    model = DropKAN(seed=j, width=[X_train.shape[1], X_train.shape[1]*2, 1])
    model.train(dataset, opt="Adam", steps=steps, batch=32, lr=0.01, loss_fn=torch.nn.L1Loss())
    set_training_mode(model, False)
    y_pred = model(dataset['test_input']).detach().numpy()
    mae = mean_absolute_error(y_test, y_pred)
    log_df.loc[len(log_df)] = [0.0, 'KAN', mae]

# Train KAN + Dropout and DropKAN with drop_rate=0.2
drop_rate = 0.2
for j in range(5):
    # KAN + Dropout
    model = DropKAN(seed=j, width=[X_train.shape[1], X_train.shape[1]*2, 1], drop_rate=drop_rate, drop_mode='dropout')
    model.train(dataset, opt="Adam", steps=steps, batch=32, lr=0.01, loss_fn=torch.nn.L1Loss())
    set_training_mode(model, False)
    y_pred = model(dataset['test_input']).detach().numpy()
    mae = mean_absolute_error(y_test, y_pred)
    log_df.loc[len(log_df)] = [drop_rate, 'KAN + Dropout', mae]

    # DropKAN
    model = DropKAN(seed=j, width=[X_train.shape[1], X_train.shape[1]*2, 1], drop_rate=drop_rate, drop_mode='postact')
    model.train(dataset, opt="Adam", steps=steps, batch=32, lr=0.01, loss_fn=torch.nn.L1Loss())
    set_training_mode(model, False)
    y_pred = model(dataset['test_input']).detach().numpy()
    mae = mean_absolute_error(y_test, y_pred)
    log_df.loc[len(log_df)] = [drop_rate, 'DropKAN', mae]
