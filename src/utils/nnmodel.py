# Copyright (C) H.R. Oosterhuis 2022.
# Distributed under the MIT License (see the accompanying README.md and LICENSE files).
import torch.nn as nn

class TorchModel(nn.Module):
    def __init__(self, input_dim, hidden_units_list, use_sigmoid_out=True):
        super(TorchModel, self).__init__()
        layers = []
        current_dim = input_dim
        for hidden_units in hidden_units_list:
            layers.append(nn.Linear(current_dim, hidden_units))
            layers.append(nn.ReLU())
            current_dim = hidden_units

        layers.append(nn.Linear(current_dim, 1))
        if use_sigmoid_out:
            layers.append(nn.Sigmoid())

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


def init_torch_model(model_params):
    # Assuming num_features is available or can be passed via model_params
    # This needs to be determined from the data before model initialization
    # For now, let's expect it in model_params
    input_dim = model_params.get("input_dim")
    if input_dim is None:
        raise ValueError(
            "model_params must contain 'input_dim' for PyTorch model initialization"
        )

    hidden_units = model_params["hidden units"]

    # PyTorch models are typically float32 by default.
    # If float64 is strictly needed, you can call .double() on the model
    # and ensure input tensors are also float64.
    model = TorchModel(input_dim, hidden_units)
    return model
