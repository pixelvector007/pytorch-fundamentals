# PyTorch Workflow Fundamentals

A hands-on collection of notebooks covering the fundamentals of building, training, evaluating, saving, and loading machine learning models with **PyTorch**.

The notebooks start with a linear regression model built using `nn.Parameter` and then move to PyTorch's built-in `nn.Linear` layer.

## 📚 Notebooks

### 01 — PyTorch Workflow Fundamentals

**File:** `01_pytorch_workflow_fundamentals.ipynb`

Introduces the complete PyTorch workflow using a simple linear regression problem.

Topics covered:

- Creating and working with tensors
- Creating synthetic data
- Training and testing data
- Data visualization
- Building a custom model with `nn.Module`
- Using `nn.Parameter`
- Forward propagation
- Loss functions
- Optimizers
- Backpropagation
- Training loops
- `model.train()` and `model.eval()`
- `torch.inference_mode()`
- Training and testing loss
- Saving and loading model weights
- `state_dict()`
- Making predictions with a trained model

### 02 — PyTorch Workflow with `nn.Linear`

**File:** `02_pytorch_workflow_with_nn_linear.ipynb`

Builds on the first notebook by replacing manually created parameters with PyTorch's built-in `nn.Linear` layer.

Topics covered:

- `nn.Linear`
- `in_features` and `out_features`
- Model parameters
- Device selection
- Training and evaluation
- Loss and optimizer
- Training loops
- `model.train()` and `model.eval()`
- `torch.inference_mode()`
- Saving and loading model weights
- Making predictions

## 🔄 PyTorch Workflow

Both notebooks follow the core machine learning workflow:

```text
Create Data
    ↓
Split Data
    ↓
Visualize Data
    ↓
Build Model
    ↓
Make Predictions
    ↓
Calculate Loss
    ↓
Backpropagation
    ↓
Optimizer Step
    ↓
Evaluate Model
    ↓
Save Model
    ↓
Load Model
    ↓
Inference
```

## 🧠 Models

The first notebook creates the linear regression model manually:

```python
class LinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.weights = nn.Parameter(torch.randn(1))
        self.bias = nn.Parameter(torch.randn(1))

    def forward(self, x):
        return self.weights * x + self.bias
```

The second notebook uses PyTorch's built-in linear layer:

```python
class LinearRegressionModelV2(nn.Module):
    def __init__(self):
        super().__init__()

        self.linear_layer = nn.Linear(
            in_features=1,
            out_features=1
        )

    def forward(self, x):
        return self.linear_layer(x)
```

This progression helps demonstrate how PyTorch moves from manually defined learnable parameters to reusable neural network layers.

## 🛠️ Technologies

- Python
- PyTorch
- NumPy
- Matplotlib

## 🎯 Purpose

The goal of this repository is to build a strong understanding of the **PyTorch workflow** before moving on to more complex neural networks and deep learning projects.

This serves as a foundation for future work in **Deep Learning, Computer Vision, and Artificial Intelligence**.

## 📂 Repository Structure

```text
.
├── 01_pytorch_workflow_fundamentals.ipynb
├── 02_pytorch_workflow_with_nn_linear.ipynb
└── README.md
```

---

⭐ More PyTorch projects and experiments will be added as I continue learning **Machine Learning, Deep Learning, and Artificial Intelligence**.

Made with ❤️ by Arpit Kushwaha
