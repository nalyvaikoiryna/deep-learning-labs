from sklearn.datasets import load_iris
import numpy as np
import torch
import torch.nn as nn
import argparse

def backward(X, y, Z_1, A_1, W_2, softmax, buggy=False):
    N = len(y)

    dZ2 = softmax.copy()
    dZ2[np.arange(N), y] -= 1

    if not buggy:
        dZ2 /= N

    dW2 = A_1.T @ dZ2
    db2 = np.sum(dZ2, axis=0)

    dA1 = dZ2 @ W_2.T
    dZ1 = dA1 * (Z_1 > 0)

    dW1 = X.T @ dZ1
    db1 = np.sum(dZ1, axis=0)

    return dW1, db1, dW2, db2

parser = argparse.ArgumentParser()

parser.add_argument(
    "--buggy",
    action="store_true"
)

args = parser.parse_args()

def compute_loss(X, y, W1, b1, W2, b2):
    Z1 = X @ W1 + b1
    A1 = np.maximum(0, Z1)
    Z2 = A1 @ W2 + b2

    shifted = Z2 - np.max(Z2, axis=1, keepdims=True)

    log_probs = shifted - np.log(
        np.sum(np.exp(shifted), axis=1, keepdims=True)
    )

    return -np.mean(
        log_probs[np.arange(len(y)), y]
    )




iris = load_iris()

X = iris.data.astype(np.float64)
y = iris.target


rng = np.random.default_rng(0)

train_indices = []
test_indices = []

for class_id in [0, 1, 2]:
    indices = np.flatnonzero(y == class_id)

    rng.shuffle(indices)

    train_indices.extend(indices[:35])
    test_indices.extend(indices[35:])


train_indices = np.array(train_indices)
test_indices = np.array(test_indices)


X_train = X[train_indices]
y_train = y[train_indices]

X_test = X[test_indices]
y_test = y[test_indices]


mean = X_train.mean(axis=0)
std = X_train.std(axis=0, ddof=0)

X_train = (X_train - mean) / std
X_test = (X_test - mean) / std

rng = np.random.default_rng(0)

# numpy implementation
W_1 = rng.normal(
    loc=0.0,
    scale=np.sqrt(2 / 4),
    size=(4, 8)
)

b_1 = np.zeros((8,), dtype=np.float64)

W_2 = rng.normal(
    loc=0.0,
    scale=np.sqrt(2 / (8 + 3)),
    size=(8, 3)
)


b_2 = np.zeros((3,), dtype=np.float64)

Z_1 = X_train @ W_1 + b_1
A_1 = np.maximum(0, Z_1)

Z_2 = A_1 @ W_2 + b_2





loss_numpy = compute_loss(X, y, W_1, b_1, W_2, b_2)
shifted = Z_2 - np.max(Z_2, axis=1, keepdims=True)
exp_shifted = np.exp(shifted)
softmax = exp_shifted / np.sum(
    exp_shifted,
    axis=1,
    keepdims=True
)
# print(loss_numpy)


# numpy grads
dW1, db1, dW2, db2 = backward(
        X_train,
        y_train,
        Z_1,
        A_1,
        W_2,
        softmax,
        buggy=args.buggy
    )

# Torch implementation
X_torch = torch.tensor(X_train, dtype=torch.float64)
y_torch = torch.tensor(y_train, dtype=torch.long)
model = nn.Sequential(
    nn.Linear(4, 8),
    nn.ReLU(),
    nn.Linear(8, 3)
)
model = model.double()
with torch.no_grad():
    model[0].weight.copy_(
        torch.tensor(W_1.T, dtype=torch.float64)
    )

    model[0].bias.copy_(
        torch.tensor(b_1, dtype=torch.float64)
    )

    model[2].weight.copy_(
        torch.tensor(W_2.T, dtype=torch.float64)
    )

    model[2].bias.copy_(
        torch.tensor(b_2, dtype=torch.float64)
    )

logits = model(X_torch)
loss_torch = nn.functional.cross_entropy(logits, y_torch)



#torch grads
loss_torch.backward()
dW1_torch = model[0].weight.grad.detach().numpy().T
db1_torch = model[0].bias.grad.detach().numpy()

dW2_torch = model[2].weight.grad.detach().numpy().T
db2_torch = model[2].bias.grad.detach().numpy()



# difference check 
diff_loss = abs(loss_numpy - loss_torch)

diff_W1 = np.max(np.abs(dW1 - dW1_torch))
diff_b1 = np.max(np.abs(db1 - db1_torch))

diff_W2 = np.max(np.abs(dW2 - dW2_torch))
diff_b2 = np.max(np.abs(db2 - db2_torch))

print("----------------NumPy and PyTorch grads comparison:")
print("NumPy loss:", loss_numpy)
print("PyTorch loss:", loss_torch)

print("Loss diff:", diff_loss)
print("W1 diff:", diff_W1)
print("b1 diff:", diff_b1)
print("W2 diff:", diff_W2)
print("b2 diff:", diff_b2)

print("Loss passed:", diff_loss <= 1e-12)
print("W1 passed:", diff_W1 <= 1e-12)
print("b1 passed:", diff_b1 <= 1e-12)
print("W2 passed:", diff_W2 <= 1e-12)
print("b2 passed:", diff_b2 <= 1e-12)

#numereical grads

to_check = [
    (W_1, (0, 0)),
    (b_1, (0,)),
    (W_2, (0, 0)),
    (b_2, (0,))
]
numerical_grads = []
eps = 1e-6

for param, idx in to_check:
    original = param[idx]

    param[idx] = original + eps
    loss_plus = compute_loss(
        X_train, y_train,
        W_1, b_1, W_2, b_2
    )

    param[idx] = original - eps
    loss_minus = compute_loss(
        X_train, y_train,
        W_1, b_1, W_2, b_2
    )

    param[idx] = original

    grad_num = (loss_plus - loss_minus) / (2 * eps)
    numerical_grads.append(grad_num)

print("-----------------Numerical and manual gradients comparison:")
print("manual dW1:", dW1[0, 0])
print("numerical dW1:", numerical_grads[0])
print("diff dW1:", abs(dW1[0, 0] - numerical_grads[0]))
print("manual db1:", db1[0])
print("numerical db1:", numerical_grads[1])
print("diff db1:", abs(db1[0] - numerical_grads[1]))
print("manual dW2:", dW2[0, 0])
print("numerical dW2:", numerical_grads[2])
print("diff dW2:", abs(dW2[0, 0] - numerical_grads[2]))
print("manual db2:", db2[0])
print("numerical db2:", numerical_grads[3])
print("W1[0, 0] passed:", diff_W1 <= 1e-7)
print("b1[0] passed:", diff_b1 <= 1e-7)
print("W2[0, 0] passed:", diff_W2 <= 1e-7)
print("b2[0] passed:", diff_b2 <= 1e-7)