import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt

# Read CSV
train_df = pd.read_csv("train_avalanche_data.csv")
test_df = pd.read_csv("test_avalanche_data.csv")

# Define some metaparameters
num_features = 20 + 1 # Add a bias term
weights = np.zeros(num_features)

# First 20 columns are features, column 21 is labels
X_train_raw = train_df.iloc[:, :20].values
y_train = train_df.iloc[:, 20].values

X_test_raw = test_df.iloc[:, :20].values
y_test = test_df.iloc[:, 20].values

# Bias column
N_train = X_train_raw.shape[0]
N_test  = X_test_raw.shape[0]

X_train = np.hstack([np.ones((N_train, 1)), X_train_raw])
X_test  = np.hstack([np.ones((N_test, 1)),  X_test_raw])

# Define a helper sigmoid function
def sigmoid(z):
    z = np.clip(z, -50, 50)
    return 1.0 / (1.0 + np.exp(-z))

def train_log_reg(X, y, n_steps=2000, step_size=0.01):
    N, D = X.shape # dimensions of X, our data
    w = np.zeros(D) # Get a weight vector of zeroes

    for step in range(n_steps):
        z = X @ w # our theta = features times weights
        p = sigmoid(z) # plug into sigmoid to get p
        error = y - p      # check error

        grad = X.T @ error # calculate gradient
        w += step_size * grad # update weights

    return w # final weights

w_hat = train_log_reg(X_train, y_train)
p_test = sigmoid(X_test @ w_hat)
y_pred = (p_test >= 0.5).astype(int)

## Now get distribution of p with bootstrapping
n_boot = 300
N_train, D = X_train.shape

boot_weights = np.zeros((n_boot, D))  # each row = one theta sample

for b in range(n_boot):
    # sample indices with replacement
    idx = np.random.randint(0, N_train, size=N_train)
    Xb = X_train[idx]
    yb = y_train[idx]

    wb = train_log_reg(Xb, yb, n_steps=2000, step_size=0.01)
    boot_weights[b, :] = wb

# Now boot_weights[b] is the weight vector from bootstrap run b
w_mean = boot_weights.mean(axis=0)
w_std  = boot_weights.std(axis=0)

## Test on a sample day:
x_star_raw = np.array([
    0,  # surface_hoar
    0,  # depth_hoar
    1,  # slope_gt_30
    1,  # slope_gt_40
    0,
    0,
    0,
    1,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    1,
    0,
    0,
    0
])

# Add bias
x_star = np.concatenate(([1.0], x_star_raw))
this_theta = logits = boot_weights @ x_star
p_boot = sigmoid(this_theta)

mu = p_boot.mean()
sigma = p_boot.std()

xs = np.linspace(0, 1, 400)
gauss = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((xs - mu) / sigma)**2)

# draw it out
plt.figure()
plt.hist(p_boot, bins=20, range=(0,1), density=True, alpha=0.4, label="Bootstrap histogram")  # counts, not density

# Smooth Gaussian curve
plt.plot(xs, gauss, linewidth=2, label="Gaussian fit")

plt.xlabel("Avalanche probability p")
plt.ylabel("Density")
plt.title("Bootstrap distribution of avalanche probability (smoothed)")
plt.xlim(0,1)

# Add MLE for comparison
p_mle = sigmoid(w_hat @ x_star)
plt.axvline(p_mle, color="black", linestyle="--", linewidth=2,
            label=f"MLE p = {p_mle:.2f}")
plt.legend()
plt.show()
