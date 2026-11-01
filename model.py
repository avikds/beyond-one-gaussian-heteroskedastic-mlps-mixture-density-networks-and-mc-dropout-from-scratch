"""
Beyond One Gaussian: Heteroskedastic MLPs, Mixture Density Networks and MC Dropout from Scratch

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - make_growing_noise_data
import numpy as np

def make_growing_noise_data(n, seed=0, gap=None):
    """
    Generate the growing-noise seasonal regression dataset
    from Figure 13.6.

    Parameters
    ----------
    n : int
        Number of data points to generate.
    seed : int, default=0
        Random seed for reproducibility.
    gap : tuple (a, b) or None, default=None
        Optional interval (a, b) to exclude from the input range.

    Returns
    -------
    x : np.ndarray
        Input values.
    y : np.ndarray
        Noisy target values.
    mu : np.ndarray
        True conditional mean.
    sigma : np.ndarray
        True conditional noise standard deviation.
    """

    if not isinstance(n, (int, np.integer)) or n < 0:
        raise ValueError("n must be a non-negative integer")

    # Use a dedicated random stream for this generator.
    rng = np.random.default_rng([seed, 1])

    # Step 1: Generate input values.
    if gap is None:
        # Draw n values uniformly from [-20, 60].
        x = rng.uniform(-20.0, 60.0, n)

    else:
        a, b = gap

        if a > b:
            raise ValueError("gap must be supplied as (a, b) with a <= b")

        if n > 0 and a <= -20.0 and b >= 60.0:
            raise ValueError("gap cannot cover the entire input range [-20, 60]")

        # Draw 2*n candidates and remove values inside the open interval (a, b).
        candidates = rng.uniform(-20.0, 60.0, 2 * n)

        x = candidates[~((candidates > a) & (candidates < b))][:n]

        # Replenish candidates if an unusually wide gap leaves fewer than n values.
        while x.size < n:
            candidates = rng.uniform(-20.0, 60.0, 2 * n)

            valid = candidates[~((candidates > a) & (candidates < b))]

            x = np.concatenate((x, valid[:n - x.size]))

    # Ensure inputs use float64.
    x = np.asarray(x, dtype=np.float64)

    # Step 2: Calculate the normalized input.
    g = (x + 20.0) / 80.0

    # Calculate the true conditional mean.
    mu = 5.0 * g + np.sin(2.0 * np.pi * x / 20.0)

    # Calculate the input-dependent noise standard deviation.
    sigma = 0.1 + 0.9 * g

    # Step 3: Generate noisy target values.
    y = mu + sigma * rng.standard_normal(n)

    # Return four float64 arrays.
    return (
        x.astype(np.float64, copy=False),
        y.astype(np.float64, copy=False),
        mu.astype(np.float64, copy=False),
        sigma.astype(np.float64, copy=False)
    )

# Step 2 - make_inverse_data
def forward_curve(y):
    """
    Compute the forward curve used in the inverse problem.

    Parameters
    ----------
    y : float or np.ndarray
        Target values.

    Returns
    -------
    float or np.ndarray
        Forward curve values.
    """
    return y + 0.3 * np.sin(2.0 * np.pi * y)


def make_inverse_data(n, seed=0):
    """
    Generate the one-to-many inverse regression dataset
    from Figure 13.23a.

    Parameters
    ----------
    n : int
        Number of data points to generate.
    seed : int, default=0
        Random seed for reproducibility.

    Returns
    -------
    x : np.ndarray
        Input values generated from the forward curve and noise.
    y : np.ndarray
        Target values sampled uniformly from [0, 1].
    """

    if not isinstance(n, (int, np.integer)) or n < 0:
        raise ValueError("n must be a non-negative integer")

    # Use a dedicated random stream for this generator.
    rng = np.random.default_rng([seed, 2])

    # Step 1: Draw target values uniformly from [0, 1].
    y = rng.uniform(0.0, 1.0, n)

    # Step 2: Generate uniform noise in [-0.1, 0.1].
    eta = rng.uniform(-0.1, 0.1, n)

    # Step 3: Compute the input values using the forward curve.
    x = forward_curve(y) + eta

    # Return two float64 arrays.
    return (
        np.asarray(x, dtype=np.float64),
        np.asarray(y, dtype=np.float64)
    )

# Step 3 - inverse_oracle_nll
def in_support(yhat, x):
    """
    Check whether predicted target values are inside the
    true conditional support of the inverse problem.

    A prediction is supported when:
        0 <= yhat <= 1
        abs(x - forward_curve(yhat)) < 0.1

    Parameters
    ----------
    yhat : np.ndarray
        Predicted target values.
    x : np.ndarray
        Corresponding input values.

    Returns
    -------
    np.ndarray
        Boolean array indicating whether each prediction
        lies inside the true conditional support.
    """

    yhat = np.asarray(yhat, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)

    return (
        (yhat >= 0.0) &
        (yhat <= 1.0) &
        (np.abs(x - forward_curve(yhat)) < 0.1)
    )


def support_size(x):
    """
    Estimate the total length of the true conditional support
    for each input value.

    The support is defined as:
        S(x) = {y in [0, 1] : abs(x - forward_curve(y)) < 0.1}

    The length is estimated using a grid of 20001 equally
    spaced points, processed in chunks of 500.

    Parameters
    ----------
    x : np.ndarray
        One-dimensional array of input values.

    Returns
    -------
    np.ndarray
        Estimated support lengths for each input value.
    """

    x = np.asarray(x, dtype=np.float64)

    # Create the evaluation grid over [0, 1].
    grid = np.linspace(0.0, 1.0, 20001)

    # Evaluate the forward curve once for the entire grid.
    curve_grid = forward_curve(grid)

    # Store the estimated support lengths.
    sizes = np.empty(len(x), dtype=np.float64)

    # Process inputs in chunks of 500 to limit memory usage.
    for start in range(0, len(x), 500):
        end = min(start + 500, len(x))

        x_chunk = x[start:end]

        # Check which grid points belong to the support.
        inside = (
            np.abs(x_chunk[:, None] - curve_grid[None, :]) < 0.1
        )

        # Estimate support length as the fraction of grid points
        # satisfying the support condition.
        sizes[start:end] = np.mean(inside, axis=1)

    return sizes


def inverse_oracle_nll(x):
    """
    Calculate the expected negative log-likelihood of the
    true conditional density for the inverse problem.

    Since Y | X=x is uniform over its conditional support,
    its negative log-likelihood is log(support_size(x)).

    Parameters
    ----------
    x : np.ndarray
        One-dimensional array of input values.

    Returns
    -------
    float
        Expected negative log-likelihood as a Python float.
    """

    # Estimate the conditional support lengths.
    sizes = support_size(x)

    # Calculate the mean negative log-likelihood.
    return float(np.mean(np.log(sizes)))

# Step 4 - Standardizer
class Standardizer:
    """
    Standardize numerical data using its mean and population standard deviation.

    Supports arrays of arbitrary shape and provides methods to transform
    values back to their original scale.
    """

    def __init__(self, v):
        # Store the mean and population standard deviation as Python floats.
        self.mean = float(np.mean(v))
        self.std = float(np.std(v))

    def forward(self, v):
        """
        Transform values into standardized units.

        Formula:
            z = (v - mean) / std
        """
        return (np.asarray(v) - self.mean) / self.std

    def inverse(self, z):
        """
        Transform standardized values back to their original units.

        Formula:
            v = z * std + mean
        """
        return np.asarray(z) * self.std + self.mean

    def scale_density(self, mu_z, sigma_z):
        """
        Convert Gaussian parameters from standardized units
        back to the original units.

        Parameters
        ----------
        mu_z : array-like
            Gaussian mean in standardized units.
        sigma_z : array-like
            Gaussian standard deviation in standardized units.

        Returns
        -------
        tuple
            (mu, sigma) in the original units.
        """

        # Transform the mean and scale the standard deviation.
        mu = self.inverse(mu_z)
        sigma = np.asarray(sigma_z) * self.std

        return mu, sigma

# Step 5 - softplus
from tinygrad import Tensor


def softplus(a):
    """
    Numerically stable Softplus activation.

    Computes:
        softplus(a) = log(1 + exp(a))

    Uses the equivalent stable expression:
        max(a, 0) + log(1 + exp(-abs(a)))

    This avoids overflow for large positive inputs.

    Parameters
    ----------
    a : Tensor
        Input tinygrad tensor.

    Returns
    -------
    Tensor
        Softplus activation applied element-wise.
    """

    return a.relu() + (1 + (-a.abs()).exp()).log()


def softplus_inv(s):
    """
    Numerically stable inverse Softplus.

    Computes:
        softplus_inv(s) = log(exp(s) - 1)

    Uses the equivalent expression:
        s + log(1 - exp(-s))

    The expm1 function improves numerical accuracy
    for very small positive values.

    Parameters
    ----------
    s : float or np.ndarray
        Positive input values.

    Returns
    -------
    np.ndarray or np.float64
        Inverse Softplus values in float64 precision.
    """

    s = np.asarray(s, dtype=np.float64)

    if np.any(s <= 0):
        raise ValueError("softplus_inv requires strictly positive inputs")

    return s + np.log(-np.expm1(-s))

# Step 6 - DenseNet
import math


class DenseNet:
    """
    A fully connected tanh neural network with optional dropout masks.

    Hidden layers use tanh activations.
    The output layer uses a linear activation.
    """

    def __init__(self, d_in, hidden, d_out, seed=0):
        """
        Initialize the neural network.

        Parameters
        ----------
        d_in : int
            Number of input features.
        hidden : list or tuple
            Widths of the hidden layers.
        d_out : int
            Number of output units.
        seed : int, default=0
            Random seed for reproducible initialization.
        """

        # Store hidden layer widths as a list.
        self.hidden = list(hidden)

        # Dedicated random generator for network initialization.
        rng = np.random.default_rng(seed)

        # Define the dimensions of all layers.
        dimensions = [d_in] + self.hidden + [d_out]

        # Store network parameters as (W, b) tuples.
        self.layers = []

        # Initialize each layer in order.
        for i in range(len(dimensions) - 1):

            a = dimensions[i]
            b = dimensions[i + 1]

            # Use different initialization scales for hidden
            # and output layers.
            if i < len(self.hidden):
                std = math.sqrt(1.0 / a)
            else:
                std = math.sqrt(0.1 / a)

            # Initialize weights using a normal distribution.
            W_array = rng.normal(
                loc=0.0,
                scale=std,
                size=(a, b)
            ).astype(np.float32)

            # Initialize biases to zero without random draws.
            b_array = np.zeros(b, dtype=np.float32)

            # Convert parameters into realized contiguous tensors.
            W = Tensor(W_array).contiguous().realize()
            bias = Tensor(b_array).contiguous().realize()

            # Store the weight and bias tensors.
            self.layers.append((W, bias))

    def parameters(self):
        """
        Return all trainable parameters in layer order.

        Returns
        -------
        list
            [W0, b0, W1, b1, ...]
        """

        return [
            parameter
            for W, b in self.layers
            for parameter in (W, b)
        ]

    def __call__(self, x, masks=None, scale=1.0):
        """
        Perform a forward pass through the network.

        Parameters
        ----------
        x : Tensor
            Input tinygrad tensor.
        masks : list or None, default=None
            Optional dropout masks for hidden layers.
            Each mask has shape (n, hidden_width).
        scale : float, default=1.0
            Scaling factor applied to hidden activations
            when dropout masks are not supplied.

        Returns
        -------
        Tensor
            Network predictions.
        """

        h = x

        # Process all hidden layers.
        for i, (W, b) in enumerate(self.layers[:-1]):

            # Linear transformation followed by tanh activation.
            h = (h @ W + b).tanh()

            # Apply per-example dropout masks when provided.
            if masks is not None:
                h = h * masks[i]

            # Otherwise, apply the specified scaling factor.
            elif scale != 1.0:
                h = h * scale

        # The final output layer is linear (no activation).
        W_out, b_out = self.layers[-1]

        return h @ W_out + b_out

    def np_forward(self, X, masks=None, scale=1.0):
        """
        Perform a forward pass using NumPy input arrays.

        Parameters
        ----------
        X : np.ndarray
            Two-dimensional input array.
        masks : list or None, default=None
            Optional list of NumPy dropout masks.
        scale : float, default=1.0
            Scaling factor for hidden activations.

        Returns
        -------
        np.ndarray
            Network predictions as a float64 NumPy array.
        """

        # Convert input data to a contiguous float32 Tensor.
        X_tensor = Tensor(
            np.asarray(X, dtype=np.float32)
        ).contiguous().realize()

        # Convert each NumPy dropout mask into a tinygrad Tensor.
        if masks is not None:
            masks_tensor = [
                Tensor(
                    np.asarray(mask, dtype=np.float32)
                ).contiguous().realize()
                for mask in masks
            ]
        else:
            masks_tensor = None

        # Execute the network's forward pass.
        output = self(
            X_tensor,
            masks=masks_tensor,
            scale=scale
        )

        # Return predictions as a float64 NumPy array.
        return output.numpy().astype(np.float64)

# Step 7 - Adam
class Adam:
    """
    Adam optimizer implemented from scratch using tinygrad tensors.

    Supports in-place parameter updates and bias correction.
    """

    def __init__(self, params, b1=0.9, b2=0.999, eps=1e-8):
        """
        Initialize the Adam optimizer.

        Parameters
        ----------
        params : list
            List of trainable tinygrad tensors.
        b1 : float, default=0.9
            Exponential decay rate for the first moment.
        b2 : float, default=0.999
            Exponential decay rate for the second moment.
        eps : float, default=1e-8
            Small constant for numerical stability.
        """

        # Store optimizer parameters.
        self.params = list(params)
        self.b1 = b1
        self.b2 = b2
        self.eps = eps

        # Initialize first-moment buffers with zeros.
        self.m = [
            Tensor.zeros(*p.shape).contiguous().realize()
            for p in self.params
        ]

        # Initialize second-moment buffers with zeros.
        self.v = [
            Tensor.zeros(*p.shape).contiguous().realize()
            for p in self.params
        ]

        # Initialize the optimization step counter.
        self.t = Tensor([0.0]).contiguous().realize()

    def step(self, grads, lr):
        """
        Perform one Adam optimization step.

        Parameters
        ----------
        grads : list
            Gradients corresponding to the trainable parameters.
        lr : Tensor
            One-element tensor containing the learning rate.

        Returns
        -------
        list
            [t, *params, *m, *v]

            The caller is responsible for realizing the returned tensors.
        """

        # Increment the optimization step counter.
        t = self.t.assign(self.t + 1)

        # Calculate bias-correction factors.
        bc1 = 1 - self.b1 ** t
        bc2 = 1 - self.b2 ** t

        # Update each parameter and its moment estimates.
        for p, g, m, v in zip(self.params, grads, self.m, self.v):

            # Update the first moment (mean of gradients).
            m.assign(
                self.b1 * m + (1 - self.b1) * g
            )

            # Update the second moment (mean of squared gradients).
            v.assign(
                self.b2 * v + (1 - self.b2) * g * g
            )

            # Apply the bias-corrected Adam parameter update.
            p.assign(
                p - lr[0] * (m / bc1) / ((v / bc2).sqrt() + self.eps)
            )

        # Return updated tensors without realizing them.
        return [self.t, *self.params, *self.m, *self.v]

# Step 8 - fit
from tinygrad import TinyJit, dtypes


def fit(params, loss_fn, X, Y, steps=1000, lr=0.01,
        mask_fn=None, seed=0):
    """
    Train a neural network using a TinyJit full-batch training loop.

    Parameters
    ----------
    params : list
        Trainable tinygrad tensors.
    loss_fn : callable
        Loss function with signature loss_fn(xb, yb, masks).
        Must return a scalar tinygrad Tensor.
    X : np.ndarray
        Input training data.
    Y : np.ndarray
        Target training data.
    steps : int, default=1000
        Number of optimization steps.
    lr : float, default=0.01
        Initial learning rate.
    mask_fn : callable or None, default=None
        Optional function mask_fn(rng, n) that generates
        fresh dropout masks for each training step.
    seed : int, default=0
        Random seed for dropout mask generation.

    Returns
    -------
    list
        Training history containing (step, loss) pairs.
        Records are saved every 500 steps and at the final step.
    """

    # Initialize the Adam optimizer.
    opt = Adam(params)

    # Convert input and target arrays to realized float32 tensors.
    # This conversion is performed only once.
    xb = Tensor(
        np.asarray(X, dtype=np.float32)
    ).contiguous().realize()

    yb = Tensor(
        np.asarray(Y, dtype=np.float32)
    ).contiguous().realize()

    # Dedicated random stream for generating dropout masks.
    rng = np.random.default_rng([seed, 3])

    # Define the inner training step.
    def step(xb, yb, lr_t, *masks):

        # Pass None when dropout is disabled.
        # Otherwise, pass the tuple of mask tensors.
        current_masks = tuple(masks) if mask_fn is not None else None

        # Compute the loss.
        loss = loss_fn(xb, yb, current_masks)

        # Calculate gradients with respect to all parameters.
        grads = loss.gradient(*params)

        # Update the model parameters using Adam.
        updated_tensors = opt.step(grads, lr_t)

        # Realize the loss, optimizer counter, parameters,
        # first moments, and second moments.
        Tensor.realize(loss, *updated_tensors)

        # Return the realized loss.
        return loss

    # Compile the training step using TinyJit.
    step = TinyJit(step)

    # Store training loss history.
    history = []

    # Full-batch training loop.
    for t in range(steps):

        # Calculate the cosine learning-rate schedule.
        lr_value = lr * 0.5 * (
            1.0 + math.cos(math.pi * t / steps)
        )

        # Convert learning rate into a one-element float32 Tensor.
        lr_t = Tensor(
            [lr_value],
            dtype=dtypes.float32
        ).contiguous().realize()

        # Generate fresh dropout masks for this training step.
        if mask_fn is not None:

            mask_arrays = mask_fn(rng, len(X))

            # Convert each NumPy mask into a realized float32 Tensor.
            mask_tensors = tuple(
                Tensor(
                    np.asarray(mask, dtype=np.float32)
                ).contiguous().realize()
                for mask in mask_arrays
            )

        else:
            mask_tensors = ()

        # Execute the TinyJit training step.
        loss = step(xb, yb, lr_t, *mask_tensors)

        # Record loss every 500 steps and at the final step.
        if t % 500 == 0 or t == steps - 1:
            history.append((int(t), float(loss.numpy())))

    return history

# Step 9 - mixture_logpdf
def mixture_logpdf(y, w, mu, sigma):
    """
    Compute the log probability density of a per-input Gaussian mixture.

    Parameters
    ----------
    y : np.ndarray
        Observations of shape (n,).

    w : np.ndarray
        Mixture weights of shape (n, K).
        Each row contains non-negative weights summing to 1.

    mu : np.ndarray
        Gaussian component means of shape (n, K).

    sigma : np.ndarray
        Gaussian component standard deviations of shape (n, K).

    Returns
    -------
    np.ndarray
        Log densities of shape (n,), with dtype float64.
    """

    # Convert all inputs to float64 for numerical accuracy.
    y = np.asarray(y, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)

    # Calculate log mixture weights.
    # Zero weights correspond to log probability = -inf.
    # This removes those components without producing NaN values.
    log_w = np.full(w.shape, -np.inf, dtype=np.float64)

    np.log(w, out=log_w, where=(w > 0))

    # Calculate standardized residuals for each Gaussian component.
    z = (y[:, None] - mu) / sigma

    # Calculate the log density of each weighted Gaussian component.
    log_components = (
        log_w
        - np.log(sigma)
        - 0.5 * np.log(2.0 * np.pi)
        - 0.5 * z**2
    )

    # Numerically stable Log-Sum-Exp calculation.
    # Subtract the maximum before exponentiation to avoid underflow.
    max_log = np.max(log_components, axis=1, keepdims=True)

    log_density = (
        max_log[:, 0]
        + np.log(
            np.sum(
                np.exp(log_components - max_log),
                axis=1
            )
        )
    )

    return log_density.astype(np.float64, copy=False)

# Step 10 - pit_values
from scipy.special import ndtr


def pit_values(y, w, mu, sigma):
    """
    Calculate the Probability Integral Transform (PIT) values
    for observations under Gaussian mixture distributions.

    Parameters
    ----------
    y : np.ndarray
        Observations of shape (n,).

    w : np.ndarray
        Mixture weights of shape (n, K).

    mu : np.ndarray
        Gaussian component means of shape (n, K).

    sigma : np.ndarray
        Gaussian component standard deviations of shape (n, K).

    Returns
    -------
    np.ndarray
        Predictive CDF values of shape (n,), with dtype float64.
    """

    # Convert inputs to float64 arrays.
    y = np.asarray(y, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)

    # Calculate the standardized residual for every component.
    z = (y[:, None] - mu) / sigma

    # Evaluate the standard normal CDF for every component.
    component_cdf = ndtr(z)

    # Calculate the weighted sum of component CDFs.
    pit = np.sum(w * component_cdf, axis=1)

    return pit.astype(np.float64, copy=False)


def pit_ks(p):
    """
    Calculate the Kolmogorov-Smirnov distance between
    the empirical PIT distribution and Uniform(0, 1).

    Parameters
    ----------
    p : np.ndarray
        One-dimensional array of PIT values.

    Returns
    -------
    float
        Kolmogorov-Smirnov statistic as a Python float.
    """

    # Sort PIT values in ascending order.
    p = np.sort(np.asarray(p, dtype=np.float64).ravel())

    n = len(p)

    # Handle an empty input safely.
    if n == 0:
        return 0.0

    # Construct the empirical CDF upper and lower bounds.
    i = np.arange(1, n + 1, dtype=np.float64)

    upper = i / n
    lower = (i - 1) / n

    # Calculate the maximum distance in both directions.
    d_plus = upper - p
    d_minus = p - lower

    # Return the largest KS deviation.
    return float(max(np.max(d_plus), np.max(d_minus)))

