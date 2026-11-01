"""
Beyond One Gaussian: Heteroskedastic MLPs, Mixture Density Networks and MC Dropout from Scratch

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - make_growing_noise_data
import numpy as np
import time

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
from tinygrad import Tensor, TinyJit, dtypes

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
    """

    def __init__(self, params, b1=0.9, b2=0.999, eps=1e-8):
        # Store trainable parameters and optimizer settings.
        self.params = list(params)
        self.b1 = b1
        self.b2 = b2
        self.eps = eps

        # Initialize first and second moment estimates.
        self.m = [
            Tensor.zeros(*p.shape).contiguous().realize()
            for p in self.params
        ]

        self.v = [
            Tensor.zeros(*p.shape).contiguous().realize()
            for p in self.params
        ]

        # Initialize optimization counter.
        self.t = Tensor([0.0]).contiguous().realize()

    def step(self, grads, lr):
        """
        Perform one Adam optimization step.

        Parameters
        ----------
        grads : Tensor or list of Tensor
            Gradients corresponding to parameters.
        lr : Tensor
            Learning rate supplied on every call.

        Returns
        -------
        list
            [t, *params, *m, *v]
        """

        # Support both a single gradient and a list of gradients.
        if isinstance(grads, Tensor):
            grads = [grads]
        else:
            grads = list(grads)

        if len(grads) != len(self.params):
            raise ValueError(
                "Number of gradients must match number of parameters"
            )

        # Convert a numeric learning rate to a Tensor if necessary.
        if not isinstance(lr, Tensor):
            lr = Tensor([float(lr)]).contiguous().realize()

        # Increment the step counter.
        self.t.assign(self.t + 1.0).realize()

        # Calculate bias-correction factors.
        bc1 = 1.0 - self.b1 ** self.t
        bc2 = 1.0 - self.b2 ** self.t

        # Update each parameter and its moment estimates.
        for p, g, m, v in zip(
            self.params,
            grads,
            self.m,
            self.v
        ):

            # Realize the gradient before modifying the parameter.
            g.realize()

            # Update first moment.
            m.assign(
                self.b1 * m + (1.0 - self.b1) * g
            ).realize()

            # Update second moment.
            v.assign(
                self.b2 * v + (1.0 - self.b2) * (g * g)
            ).realize()

            # Bias-corrected moment estimates.
            m_hat = m / bc1
            v_hat = v / bc2

            # Calculate the Adam update.
            update = m_hat / (v_hat.sqrt() + self.eps)

            # Detach the parameter before its in-place update.
            # Use the learning rate supplied to this call.
            p.assign(
                p.detach() - lr * update
            ).realize()

        # Return counter, parameters, and both moment estimates.
        return [
            self.t,
            *self.params,
            *self.m,
            *self.v
        ]

# Step 8 - fit
def fit(params, loss_fn, X, Y, steps=1000, lr=0.01,
        mask_fn=None, seed=0):

    opt = Adam(params)

    xb = Tensor(
        np.asarray(X, dtype=np.float32)
    ).contiguous().realize()

    yb = Tensor(
        np.asarray(Y, dtype=np.float32)
    ).contiguous().realize()

    rng = np.random.default_rng([seed, 3])
    history = []

    # Use TinyJit only when there are no dropout masks.
    if mask_fn is None:

        def gradient_step(xb, yb):
            loss = loss_fn(xb, yb, None)
            grads = loss.gradient(*params)

            for g in grads:
                g.realize()

            loss.realize()
            return loss, *grads

        gradient_step = TinyJit(gradient_step)

    for t in range(steps):

        lr_value = lr * 0.5 * (
            1.0 + math.cos(math.pi * t / steps)
        )

        lr_t = Tensor(
            [lr_value],
            dtype=dtypes.float32,
        ).contiguous().realize()

        if mask_fn is not None:
            mask_arrays = mask_fn(rng, len(X))

            masks = tuple(
                Tensor(
                    np.asarray(mask, dtype=np.float32)
                ).contiguous().realize()
                for mask in mask_arrays
            )

            # Eager forward and backward pass for dropout.
            loss = loss_fn(xb, yb, masks)
            grads = loss.gradient(*params)

            for g in grads:
                g.realize()

            loss.realize()

        else:
            outputs = gradient_step(xb, yb)
            loss = outputs[0]
            grads = outputs[1:]

        opt.step(grads, lr_t)

        if t % 500 == 0 or t == steps - 1:
            history.append(
                (int(t), float(loss.numpy()))
            )

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

# Step 11 - interval_report
def mixture_quantile(q, w, mu, sigma, iters=60):
    """
    Calculate per-input quantiles of Gaussian mixture distributions
    using vectorized bisection on the mixture CDF.

    Parameters
    ----------
    q : float
        Desired quantile, strictly between 0 and 1.
    w : np.ndarray
        Mixture weights of shape (n, K).
    mu : np.ndarray
        Gaussian component means of shape (n, K).
    sigma : np.ndarray
        Gaussian component standard deviations of shape (n, K).
    iters : int, default=60
        Number of bisection iterations.

    Returns
    -------
    np.ndarray
        Quantile values of shape (n,), with dtype float64.
    """

    # Convert inputs to float64 for numerical precision.
    w = np.asarray(w, dtype=np.float64)
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)

    q = float(q)

    if not 0.0 < q < 1.0:
        raise ValueError("q must be strictly between 0 and 1")

    # Establish initial lower and upper brackets for each input.
    lo = np.min(mu - 10.0 * sigma, axis=1)
    hi = np.max(mu + 10.0 * sigma, axis=1)

    # Perform vectorized bisection.
    for _ in range(iters):

        # Calculate the midpoint of each interval.
        mid = 0.5 * (lo + hi)

        # Evaluate the Gaussian mixture CDF at each midpoint.
        z = (mid[:, None] - mu) / sigma

        cdf = np.sum(w * ndtr(z), axis=1)

        # Update the brackets according to the CDF.
        below = cdf < q

        lo = np.where(below, mid, lo)
        hi = np.where(below, hi, mid)

    # Return the final midpoint estimates.
    return (0.5 * (lo + hi)).astype(np.float64, copy=False)


def interval_report(y, w, mu, sigma, levels=(0.5, 0.8, 0.9, 0.95)):
    """
    Measure the empirical coverage and average width of central
    predictive intervals for Gaussian mixture distributions.

    Parameters
    ----------
    y : np.ndarray
        Observed target values of shape (n,).
    w : np.ndarray
        Mixture weights of shape (n, K).
    mu : np.ndarray
        Gaussian component means of shape (n, K).
    sigma : np.ndarray
        Gaussian component standard deviations of shape (n, K).
    levels : tuple, default=(0.5, 0.8, 0.9, 0.95)
        Desired central prediction interval levels.

    Returns
    -------
    list
        A list of dictionaries containing:
        - level: requested interval level.
        - coverage: fraction of observations inside the interval.
        - width: average interval width.
    """

    # Convert observations to a float64 NumPy array.
    y = np.asarray(y, dtype=np.float64).reshape(-1)

    reports = []

    # Calculate each requested central prediction interval.
    for L in levels:

        L = float(L)

        if not 0.0 < L < 1.0:
            raise ValueError("Each interval level must be between 0 and 1")

        # Determine the lower and upper quantile probabilities.
        q_lo = (1.0 - L) / 2.0
        q_hi = (1.0 + L) / 2.0

        # Calculate the lower and upper interval boundaries.
        lower = mixture_quantile(q_lo, w, mu, sigma)
        upper = mixture_quantile(q_hi, w, mu, sigma)

        # Calculate empirical coverage.
        coverage = float(
            np.mean((y >= lower) & (y <= upper))
        )

        # Calculate mean prediction interval width.
        width = float(
            np.mean(upper - lower)
        )

        # Store the results for this interval level.
        reports.append({
            "level": L,
            "coverage": coverage,
            "width": width
        })

    return reports

# Step 12 - coverage_by_region
def coverage_by_region(x, y, w, mu, sigma, edges, level=0.9):
    """
    Calculate predictive interval coverage and negative log-likelihood
    separately for consecutive regions of the input space.

    Parameters
    ----------
    x : np.ndarray
        Input values of shape (n,).

    y : np.ndarray
        Observed target values of shape (n,).

    w : np.ndarray
        Gaussian mixture weights of shape (n, K).

    mu : np.ndarray
        Gaussian component means of shape (n, K).

    sigma : np.ndarray
        Gaussian component standard deviations of shape (n, K).

    edges : array-like
        Increasing sequence of bin edges.

    level : float, default=0.9
        Central prediction interval level.

    Returns
    -------
    list
        One dictionary per input region containing:
        - lo: Lower boundary of the region.
        - hi: Upper boundary of the region.
        - n: Number of observations in the region.
        - coverage: Empirical interval coverage.
        - width: Mean prediction interval width.
        - nll: Mean negative log-likelihood.
    """

    # Convert inputs to float64 NumPy arrays.
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    edges = np.asarray(edges, dtype=np.float64)

    # Validate the input dimensions.
    if x.ndim != 1 or y.ndim != 1:
        raise ValueError("x and y must be one-dimensional arrays")

    if w.ndim != 2 or mu.ndim != 2 or sigma.ndim != 2:
        raise ValueError("w, mu and sigma must be two-dimensional arrays")

    if not (len(x) == len(y) == w.shape[0] == mu.shape[0] == sigma.shape[0]):
        raise ValueError("All input arrays must have the same number of observations")

    if w.shape != mu.shape or w.shape != sigma.shape:
        raise ValueError("w, mu and sigma must have identical shapes")

    if edges.ndim != 1 or len(edges) < 2:
        raise ValueError("edges must contain at least two values")

    if not np.all(np.diff(edges) > 0):
        raise ValueError("edges must be strictly increasing")

    # Store the regional evaluation results.
    results = []

    # Process each consecutive pair of bin edges.
    for a, b in zip(edges[:-1], edges[1:]):

        # Select observations belonging to the current region.
        # The lower boundary is inclusive and the upper boundary
        # is exclusive: a <= x < b.
        mask = (x >= a) & (x < b)

        # Count observations in this region.
        n_region = int(np.sum(mask))

        # Handle empty regions: their evaluation metrics are undefined.
        if n_region == 0:
            results.append({
                "lo": float(a),
                "hi": float(b),
                "n": 0,
                "coverage": float("nan"),
                "width": float("nan"),
                "nll": float("nan")
            })
            continue

        # Extract observations and mixture parameters for this region.
        x_region = x[mask]
        y_region = y[mask]
        w_region = w[mask]
        mu_region = mu[mask]
        sigma_region = sigma[mask]

        # Calculate the interval coverage and average width.
        report = interval_report(
            y_region,
            w_region,
            mu_region,
            sigma_region,
            levels=(level,)
        )

        # Extract the report for the requested interval level.
        coverage = float(report[0]["coverage"])
        width = float(report[0]["width"])

        # Calculate mean negative log-likelihood for this region.
        nll = float(
            -np.mean(
                mixture_logpdf(
                    y_region,
                    w_region,
                    mu_region,
                    sigma_region
                )
            )
        )

        # Store the results for this region.
        results.append({
            "lo": float(a),
            "hi": float(b),
            "n": n_region,
            "coverage": coverage,
            "width": width,
            "nll": nll
        })

    return results

# Step 13 - gaussian_nll
# Minimum permitted predicted standard deviation in standardized units.
SIGMA_FLOOR = 1e-3


def gaussian_nll(y, mu, sigma):
    """
    Calculate the mean Gaussian negative log-likelihood.

    Parameters
    ----------
    y : Tensor
        Actual target values.
    mu : Tensor
        Predicted Gaussian means.
    sigma : Tensor
        Predicted Gaussian standard deviations (strictly positive).

    Returns
    -------
    Tensor
        Scalar mean negative log-likelihood.
    """

    # Calculate the Gaussian negative log-likelihood element-wise.
    nll = (
        sigma.log()
        + 0.5 * math.log(2.0 * math.pi)
        + ((y - mu) ** 2) / (2.0 * sigma ** 2)
    )

    # Return the mean NLL across all observations.
    return nll.mean()

# Step 14 - fit_fixed_sigma
def fit_fixed_sigma(x, y, hidden=(32, 32), steps=1000, lr=0.01, seed=0):
    """
    Train a homoskedastic Gaussian MLP with a fixed noise standard deviation.

    The network predicts the conditional mean, while a single global
    standard deviation is estimated from the training residuals.

    Parameters
    ----------
    x : np.ndarray
        One-dimensional input array.
    y : np.ndarray
        One-dimensional target array.
    hidden : tuple or list, default=(32, 32)
        Hidden layer widths of the neural network.
    steps : int, default=1000
        Number of training steps.
    lr : float, default=0.01
        Initial learning rate.
    seed : int, default=0
        Random seed for network initialization and training.

    Returns
    -------
    predict : callable
        Prediction function accepting a one-dimensional input array.

        Returns
        -------
        w : np.ndarray
            Mixture weights of shape (n, 1), all equal to 1.
        mu : np.ndarray
            Predicted means in original target units, shape (n, 1).
        sigma : np.ndarray
            Constant predicted standard deviations in original
            target units, shape (n, 1).
    """

    # Convert the input arrays to float64.
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    # Step 1: Standardize the inputs and targets.
    sx = Standardizer(x)
    sy = Standardizer(y)

    X = sx.forward(x)[:, None]
    Y = sy.forward(y)

    # Step 2: Initialize and train the neural network.
    net = DenseNet(1, hidden, 1, seed=seed)

    # Train the network by minimizing the mean squared error.
    fit(
        net.parameters(),
        lambda xb, yb, masks: (
            (net(xb)[:, 0] - yb) ** 2
        ).mean(),
        X,
        Y,
        steps=steps,
        lr=lr,
        seed=seed
    )

    # Step 3: Estimate the global noise standard deviation
    # using the residuals on the training data.
    train_predictions = net.np_forward(X)[:, 0]

    residuals = train_predictions - Y

    sigma_z = float(np.sqrt(np.mean(residuals ** 2)))

    # Step 4: Define the prediction function.
    def predict(xq):
        """
        Generate Gaussian predictions for new inputs.

        Parameters
        ----------
        xq : np.ndarray
            One-dimensional array of query inputs.

        Returns
        -------
        tuple
            (w, mu, sigma), each having shape (n, 1).
        """

        xq = np.asarray(xq, dtype=np.float64).reshape(-1)

        # Standardize the query inputs.
        Xq = sx.forward(xq)[:, None]

        # Predict the conditional mean in standardized units.
        mu_z = net.np_forward(Xq)

        # Convert mean and standard deviation to original units.
        mu, sigma = sy.scale_density(
            mu_z,
            np.full_like(mu_z, sigma_z, dtype=np.float64)
        )

        # A single Gaussian component has weight 1 for every input.
        w = np.ones_like(mu, dtype=np.float64)

        return w, mu, sigma

    return predict

# Step 15 - fit_two_head
def gaussian_head_params(o, sy):
    """
    Convert two-headed network outputs into Gaussian parameters
    in the original target units.

    Parameters
    ----------
    o : np.ndarray
        Network output of shape (n, 2).
        Column 0 contains the predicted mean.
        Column 1 contains the raw predicted standard deviation.
    sy : Standardizer
        Target standardizer.

    Returns
    -------
    tuple
        w, mu, sigma, each with shape (n, 1).
    """

    # Convert network outputs to float64.
    o = np.asarray(o, dtype=np.float64)

    # Validate the network output shape.
    if o.ndim != 2 or o.shape[1] != 2:
        raise ValueError("o must have shape (n, 2)")

    # First output head: mean in standardized units.
    mu_z = o[:, 0:1]

    # Second output head: positive scale in standardized units.
    sigma_z = np.logaddexp(0.0, o[:, 1:2]) + SIGMA_FLOOR

    # Convert mean and standard deviation to original units.
    mu, sigma = sy.scale_density(mu_z, sigma_z)

    # This model has one Gaussian component per input.
    w = np.ones_like(mu, dtype=np.float64)

    return w, mu, sigma

def fit_two_head(x, y, hidden=(32, 32), steps=1000, lr=0.01, seed=0):
    """
    Train a heteroskedastic Gaussian MLP with two output heads.

    The first output predicts the conditional mean.
    The second output predicts the input-dependent standard deviation.

    Backpropagation and Adam updates are computed in NumPy to avoid
    a tinygrad CFGContext compilation failure in this model's
    two-headed Gaussian likelihood graph.

    Returns a prediction function producing (w, mu, sigma) in
    original target units.
    """

    # Convert training data to float64 and flatten.
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)

    if x.size != y.size:
        raise ValueError(
            "x and y must contain the same number of observations"
        )

    if x.size == 0:
        raise ValueError("Training data cannot be empty")

    if steps < 0:
        raise ValueError("steps must be non-negative")

    # Standardize inputs and targets.
    sx = Standardizer(x)
    sy = Standardizer(y)

    X = sx.forward(x)[:, None].astype(np.float64)
    Y = sy.forward(y).astype(np.float64)

    # Create the shared-backbone, two-headed network.
    net = DenseNet(
        d_in=1,
        hidden=hidden,
        d_out=2,
        seed=seed
    )

    # Initialize the output bias so the initial standardized
    # predicted standard deviation is approximately one.
    initial_bias = Tensor(
        [0.0, float(softplus_inv(1.0))],
        dtype=dtypes.float32
    ).contiguous().realize()

    net.layers[-1][1].assign(initial_bias).realize()

    # Copy the tinygrad parameters into NumPy arrays.
    # The network architecture and initialization remain unchanged.
    weights = [
        W.numpy().astype(np.float64)
        for W, b in net.layers
    ]

    biases = [
        b.numpy().astype(np.float64)
        for W, b in net.layers
    ]

    # Initialize Adam moment estimates.
    m_w = [np.zeros_like(W) for W in weights]
    v_w = [np.zeros_like(W) for W in weights]
    m_b = [np.zeros_like(b) for b in biases]
    v_b = [np.zeros_like(b) for b in biases]

    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-8

    n = len(X)

    # Train using full-batch Gaussian negative log-likelihood.
    for t in range(1, steps + 1):

        # Forward pass through the hidden layers.
        activations = [X]

        for W, b in zip(weights[:-1], biases[:-1]):
            h = np.tanh(activations[-1] @ W + b)
            activations.append(h)

        # Linear output layer.
        output = activations[-1] @ weights[-1] + biases[-1]

        mu_z = output[:, 0]
        raw_sigma = output[:, 1]

        # Stable NumPy Softplus and its derivative.
        sigma_z = np.logaddexp(0.0, raw_sigma) + SIGMA_FLOOR

        sigmoid_raw = 1.0 / (
            1.0 + np.exp(-np.clip(raw_sigma, -60.0, 60.0))
        )

        # Gaussian NLL derivatives.
        residual = Y - mu_z

        d_mu = (mu_z - Y) / (sigma_z ** 2 * n)

        d_sigma = (
            1.0 / sigma_z
            - residual ** 2 / sigma_z ** 3
        ) / n

        d_raw_sigma = d_sigma * sigmoid_raw

        # Gradient with respect to the two output columns.
        d_output = np.column_stack((d_mu, d_raw_sigma))

        # Allocate gradient arrays.
        grad_w = [None] * len(weights)
        grad_b = [None] * len(biases)

        # Output layer gradients.
        grad_w[-1] = activations[-1].T @ d_output
        grad_b[-1] = np.sum(d_output, axis=0)

        # Backpropagate through the hidden layers.
        d_hidden = d_output @ weights[-1].T

        for layer in range(len(weights) - 2, -1, -1):

            hidden_output = activations[layer + 1]

            d_pre = d_hidden * (1.0 - hidden_output ** 2)

            grad_w[layer] = activations[layer].T @ d_pre
            grad_b[layer] = np.sum(d_pre, axis=0)

            if layer > 0:
                d_hidden = d_pre @ weights[layer].T

        # Cosine learning-rate schedule, matching Step 8.
        lr_t = lr * 0.5 * (
            1.0 + math.cos(math.pi * (t - 1) / max(steps, 1))
        )

        # Adam updates.
        for j in range(len(weights)):

            m_w[j] = beta1 * m_w[j] + (1.0 - beta1) * grad_w[j]
            v_w[j] = beta2 * v_w[j] + (1.0 - beta2) * grad_w[j] ** 2

            m_b[j] = beta1 * m_b[j] + (1.0 - beta1) * grad_b[j]
            v_b[j] = beta2 * v_b[j] + (1.0 - beta2) * grad_b[j] ** 2

            # Bias correction.
            m_w_hat = m_w[j] / (1.0 - beta1 ** t)
            v_w_hat = v_w[j] / (1.0 - beta2 ** t)

            m_b_hat = m_b[j] / (1.0 - beta1 ** t)
            v_b_hat = v_b[j] / (1.0 - beta2 ** t)

            # Update weights and biases.
            weights[j] -= lr_t * m_w_hat / (
                np.sqrt(v_w_hat) + eps
            )

            biases[j] -= lr_t * m_b_hat / (
                np.sqrt(v_b_hat) + eps
            )

    # Copy trained NumPy parameters back into the tinygrad network.
    for j, (W_tensor, b_tensor) in enumerate(net.layers):

        W_tensor.assign(
            Tensor(weights[j].astype(np.float32)).contiguous()
        ).realize()

        b_tensor.assign(
            Tensor(biases[j].astype(np.float32)).contiguous()
        ).realize()

    # Prediction function.
    def predict(xq):

        xq = np.asarray(xq, dtype=np.float64).reshape(-1)

        Xq = sx.forward(xq)[:, None]

        output = net.np_forward(Xq)

        return gaussian_head_params(output, sy)

    return predict

# Step 16 - moe_e_step
def moe_e_step(Phi, y, V, W, sig2):
    Phi = np.asarray(Phi, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    V = np.asarray(V, dtype=np.float64)
    W = np.asarray(W, dtype=np.float64)
    sig2 = np.asarray(sig2, dtype=np.float64).reshape(-1)

    if Phi.ndim != 2:
        raise ValueError("Phi must have shape (n, D)")

    n, d = Phi.shape

    if y.shape != (n,):
        raise ValueError("y must have shape (n,)")
    if V.ndim != 2 or V.shape[1] != d:
        raise ValueError("V must have shape (K, D)")
    if W.shape != V.shape:
        raise ValueError("W must have shape (K, D)")
    if sig2.shape != (V.shape[0],):
        raise ValueError("sig2 must have shape (K,)")
    if np.any(~np.isfinite(sig2)) or np.any(sig2 <= 0.0):
        raise ValueError("expert variances must be finite and positive")

    # Log gate probabilities: log pi_nk = log softmax(Phi @ V.T).
    gate_logits = Phi @ V.T
    gate_max = np.max(gate_logits, axis=1, keepdims=True)
    log_gate = (
        gate_logits
        - gate_max
        - np.log(np.sum(np.exp(gate_logits - gate_max), axis=1, keepdims=True))
    )

    # Log Gaussian likelihood for each expert and observation.
    expert_mean = Phi @ W.T
    residual = y[:, None] - expert_mean
    log_expert = -0.5 * (
        np.log(2.0 * np.pi * sig2)[None, :]
        + residual**2 / sig2[None, :]
    )

    # Normalize the joint log probabilities across experts.
    log_joint = log_gate + log_expert
    row_max = np.max(log_joint, axis=1, keepdims=True)
    log_norm = row_max + np.log(
        np.sum(np.exp(log_joint - row_max), axis=1, keepdims=True)
    )

    r = np.exp(log_joint - log_norm)
    loglik = float(np.mean(log_norm))

    return r, loglik

# Step 17 - moe_m_step
def moe_m_step(Phi, y, r, V, gate_steps=20, gate_lr=1.0):
    Phi = np.asarray(Phi, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    r = np.asarray(r, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)

    if Phi.ndim != 2:
        raise ValueError("Phi must have shape (n, D)")

    n, d = Phi.shape
    if y.shape != (n,):
        raise ValueError("y must have shape (n,)")
    if r.ndim != 2 or r.shape[0] != n:
        raise ValueError("r must have shape (n, K)")
    if V.shape != (r.shape[1], d):
        raise ValueError("V must have shape (K, D)")
    if np.any(r < 0.0) or not np.all(np.isfinite(r)):
        raise ValueError("responsibilities must be finite and nonnegative")

    k_count = r.shape[1]
    W_new = np.zeros((k_count, d), dtype=np.float64)
    sig2 = np.empty(k_count, dtype=np.float64)
    eye = np.eye(d, dtype=np.float64)

    # Expert updates: responsibility-weighted least squares.
    for k in range(k_count):
        weights = r[:, k]
        weighted_phi = Phi * weights[:, None]

        lhs = Phi.T @ weighted_phi + 1e-8 * eye
        rhs = Phi.T @ (weights * y)

        W_new[k] = np.linalg.solve(lhs, rhs)

        residual = y - Phi @ W_new[k]
        weight_sum = np.sum(weights)

        if weight_sum > 0.0:
            sig2[k] = max(
                np.sum(weights * residual**2) / weight_sum,
                1e-6,
            )
        else:
            # No effective responsibility: retain a finite variance.
            sig2[k] = 1e-6

    # Gate update: gradient descent on responsibility-weighted
    # softmax cross-entropy.
    V_new = V.copy()

    for _ in range(int(gate_steps)):
        logits = Phi @ V_new.T
        logits -= np.max(logits, axis=1, keepdims=True)

        exp_logits = np.exp(logits)
        pi = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

        gradient = (Phi.T @ (pi - r)).T / n
        V_new -= gate_lr * gradient

    return V_new, W_new, sig2

# Step 18 - fit_mole
def fit_mole(x, y, K=3, n_iter=200, restarts=8, seed=0, gate_steps=20, gate_lr=1.0):
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)

    if x.shape != y.shape:
        raise ValueError("x and y must have the same number of observations")
    if x.size == 0:
        raise ValueError("x and y must not be empty")
    if K < 1 or n_iter < 1 or restarts < 1:
        raise ValueError("K, n_iter, and restarts must be positive")

    # Standardize inputs and targets.
    sx = Standardizer(x)
    sy = Standardizer(y)

    xz = sx.forward(x)
    yz = sy.forward(y)

    n = x.size
    Phi = np.column_stack([np.ones(n), xz])

    best_V = None
    best_W = None
    best_sig2 = None
    best_trace = None
    best_loglik = -np.inf
    restart_logliks = []

    for i in range(restarts):
        rng = np.random.default_rng(seed + i)

        # Initialize in the specified order.
        W = rng.normal(0.0, 1.0, size=(K, 2))
        V = rng.normal(0.0, 0.1, size=(K, 2))
        sig2 = np.ones(K, dtype=np.float64)

        trace = []

        for _ in range(n_iter):
            r, loglik = moe_e_step(Phi, yz, V, W, sig2)
            trace.append(float(loglik))

            V, W, sig2 = moe_m_step(
                Phi,
                yz,
                r,
                V,
                gate_steps=gate_steps,
                gate_lr=gate_lr,
            )

        # Evaluate likelihood after the final M-step.
        _, final_loglik = moe_e_step(Phi, yz, V, W, sig2)
        final_loglik = float(final_loglik)

        # Make the last trace entry correspond to the final parameters.
        trace[-1] = final_loglik
        restart_logliks.append(final_loglik)

        # Keep the earliest restart in case of a tie.
        if final_loglik > best_loglik:
            best_loglik = final_loglik
            best_V = V.copy()
            best_W = W.copy()
            best_sig2 = sig2.copy()
            best_trace = trace.copy()

    def predict(x_new):
        x_new = np.asarray(x_new, dtype=np.float64).reshape(-1)

        x_new_z = sx.forward(x_new)
        Phi_new = np.column_stack([
            np.ones(x_new.size),
            x_new_z,
        ])

        # Softmax gate probabilities.
        logits = Phi_new @ best_V.T
        logits -= np.max(logits, axis=1, keepdims=True)

        exp_logits = np.exp(logits)
        w = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

        # Expert means and standard deviations in standardized units.
        mu_z = Phi_new @ best_W.T
        sigma_z = np.broadcast_to(
            np.sqrt(best_sig2)[None, :],
            mu_z.shape,
        )

        # Convert both Gaussian parameters to original y units.
        mu, sigma = sy.scale_density(mu_z, sigma_z)

        return w, mu, sigma

    return {
        "V": best_V,
        "W": best_W,
        "sig2": best_sig2,
        "trace": best_trace,
        "restart_logliks": restart_logliks,
        "predict": predict,
    }

# Step 19 - point_report
def mean_and_mode(w, mu):
    w = np.asarray(w, dtype=np.float64)
    mu = np.asarray(mu, dtype=np.float64)

    if w.ndim != 2 or mu.ndim != 2:
        raise ValueError("w and mu must have shape (n, K)")
    if w.shape != mu.shape:
        raise ValueError("w and mu must have the same shape")
    if w.shape[1] == 0:
        raise ValueError("at least one expert is required")

    # Conditional mean: weighted average of expert means.
    mean = np.sum(w * mu, axis=1)

    # Most-active-expert prediction: select the largest gate weight.
    active_expert = np.argmax(w, axis=1)
    mode = mu[np.arange(mu.shape[0]), active_expert]

    return mean, mode


def point_report(x, y, w, mu):
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)

    mean, mode = mean_and_mode(w, mu)

    if x.shape != y.shape or mean.shape != y.shape:
        raise ValueError("x, y, w, and mu must have matching observation counts")

    # Root mean squared prediction errors.
    rmse_mean = float(np.sqrt(np.mean((mean - y) ** 2)))
    rmse_mode = float(np.sqrt(np.mean((mode - y) ** 2)))

    # Fraction of predictions that lie inside the true conditional support.
    support_mean = float(np.mean(in_support(mean, x)))
    support_mode = float(np.mean(in_support(mode, x)))

    return {
        "rmse_mean": rmse_mean,
        "rmse_mode": rmse_mode,
        "support_mean": support_mean,
        "support_mode": support_mode,
    }

# Step 20 - mdn_nll
def mdn_nll(y, logits, mu, sigma):
    y_col = y.reshape(-1, 1)

    # A one-component mixture is exactly a Gaussian.
    # Avoid unnecessary softmax and log-sum-exp operations.
    if logits.shape[1] == 1:
        nll = (
            sigma.log()
            + 0.5 * math.log(2.0 * math.pi)
            + 0.5 * ((y_col - mu) / sigma) ** 2
        )
        return nll.mean()

    # Log mixture weights.
    log_gate = logits.log_softmax(axis=1)

    # Gaussian log-density per component.
    log_gaussian = (
        -0.5 * math.log(2.0 * math.pi)
        - sigma.log()
        - 0.5 * ((y_col - mu) / sigma) ** 2
    )

    log_joint = log_gate + log_gaussian

    # Numerically stable log-sum-exp with detached maximum.
    max_log = log_joint.max(axis=1, keepdim=True).detach()
    log_prob = (
        max_log
        + (log_joint - max_log).exp()
            .sum(axis=1, keepdim=True)
            .log()
    )

    return -log_prob.mean()

# Step 21 - mdn_grads
def mdn_grads(y, logits, mu, sigma):
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    logits = np.asarray(logits, dtype=np.float64)
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)

    if logits.ndim != 2:
        raise ValueError("logits must have shape (n, K)")
    if mu.shape != logits.shape or sigma.shape != logits.shape:
        raise ValueError("logits, mu, and sigma must have identical shapes")
    if y.shape != (logits.shape[0],):
        raise ValueError("y must have shape (n,)")
    if np.any(sigma <= 0.0) or not np.all(np.isfinite(sigma)):
        raise ValueError("sigma must be finite and strictly positive")

    n = y.size

    # Stable log-softmax for the gating network.
    shifted_logits = logits - np.max(logits, axis=1, keepdims=True)
    log_gate = shifted_logits - np.log(
        np.sum(np.exp(shifted_logits), axis=1, keepdims=True)
    )

    # Gaussian log-density for every observation and component.
    residual = y[:, None] - mu
    log_gaussian = -0.5 * np.log(2.0 * np.pi) - np.log(sigma) \
                   - 0.5 * (residual / sigma) ** 2

    # Posterior component responsibilities, computed in log space.
    log_joint = log_gate + log_gaussian
    log_joint_max = np.max(log_joint, axis=1, keepdims=True)
    log_norm = log_joint_max + np.log(
        np.sum(np.exp(log_joint - log_joint_max), axis=1, keepdims=True)
    )
    resp = np.exp(log_joint - log_norm)

    # Closed-form gradients of the mean mixture NLL.
    grad_logits = (np.exp(log_gate) - resp) / n

    grad_mu = (
        resp * (mu - y[:, None]) / (sigma ** 2)
    ) / n

    grad_sigma = (
        resp * (
            1.0 / sigma
            - residual**2 / (sigma ** 3)
        )
    ) / n

    return {
        "resp": resp,
        "logits": grad_logits,
        "mu": grad_mu,
        "sigma": grad_sigma,
    }

# Step 22 - fit_mdn
def fit_mdn(x, y, K=3, hidden=(32, 32), steps=1000, lr=0.01, seed=0):
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)

    if x.shape != y.shape:
        raise ValueError("x and y must have the same number of observations")
    if x.size == 0:
        raise ValueError("Training data cannot be empty")
    if K < 1:
        raise ValueError("K must be positive")

    # A single-component MDN is equivalent to a heteroskedastic
    # Gaussian model. Reuse the existing two-headed implementation.
    if K == 1:
        gaussian_predict = fit_two_head(
            x,
            y,
            hidden=hidden,
            steps=steps,
            lr=lr,
            seed=seed,
        )

        def predict_single(xq):
            w, mu, sigma = gaussian_predict(xq)
            return np.ones_like(mu), mu, sigma

        return predict_single

    # Standardize inputs and targets.
    sx = Standardizer(x)
    sy = Standardizer(y)

    X = sx.forward(x)[:, None]
    Y = sy.forward(y)

    # Shared network: K logits, K means, K raw scales.
    net = DenseNet(
        d_in=1,
        hidden=hidden,
        d_out=3 * K,
        seed=seed,
    )

    # Initialize output biases to break component symmetry.
    mean_bias = np.linspace(-1.2, 1.2, K)

    initial_bias = np.concatenate([
        np.zeros(K, dtype=np.float64),
        mean_bias,
        np.full(K, float(softplus_inv(0.5))),
    ])

    bias_tensor = Tensor(
        initial_bias.astype(np.float32),
        dtype=dtypes.float32,
    ).contiguous().realize()

    net.layers[-1][1].assign(bias_tensor).realize()

    # Mixture negative log-likelihood.
    def loss_fn(xb, yb, masks):
        output = net(xb)

        logits = output[:, :K]
        mu = output[:, K:2 * K]
        sigma = softplus(output[:, 2 * K:]) + SIGMA_FLOOR

        return mdn_nll(yb, logits, mu, sigma)

    # Train the shared network.
    fit(
        params=net.parameters(),
        loss_fn=loss_fn,
        X=X,
        Y=Y,
        steps=steps,
        lr=lr,
        mask_fn=None,
        seed=seed,
    )

    def predict(xq):
        xq = np.asarray(xq, dtype=np.float64).reshape(-1)

        # Standardize query inputs.
        Xq = sx.forward(xq)[:, None]

        # Network predictions.
        output = net.np_forward(Xq)

        logits = output[:, :K]
        mu_z = output[:, K:2 * K]
        raw_sigma = output[:, 2 * K:]

        # Stable softmax gate probabilities.
        logits = logits - np.max(logits, axis=1, keepdims=True)
        exp_logits = np.exp(logits)
        w = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

        # Positive component standard deviations.
        sigma_z = np.logaddexp(0.0, raw_sigma) + SIGMA_FLOOR

        # Convert means and scales to original target units.
        mu, sigma = sy.scale_density(mu_z, sigma_z)

        return w, mu, sigma

    return predict

# Step 23 - density_set_report
def density_set_report(y, w, mu, sigma, grid, level=0.9):
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    w = np.asarray(w, dtype=np.float64)
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    grid = np.asarray(grid, dtype=np.float64).reshape(-1)

    if w.ndim != 2 or mu.shape != w.shape or sigma.shape != w.shape:
        raise ValueError("w, mu, and sigma must have shape (n, K)")
    if y.shape != (w.shape[0],):
        raise ValueError("y must have shape (n,)")
    if grid.size < 2:
        raise ValueError("grid must contain at least two points")
    if not np.all(np.diff(grid) > 0):
        raise ValueError("grid must be strictly increasing")
    if np.any(sigma <= 0) or not np.all(np.isfinite(sigma)):
        raise ValueError("sigma must be finite and positive")
    if not 0.0 < level <= 1.0:
        raise ValueError("level must be in (0, 1]")

    n = y.size
    dz = float(grid[1] - grid[0])

    # Evaluate each row in chunks to limit memory use.
    thresholds = np.empty(n, dtype=np.float64)
    lengths = np.empty(n, dtype=np.float64)

    for start in range(0, n, 500):
        end = min(start + 500, n)

        # Mixture density on the grid: (chunk, grid_size, K).
        z = (
            grid[None, :, None]
            - mu[start:end, None, :]
        ) / sigma[start:end, None, :]

        component_density = (
            w[start:end, None, :]
            * np.exp(-0.5 * z**2)
            / (
                np.sqrt(2.0 * np.pi)
                * sigma[start:end, None, :]
            )
        )

        density = np.sum(component_density, axis=2)

        # Sort grid densities from highest to lowest.
        sorted_density = np.sort(density, axis=1)[:, ::-1]

        # Approximate cumulative probability mass on the grid.
        cumulative_mass = np.cumsum(sorted_density * dz, axis=1)

        # First density threshold reaching the requested mass.
        reached = cumulative_mass >= level
        first_index = np.argmax(reached, axis=1)

        threshold = sorted_density[
            np.arange(end - start),
            first_index,
        ]

        thresholds[start:end] = threshold

        # Highest-density set length = number of selected grid
        # points multiplied by the grid spacing.
        lengths[start:end] = (
            np.sum(density >= threshold[:, None], axis=1) * dz
        )

    # Evaluate the observed target's own mixture density.
    observed_log_density = mixture_logpdf(
        y,
        w,
        mu,
        sigma,
    )
    observed_density = np.exp(observed_log_density)

    # An observation is covered when its density reaches
    # the row-specific highest-density threshold.
    coverage = float(
        np.mean(observed_density >= thresholds)
    )

    return {
        "level": float(level),
        "coverage": coverage,
        "length": float(np.mean(lengths)),
    }

# Step 24 - fit_hme
def hme_forward(xb, params):
    V_top, V_sub, W, a = params

    # Design matrix: intercept and standardized input.
    Phi = Tensor.cat(
        Tensor.ones(xb.shape[0], 1),
        xb,
        dim=1,
    )

    # Top-level gate: probabilities of the two branches.
    log_top = (
        Phi @ V_top.transpose()
    ).log_softmax(axis=1)

    # One two-way sub-gate for each top-level branch.
    log_sub_0 = (
        Phi @ V_sub[0].transpose()
    ).log_softmax(axis=1)

    log_sub_1 = (
        Phi @ V_sub[1].transpose()
    ).log_softmax(axis=1)

    # Four joint path log-weights, ordered by (top, sub).
    log_w = Tensor.cat(
        log_top[:, 0:1] + log_sub_0[:, 0:1],
        log_top[:, 0:1] + log_sub_0[:, 1:2],
        log_top[:, 1:2] + log_sub_1[:, 0:1],
        log_top[:, 1:2] + log_sub_1[:, 1:2],
        dim=1,
    )

    # Four linear expert means.
    mu = Phi @ W.transpose()

    # Four positive expert standard deviations.
    sigma = (
        softplus(a)
        .reshape(1, 4)
        .expand(xb.shape[0], 4)
        + SIGMA_FLOOR
    )

    return log_w, mu, sigma


def fit_hme(x, y, steps=1500, lr=0.01, seed=0):
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)

    if x.shape != y.shape:
        raise ValueError("x and y must have the same number of observations")
    if x.size == 0:
        raise ValueError("Training data cannot be empty")

    # Standardize inputs and targets.
    sx = Standardizer(x)
    sy = Standardizer(y)

    X = sx.forward(x).reshape(-1, 1)
    Y = sy.forward(y)

    # Initialize parameters in the specified order.
    rng = np.random.default_rng(seed)

    V_top = Tensor(
        rng.normal(0.0, 0.5, size=(2, 2)).astype(np.float32)
    ).contiguous().realize()

    V_sub = Tensor(
        rng.normal(0.0, 0.5, size=(2, 2, 2)).astype(np.float32)
    ).contiguous().realize()

    W = Tensor(
        rng.normal(0.0, 0.5, size=(4, 2)).astype(np.float32)
    ).contiguous().realize()

    a = Tensor(
        np.full(
            4,
            float(softplus_inv(0.5)),
            dtype=np.float32,
        )
    ).contiguous().realize()

    # Train all hierarchical gates and experts jointly.
    params = [V_top, V_sub, W, a]

    def loss_fn(xb, yb, masks):
        log_w, mu, sigma = hme_forward(xb, params)

        return mdn_nll(
            yb,
            log_w,
            mu,
            sigma,
        )

    fit(
        params=params,
        loss_fn=loss_fn,
        X=X,
        Y=Y,
        steps=steps,
        lr=lr,
        mask_fn=None,
        seed=seed,
    )

    def predict(xq):
        xq = np.asarray(xq, dtype=np.float64).reshape(-1)

        # Standardize query inputs.
        xq_z = sx.forward(xq).reshape(-1, 1)

        xb = Tensor(
            xq_z.astype(np.float32)
        ).contiguous().realize()

        # Evaluate the trained hierarchical model.
        log_w, mu_z, sigma_z = hme_forward(
            xb,
            params,
        )

        # Convert predictions to NumPy arrays.
        log_w_np = log_w.numpy().astype(np.float64)
        mu_z_np = mu_z.numpy().astype(np.float64)
        sigma_z_np = sigma_z.numpy().astype(np.float64)

        # Convert joint log-weights into probabilities.
        log_w_np -= np.max(log_w_np, axis=1, keepdims=True)
        exp_log_w = np.exp(log_w_np)
        w = exp_log_w / np.sum(
            exp_log_w,
            axis=1,
            keepdims=True,
        )

        # Convert expert means and scales to original y units.
        mu, sigma = sy.scale_density(
            mu_z_np,
            sigma_z_np,
        )

        return w, mu, sigma

    return predict

# Step 25 - fit_dropout
def dropout_masks(rng, n, hidden, p):
    """
    Generate one independent Bernoulli dropout mask per hidden layer.

    Each mask has shape (n, width) and dtype float32.
    A unit is retained with probability 1 - p.
    """

    if not 0.0 <= p < 1.0:
        raise ValueError("p must be in the range [0, 1)")

    if n < 0:
        raise ValueError("n must be non-negative")

    return [
        (rng.random((n, width)) >= p).astype(np.float32)
        for width in hidden
    ]


def fit_dropout(x, y, p, hidden=(32, 32), steps=1000, lr=0.01, seed=0):
    """
    Train a two-headed heteroskedastic Gaussian MLP using
    Bernoulli dropout on its hidden activations.

    The network predicts:
        output[:, 0] : conditional mean
        output[:, 1] : raw conditional standard deviation

    Training uses NumPy backpropagation and Adam to avoid
    the tinygrad CFGContext gradient compilation failure.

    Returns:
        {
            "net": trained DenseNet,
            "sx": input Standardizer,
            "sy": target Standardizer,
            "p": dropout probability
        }
    """

    # --------------------------------------------------
    # Validate and prepare the data.
    # --------------------------------------------------

    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)

    if x.shape != y.shape:
        raise ValueError("x and y must have the same shape")

    if x.size == 0:
        raise ValueError("Training data cannot be empty")

    if not 0.0 <= p < 1.0:
        raise ValueError("p must be in the range [0, 1)")

    if steps < 0:
        raise ValueError("steps must be non-negative")

    # --------------------------------------------------
    # Standardize inputs and targets.
    # --------------------------------------------------

    sx = Standardizer(x)
    sy = Standardizer(y)

    X = sx.forward(x)[:, None].astype(np.float64)
    Y = sy.forward(y).astype(np.float64)

    n = len(X)

    # --------------------------------------------------
    # Initialize the two-headed network.
    # --------------------------------------------------

    net = DenseNet(
        d_in=1,
        hidden=hidden,
        d_out=2,
        seed=seed,
    )

    # Initialize the mean bias to zero and the scale bias
    # so that the initial standardized sigma is approximately 1.
    initial_bias = Tensor(
        [0.0, float(softplus_inv(1.0))],
        dtype=dtypes.float32,
    ).contiguous().realize()

    net.layers[-1][1].assign(initial_bias).realize()

    # Copy parameters into NumPy arrays for training.
    weights = [
        W.numpy().astype(np.float64)
        for W, b in net.layers
    ]

    biases = [
        b.numpy().astype(np.float64)
        for W, b in net.layers
    ]

    # --------------------------------------------------
    # Initialize Adam.
    # --------------------------------------------------

    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-8

    m_w = [np.zeros_like(W) for W in weights]
    v_w = [np.zeros_like(W) for W in weights]

    m_b = [np.zeros_like(b) for b in biases]
    v_b = [np.zeros_like(b) for b in biases]

    # Use the same dedicated training random stream as Step 8.
    rng = np.random.default_rng([seed, 3])

    # --------------------------------------------------
    # Training loop.
    # --------------------------------------------------

    for t in range(1, steps + 1):

        # Generate independent masks for every hidden layer.
        if p > 0.0:
            masks = dropout_masks(rng, n, hidden, p)
        else:
            masks = [
                np.ones((n, width), dtype=np.float32)
                for width in hidden
            ]

        # --------------------------------------------------
        # Forward pass.
        # --------------------------------------------------

        activations = [X]
        raw_hidden = []

        for layer, (W, b) in enumerate(
            zip(weights[:-1], biases[:-1])
        ):

            pre_activation = activations[-1] @ W + b

            h = np.tanh(pre_activation)

            raw_hidden.append(h)

            # Apply dropout after the tanh activation.
            h = h * masks[layer]

            activations.append(h)

        # Linear output layer.
        output = activations[-1] @ weights[-1] + biases[-1]

        mu_z = output[:, 0]
        raw_sigma = output[:, 1]

        # --------------------------------------------------
        # Gaussian negative log-likelihood.
        # --------------------------------------------------

        sigma_z = (
            np.logaddexp(0.0, raw_sigma) + SIGMA_FLOOR
        )

        residual = Y - mu_z

        # Derivative of NLL with respect to the mean.
        d_mu = (
            (mu_z - Y) / (sigma_z ** 2 * n)
        )

        # Derivative of NLL with respect to sigma.
        d_sigma = (
            1.0 / sigma_z
            - residual ** 2 / sigma_z ** 3
        ) / n

        # Derivative of Softplus.
        sigmoid_raw = 1.0 / (
            1.0 + np.exp(-np.clip(raw_sigma, -60.0, 60.0))
        )

        d_raw_sigma = d_sigma * sigmoid_raw

        # Combine the two output-head gradients.
        d_output = np.column_stack(
            (d_mu, d_raw_sigma)
        )

        # --------------------------------------------------
        # Backpropagation.
        # --------------------------------------------------

        grad_w = [None] * len(weights)
        grad_b = [None] * len(biases)

        # Output-layer gradients.
        grad_w[-1] = activations[-1].T @ d_output
        grad_b[-1] = np.sum(d_output, axis=0)

        # Gradient entering the final hidden layer.
        d_hidden = d_output @ weights[-1].T

        # Backpropagate through hidden layers.
        for layer in range(len(weights) - 2, -1, -1):

            # The dropout mask multiplies the gradient too.
            d_hidden = d_hidden * masks[layer]

            # Derivative of tanh, using the unmasked activation.
            h = raw_hidden[layer]

            d_pre = d_hidden * (1.0 - h ** 2)

            grad_w[layer] = (
                activations[layer].T @ d_pre
            )

            grad_b[layer] = np.sum(
                d_pre,
                axis=0
            )

            if layer > 0:
                d_hidden = d_pre @ weights[layer].T

        # --------------------------------------------------
        # Cosine learning-rate schedule.
        # Matches Step 15.
        # --------------------------------------------------

        lr_t = lr * 0.5 * (
            1.0 + math.cos(
                math.pi * (t - 1) / max(steps, 1)
            )
        )

        # --------------------------------------------------
        # Adam parameter updates.
        # --------------------------------------------------

        for j in range(len(weights)):

            # First and second moments for weights.
            m_w[j] = (
                beta1 * m_w[j]
                + (1.0 - beta1) * grad_w[j]
            )

            v_w[j] = (
                beta2 * v_w[j]
                + (1.0 - beta2) * grad_w[j] ** 2
            )

            # First and second moments for biases.
            m_b[j] = (
                beta1 * m_b[j]
                + (1.0 - beta1) * grad_b[j]
            )

            v_b[j] = (
                beta2 * v_b[j]
                + (1.0 - beta2) * grad_b[j] ** 2
            )

            # Bias correction.
            correction1 = 1.0 - beta1 ** t
            correction2 = 1.0 - beta2 ** t

            m_w_hat = m_w[j] / correction1
            v_w_hat = v_w[j] / correction2

            m_b_hat = m_b[j] / correction1
            v_b_hat = v_b[j] / correction2

            # Update weights.
            weights[j] -= (
                lr_t * m_w_hat
                / (np.sqrt(v_w_hat) + eps)
            )

            # Update biases.
            biases[j] -= (
                lr_t * m_b_hat
                / (np.sqrt(v_b_hat) + eps)
            )

    # --------------------------------------------------
    # Copy trained parameters back into DenseNet.
    # --------------------------------------------------

    for j, (W_tensor, b_tensor) in enumerate(net.layers):

        W_tensor.assign(
            Tensor(
                weights[j].astype(np.float32)
            ).contiguous()
        ).realize()

        b_tensor.assign(
            Tensor(
                biases[j].astype(np.float32)
            ).contiguous()
        ).realize()

    # --------------------------------------------------
    # Return the trained model and standardizers.
    # --------------------------------------------------

    return {
        "net": net,
        "sx": sx,
        "sy": sy,
        "p": float(p),
    }

# Step 26 - mc_dropout_predict
def weight_scaled_predict(model, xq):
    xq = np.asarray(xq, dtype=np.float64).reshape(-1)

    # Standardize inputs.
    Xq = model["sx"].forward(xq)[:, None]

    # One deterministic forward pass using the expected dropout scaling.
    out = model["net"].np_forward(
        Xq,
        scale=1.0 - model["p"]
    )

    # Convert predictions back to the original target units.
    return gaussian_head_params(out, model["sy"])


def mc_dropout_predict(model, xq, S=30, seed=0):
    if S < 1:
        raise ValueError("S must be at least 1")

    xq = np.asarray(xq, dtype=np.float64).reshape(-1)

    # Standardize inputs.
    Xq = model["sx"].forward(xq)[:, None]
    n = len(xq)

    rng = np.random.default_rng([seed, 4])

    weights = []
    means = []
    sigmas = []

    # Each pass uses a fresh set of dropout masks.
    for _ in range(S):
        masks = dropout_masks(
            rng,
            n,
            model["net"].hidden,
            model["p"]
        )

        out = model["net"].np_forward(
            Xq,
            masks=masks
        )

        w, mu, sigma = gaussian_head_params(out, model["sy"])

        weights.append(mu)
        means.append(mu)
        sigmas.append(sigma)

    # Each stochastic pass contributes equally to the mixture.
    w = np.full((n, S), 1.0 / S, dtype=np.float64)
    mu = np.concatenate(means, axis=1)
    sigma = np.concatenate(sigmas, axis=1)

    return w, mu, sigma

# Step 27 - tune_dropout_rate
def tune_dropout_rate(
    x, y, rates=(0.003, 0.01, 0.03), val_frac=0.25,
    S=20, hidden=(32, 32), steps=1000, lr=0.01, seed=0
):
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)

    if x.shape != y.shape or x.size < 2:
        raise ValueError("x and y must have the same length of at least 2")
    if not 0.0 < val_frac < 1.0:
        raise ValueError("val_frac must be between 0 and 1")
    if not rates:
        raise ValueError("rates must contain at least one dropout rate")

    # Create a reproducible training/validation split.
    rng = np.random.default_rng([seed, 5])
    idx = rng.permutation(len(x))
    n_val = int(len(x) * val_frac)

    if n_val < 1 or n_val >= len(x):
        raise ValueError("val_frac must produce nonempty training and validation sets")

    val_idx = idx[:n_val]
    train_idx = idx[n_val:]

    x_train, y_train = x[train_idx], y[train_idx]
    x_val, y_val = x[val_idx], y[val_idx]

    scores = {}

    # Evaluate each dropout rate in the supplied order.
    for i, p in enumerate(rates):
        p = float(p)

        model = fit_dropout(
            x_train,
            y_train,
            p=p,
            hidden=hidden,
            steps=steps,
            lr=lr,
            seed=seed
        )

        w, mu, sigma = mc_dropout_predict(
            model,
            x_val,
            S=S,
            seed=seed
        )

        nll = -float(np.mean(mixture_logpdf(y_val, w, mu, sigma)))
        scores[p] = nll

    # min() preserves the earlier rate when scores are tied.
    best_p = min(scores, key=scores.get)

    # Refit using all available observations.
    refit_model = fit_dropout(
        x,
        y,
        p=best_p,
        hidden=hidden,
        steps=steps,
        lr=lr,
        seed=seed
    )

    return {
        "best_p": float(best_p),
        "scores": {p: float(score) for p, score in scores.items()},
        "model": refit_model
    }

# Step 28 - fit_ensemble
def fit_ensemble(fit_fn, x, y, M=5, seed=0):
    if M < 1:
        raise ValueError("M must be at least 1")

    # Train M independent models with deterministic, distinct seeds.
    models = [
        fit_fn(x, y, seed=seed + 100 * m)
        for m in range(M)
    ]

    def predict(xq):
        predictions = [model(xq) for model in models]

        # Each member contributes equally to the ensemble mixture.
        weights = np.concatenate(
            [w / M for w, mu, sigma in predictions],
            axis=1
        )
        means = np.concatenate(
            [mu for w, mu, sigma in predictions],
            axis=1
        )
        sigmas = np.concatenate(
            [sigma for w, mu, sigma in predictions],
            axis=1
        )

        return weights, means, sigmas

    return predict

# Step 29 - uncertainty_split
def uncertainty_split(w, mu, sigma):
    w = np.asarray(w, dtype=np.float64)
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)

    if w.shape != mu.shape or w.shape != sigma.shape or w.ndim != 2:
        raise ValueError("w, mu, and sigma must have the same (n, K) shape")

    # Normalize mixture weights row-wise.
    weight_sum = np.sum(w, axis=1, keepdims=True)
    if np.any(weight_sum <= 0):
        raise ValueError("Each row must have positive total mixture weight")

    w = w / weight_sum

    # Predictive mean.
    mean = np.sum(w * mu, axis=1)

    # Aleatoric uncertainty: expected within-component variance.
    aleatoric = np.sum(w * sigma**2, axis=1)

    # Epistemic uncertainty: variance of component means.
    epistemic = np.sum(
        w * (mu - mean[:, None])**2,
        axis=1
    )

    # Protect against tiny negative values from floating-point rounding.
    epistemic = np.maximum(epistemic, 0.0)

    return mean, aleatoric, epistemic

# Step 30 - beyond_one_gaussian_report
def score_predictive(y, w, mu, sigma):
    y = np.asarray(y, dtype=np.float64).reshape(-1)

    nll = -float(np.mean(mixture_logpdf(y, w, mu, sigma)))
    ks = float(pit_ks(pit_values(y, w, mu, sigma)))

    interval90 = interval_report(
        y, w, mu, sigma, levels=(0.9,)
    )[0]

    return {
        "nll": float(nll),
        "ks": float(ks),
        "cov90": float(interval90["coverage"]),
        "width90": float(interval90["width"])
    }


def beyond_one_gaussian_report(
    seed=0, n=1000, n_small=200, n_test=5000, steps=1000
):
    t0 = time.time()

    # ============================================================
    # 1. Growing-noise regression experiment
    # ============================================================

    x_train, y_train, _, _ = make_growing_noise_data(
        n, seed=seed
    )

    x_test, y_test, mu_true, sigma_true = make_growing_noise_data(
        n_test, seed=10000 + seed
    )

    # Exact oracle NLL for the true conditional Gaussian.
    oracle_series = float(np.mean(
        0.5 * np.log(2.0 * np.pi * sigma_true**2)
        + 0.5 * ((y_test - mu_true) / sigma_true)**2
    ))

    # Fixed-sigma Gaussian model.
    fixed_predict = fit_fixed_sigma(
        x_train, y_train, steps=steps, seed=seed
    )
    w_fixed, mu_fixed, sigma_fixed = fixed_predict(x_test)

    fixed_score = score_predictive(
        y_test, w_fixed, mu_fixed, sigma_fixed
    )

    # Heteroskedastic two-head Gaussian model.
    two_head_predict = fit_two_head(
        x_train, y_train, steps=steps, seed=seed
    )
    w_two, mu_two, sigma_two = two_head_predict(x_test)

    two_head_score = score_predictive(
        y_test, w_two, mu_two, sigma_two
    )

    # Coverage by input region.
    region_edges = [-20, 0, 20, 40, 60.01]

    fixed_regions = coverage_by_region(
        x_test, y_test,
        w_fixed, mu_fixed, sigma_fixed,
        edges=region_edges
    )

    two_head_regions = coverage_by_region(
        x_test, y_test,
        w_two, mu_two, sigma_two,
        edges=region_edges
    )

    # RMSE of predicted noise scale and conditional mean.
    sigma_rmse = float(np.sqrt(np.mean(
        (sigma_two[:, 0] - sigma_true)**2
    )))

    mean_rmse = {
        "fixed": float(np.sqrt(np.mean(
            (mu_fixed[:, 0] - mu_true)**2
        ))),
        "two_head": float(np.sqrt(np.mean(
            (mu_two[:, 0] - mu_true)**2
        )))
    }

    series = {
        "oracle": float(oracle_series),
        "fixed": fixed_score,
        "two_head": two_head_score,
        "fixed_regions": fixed_regions,
        "two_head_regions": two_head_regions,
        "sigma_rmse": sigma_rmse,
        "mean_rmse": mean_rmse
    }

    # ============================================================
    # 2. One-to-many inverse problem
    # ============================================================

    x_inv, y_inv = make_inverse_data(n, seed=seed)

    x_inv_test, y_inv_test = make_inverse_data(
        n_test, seed=10000 + seed
    )

    oracle_inverse = float(inverse_oracle_nll(x_inv_test))

    oracle_set_length = float(
        0.9 * np.mean(support_size(x_inv_test))
    )

    # One Gaussian.
    gaussian_predict = fit_two_head(
        x_inv, y_inv, steps=steps, seed=seed
    )
    wg, mug, sg = gaussian_predict(x_inv_test)

    gaussian_score = score_predictive(
        y_inv_test, wg, mug, sg
    )

    # Deep ensemble of three Gaussian models.
    ensemble_predict = fit_ensemble(
        lambda xa, ya, seed: fit_two_head(
            xa, ya, steps=steps, seed=seed
        ),
        x_inv, y_inv, M=3, seed=seed
    )

    we, mue, se = ensemble_predict(x_inv_test)

    ensemble_score = score_predictive(
        y_inv_test, we, mue, se
    )

    # MC dropout Gaussian mixture.
    dropout_model = fit_dropout(
        x_inv, y_inv,
        p=0.01,
        steps=steps,
        seed=seed
    )

    wd, mud, sd = mc_dropout_predict(
        dropout_model, x_inv_test, S=30, seed=seed
    )

    dropout_score = score_predictive(
        y_inv_test, wd, mud, sd
    )

    # Mixture of linear experts.
    mole_model = fit_mole(
        x_inv, y_inv, K=3, seed=seed
    )

    wm, mum, sm = mole_model["predict"](x_inv_test)

    mole_score = score_predictive(
        y_inv_test, wm, mum, sm
    )

    # Hierarchical mixture of experts.
    hme_predict = fit_hme(
        x_inv, y_inv,
        steps=steps * 3 // 2,
        seed=seed
    )

    wh, muh, sh = hme_predict(x_inv_test)

    hme_score = score_predictive(
        y_inv_test, wh, muh, sh
    )

    # Fit a separate MDN for each requested component count.
    # fit_mdn returns a prediction function, not a dictionary.
    mdn_models = {}

    for K in (2, 3, 5):
        mdn_models[K] = fit_mdn(
            x_inv, y_inv,
            K=K,
            steps=steps,
            seed=seed
        )

    mdn_scores = {}

    for K in (2, 3, 5):
        wk, muk, sk = mdn_models[K](x_inv_test)

        mdn_scores[K] = score_predictive(
            y_inv_test, wk, muk, sk
        )

    # Point prediction reports.
    mole_points = point_report(
        x_inv_test, y_inv_test, wm, mum
    )

    w3, mu3, sigma3 = mdn_models[3](x_inv_test)

    mdn3_points = point_report(
        x_inv_test, y_inv_test, w3, mu3
    )

    # Highest-density sets on the first 1500 test observations.
    density_grid = np.linspace(-0.5, 1.5, 801)

    gaussian_sets = density_set_report(
        y_inv_test[:1500],
        wg[:1500], mug[:1500], sg[:1500],
        grid=density_grid,
        level=0.9
    )

    mdn3_sets = density_set_report(
        y_inv_test[:1500],
        w3[:1500], mu3[:1500], sigma3[:1500],
        grid=density_grid,
        level=0.9
    )

    inverse = {
        "oracle": oracle_inverse,
        "oracle_set_length": oracle_set_length,
        "gaussian": gaussian_score,
        "ensemble_gaussian": ensemble_score,
        "mc_dropout_gaussian": dropout_score,
        "mole": mole_score,
        "hme": hme_score,
        "mdn": mdn_scores,
        "points": {
            "mole": mole_points,
            "mdn3": mdn3_points
        },
        "sets": {
            "gaussian": gaussian_sets,
            "mdn3": mdn3_sets
        }
    }

    # ============================================================
    # 3. Small-data experiment with a hole in the training inputs
    # ============================================================

    x_small, y_small, _, _ = make_growing_noise_data(
        n_small,
        seed=20000 + seed,
        gap=(10.0, 25.0)
    )

    # Use the no-gap test set from the first experiment.
    inside = (x_test > 10.0) & (x_test < 25.0)
    outside = ~inside

    def small_model_report(w, mu, sigma):
        nll_in = -float(np.mean(
            mixture_logpdf(
                y_test[inside],
                w[inside], mu[inside], sigma[inside]
            )
        ))

        nll_out = -float(np.mean(
            mixture_logpdf(
                y_test[outside],
                w[outside], mu[outside], sigma[outside]
            )
        ))

        interval_in = interval_report(
            y_test[inside],
            w[inside], mu[inside], sigma[inside],
            levels=(0.9,)
        )[0]

        return {
            "nll_out": float(nll_out),
            "nll_in": float(nll_in),
            "cov90_in": float(interval_in["coverage"])
        }

    # Single two-head model.
    single_predict = fit_two_head(
        x_small, y_small,
        steps=steps,
        seed=seed
    )

    w_single, mu_single, sigma_single = single_predict(x_test)

    single_report = small_model_report(
        w_single, mu_single, sigma_single
    )

    # Five-member deep ensemble.
    small_ensemble_predict = fit_ensemble(
        lambda xa, ya, seed: fit_two_head(
            xa, ya, steps=steps, seed=seed
        ),
        x_small, y_small,
        M=5,
        seed=seed
    )

    we_s, mue_s, se_s = small_ensemble_predict(x_test)

    ensemble_report = small_model_report(
        we_s, mue_s, se_s
    )

    # Select dropout rate by held-out likelihood.
    tuned = tune_dropout_rate(
        x_small, y_small,
        rates=(0.003, 0.01, 0.03),
        val_frac=0.25,
        S=20,
        hidden=(32, 32),
        steps=steps,
        lr=0.01,
        seed=seed
    )

    best_dropout_model = tuned["model"]
    best_p = float(tuned["best_p"])

    # MC dropout predictions.
    w_mc, mu_mc, sigma_mc = mc_dropout_predict(
        best_dropout_model,
        x_test,
        S=30,
        seed=seed
    )

    mc_report = small_model_report(
        w_mc, mu_mc, sigma_mc
    )

    # Deterministic weight-scaled predictions.
    w_scaled, mu_scaled, sigma_scaled = weight_scaled_predict(
        best_dropout_model, x_test
    )

    weight_scaled_report = small_model_report(
        w_scaled, mu_scaled, sigma_scaled
    )

    # Epistemic uncertainty: variance of component means.
    _, _, epistemic_ensemble = uncertainty_split(
        we_s, mue_s, se_s
    )

    _, _, epistemic_mc = uncertainty_split(
        w_mc, mu_mc, sigma_mc
    )

    ensemble_report["epistemic_in"] = float(
        np.sqrt(np.mean(epistemic_ensemble[inside]))
    )

    ensemble_report["epistemic_out"] = float(
        np.sqrt(np.mean(epistemic_ensemble[outside]))
    )

    mc_report["epistemic_in"] = float(
        np.sqrt(np.mean(epistemic_mc[inside]))
    )

    mc_report["epistemic_out"] = float(
        np.sqrt(np.mean(epistemic_mc[outside]))
    )

    small = {
        "single": single_report,
        "ensemble": ensemble_report,
        "mc_dropout": mc_report,
        "weight_scaled": weight_scaled_report,
        "best_p": best_p
    }

    return {
        "series": series,
        "inverse": inverse,
        "small": small,
        "seconds": float(time.time() - t0)
    }

