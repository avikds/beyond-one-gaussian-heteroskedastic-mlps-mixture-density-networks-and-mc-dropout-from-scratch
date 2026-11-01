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
def fit(params, loss_fn, X, Y, steps=1000, lr=0.01,
        mask_fn=None, seed=0):

    # Initialize the Adam optimizer.
    opt = Adam(params)

    # Convert training data to realized float32 tensors once.
    xb = Tensor(
        np.asarray(X, dtype=np.float32)
    ).contiguous().realize()

    yb = Tensor(
        np.asarray(Y, dtype=np.float32)
    ).contiguous().realize()

    # Dedicated random stream for dropout masks.
    rng = np.random.default_rng([seed, 3])

    def step(xb, yb, lr_t, *masks):

        current_masks = tuple(masks) if mask_fn is not None else None

        # Calculate the loss.
        loss = loss_fn(xb, yb, current_masks)

        # Calculate gradients using the current parameters.
        grads = loss.gradient(*params)

        # IMPORTANT:
        # Realize the loss and gradients BEFORE modifying parameters.
        # This prevents problematic dependencies in the JIT graph.
        Tensor.realize(loss, *grads)

        # Perform the Adam update.
        updated_tensors = opt.step(grads, lr_t)

        # Realize the updated counter, parameters and optimizer buffers.
        Tensor.realize(*updated_tensors)

        return loss

    # Compile the training step.
    step = TinyJit(step)

    history = []

    for t in range(steps):

        # Cosine learning-rate schedule.
        lr_value = lr * 0.5 * (
            1.0 + math.cos(math.pi * t / steps)
        )

        lr_t = Tensor(
            [lr_value],
            dtype=dtypes.float32
        ).contiguous().realize()

        # Generate fresh dropout masks when required.
        if mask_fn is not None:

            mask_arrays = mask_fn(rng, len(X))

            mask_tensors = tuple(
                Tensor(
                    np.asarray(mask, dtype=np.float32)
                ).contiguous().realize()
                for mask in mask_arrays
            )

        else:
            mask_tensors = ()

        # Execute one training step.
        loss = step(xb, yb, lr_t, *mask_tensors)

        # Record loss every 500 steps and at the final step.
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
    Convert two-headed Gaussian network outputs into predictive
    Gaussian parameters in the original target units.

    Parameters
    ----------
    o : np.ndarray
        Network outputs of shape (n, 2).
        Column 0: predicted mean.
        Column 1: raw predicted standard deviation.
    sy : Standardizer
        Target standardizer.

    Returns
    -------
    tuple
        (w, mu, sigma), each with shape (n, 1).
    """

    # Ensure float64 precision.
    o = np.asarray(o, dtype=np.float64)

    # Extract the predicted mean in standardized units.
    mu_z = o[:, 0:1]

    # Calculate a numerically stable Softplus using NumPy.
    sigma_z = np.logaddexp(0.0, o[:, 1:2]) + SIGMA_FLOOR

    # Convert Gaussian parameters into original target units.
    mu, sigma = sy.scale_density(mu_z, sigma_z)

    # One Gaussian component per input.
    w = np.ones_like(mu, dtype=np.float64)

    return w, mu, sigma


def fit_two_head(x, y, hidden=(32, 32), steps=1000, lr=0.01, seed=0):
    """
    Train a heteroskedastic Gaussian MLP with a shared backbone
    and separate mean and standard-deviation output heads.

    Parameters
    ----------
    x : np.ndarray
        One-dimensional input array.
    y : np.ndarray
        One-dimensional target array.
    hidden : tuple or list
        Hidden layer widths.
    steps : int
        Number of training steps.
    lr : float
        Initial learning rate.
    seed : int
        Random seed.

    Returns
    -------
    predict : callable
        Prediction function returning (w, mu, sigma), each
        of shape (n, 1), in original target units.
    """

    # Convert inputs and targets to float64.
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    # Step 1: Standardize inputs and targets.
    sx = Standardizer(x)
    sy = Standardizer(y)

    X = sx.forward(x)[:, None]
    Y = sy.forward(y)

    # Step 2: Initialize a network with two output units.
    net = DenseNet(1, hidden, 2, seed=seed)

    # Initialize output biases so that the initial predictions are:
    # mean = 0 and standard deviation approximately = 1.
    initial_bias = Tensor(
        [0.0, float(softplus_inv(1.0))],
        dtype=dtypes.float32
    ).contiguous().realize()

    net.layers[-1][1].assign(initial_bias).realize()

    # Step 3: Define the Gaussian negative log-likelihood loss.
    def loss_fn(xb, yb, masks):

        # Forward pass.
        o = net(xb)

        # Extract the two output heads.
        mu_z = o[:, 0].contiguous()
        raw_sigma = o[:, 1].contiguous()

        # Smooth, equivalent Softplus expression.
        # The standardized network output stays in the normal
        # operating range during this training task.
        sigma_z = (1.0 + raw_sigma.exp()).log() + SIGMA_FLOOR

        # Gaussian negative log-likelihood.
        return gaussian_nll(yb, mu_z, sigma_z)

    # Train using the previously implemented TinyJit training loop.
    fit(
        net.parameters(),
        loss_fn,
        X,
        Y,
        steps=steps,
        lr=lr,
        seed=seed
    )

    # Step 4: Define the prediction function.
    def predict(xq):

        # Convert query inputs to a one-dimensional float64 array.
        xq = np.asarray(xq, dtype=np.float64).reshape(-1)

        # Standardize query inputs.
        Xq = sx.forward(xq)[:, None]

        # Obtain the two network outputs.
        o = net.np_forward(Xq)

        # Convert into Gaussian mixture parameters.
        return gaussian_head_params(o, sy)

    return predict

