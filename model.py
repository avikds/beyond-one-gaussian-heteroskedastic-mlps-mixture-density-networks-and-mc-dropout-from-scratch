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

