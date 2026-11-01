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

