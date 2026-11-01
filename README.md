# Beyond One Gaussian: Heteroskedastic MLPs, Mixture Density Networks and MC Dropout from Scratch

Most regression models assume the output is one Gaussian with one noise level. This project works through chapter 13 of Murphy's Probabilistic Machine Learning to find out what that assumption costs, on two problems where it fails and whose true densities you can compute, so every model is scored against the truth. It is built in tinygrad, with every method written by you. You start with the growing-noise series of Figure 13.6, the one-to-many inverse problem of Figure 13.23, a stable softplus, a small tanh network with dropout masks, a hand-written Adam and one TinyJit training loop. Every model then returns a Gaussian mixture per input, so a single scoreboard covers them all: negative log-likelihood by logsumexp, probability integral transform, central intervals and coverage by region of the input. On that scoreboard you compare a fixed sigma with the two-headed MLP of eq 13.21; a mixture of linear experts fitted by EM with restarts (eqs 13.109-13.110); mixture density networks, with closed-form gradients checked against autograd and highest-density sets that respect the gaps between branches (eqs 13.106-13.108); a 2-level hierarchical mixture of experts; Monte Carlo dropout against weight scaling, with the rate chosen on held-out likelihood (eq 13.96); and a deep ensemble as a second approximation to the Bayesian predictive (eq 13.97), with the law of total variance splitting data noise from model uncertainty. Headline numbers, mean test NLL over five data seeds on the prod CPU image. Growing noise: fixed sigma 0.95, two heads 0.71, true density 0.67. The fixed sigma covers about 90% with its 90% interval overall, but 100% on the quiet left and 74% on the noisy right, while the two heads stay between 87% and 90%. Inverse problem: one Gaussian -0.59, an ensemble of them -0.59, the same with dropout -0.56, three linear experts -0.90, a tree of four -0.92, MDN with K=3 -1.09, true density -1.27. The MDN's 90% highest-density set has length 0.33, matching the true density, against 0.59 for one Gaussian. You also test the book's mean-versus-mode comparison. The mean wins on RMSE on every seed (0.213 against 0.238 for linear experts), while the most active expert is a value that can actually occur for 98% of inputs against 44% for the mean. With 200 points and a hole in the input range (20 data seeds), five ensemble members lower the NLL outside the hole from 0.89 to 0.86, and model uncertainty is a median 3.2 times larger inside the hole for the ensemble but only 1.02 times for MC dropout, although their nominal 90% intervals cover only 65% and 58% there. The assembled script runs the whole study for one data seed in about two minutes.

## How to run

```bash
python scaffold.py
```

## Steps

- [x] **1.** make_growing_noise_data
- [x] **2.** make_inverse_data
- [x] **3.** inverse_oracle_nll
- [x] **4.** Standardizer
- [x] **5.** softplus
- [x] **6.** DenseNet

---

Built on Deep-ML.
