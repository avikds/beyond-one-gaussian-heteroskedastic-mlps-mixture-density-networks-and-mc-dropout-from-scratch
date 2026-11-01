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
- [x] **7.** Adam
- [x] **8.** fit
- [x] **9.** mixture_logpdf
- [x] **10.** pit_values
- [x] **11.** interval_report
- [x] **12.** coverage_by_region
- [x] **13.** gaussian_nll
- [x] **14.** fit_fixed_sigma
- [x] **15.** fit_two_head
- [x] **16.** moe_e_step
- [x] **17.** moe_m_step
- [x] **18.** fit_mole
- [x] **19.** point_report
- [x] **20.** mdn_nll
- [x] **21.** mdn_grads
- [x] **22.** fit_mdn
- [x] **23.** density_set_report
- [x] **24.** fit_hme
- [x] **25.** fit_dropout
- [x] **26.** mc_dropout_predict
- [x] **27.** tune_dropout_rate
- [x] **28.** fit_ensemble
- [x] **29.** uncertainty_split
- [x] **30.** beyond_one_gaussian_report

## Results

```
1. Growing noise, Figure 13.6 (1000 training points, 5000 test points)
   true density                 NLL 0.685   [0.67-0.69]
   fixed sigma (13.6b)          NLL 0.965   [0.93-0.97]   overall 90% coverage 0.895, width 2.02
   two heads (eq 13.21, 13.6a)  NLL 0.719   [0.70-0.72]   overall 90% coverage 0.881, width 1.78
   learned sigma vs the truth: RMSE 0.061   mean vs the truth: fixed 0.181, two heads 0.149
   90% interval coverage by region of x:
   region            fixed sigma   two heads   two-head width
   [ -20,    0)          0.998       0.899        0.74
   [   0,   20)          0.966       0.881        1.44
   [  20,   40)          0.862       0.871        2.14
   [  40,   60)          0.750       0.873        2.83
   The fixed sigma looks calibrated overall and is wrong in every region: too wide on the left, too narrow on the right.

2. One-to-many inverse problem, Figure 13.23 (1000 training points, 5000 test points)
   true density                 NLL -1.258   [-1.29 to -1.25]
   one Gaussian (eq 13.21)            NLL -0.577   [-0.62 to -0.57]
   3 Gaussians, deep ensemble         NLL -0.578   [-0.62 to -0.57]
   one Gaussian, MC dropout           NLL -0.546   [-0.58 to -0.54]
   3 linear experts by EM             NLL -0.889   [-0.93 to -0.89]
   2-level tree of 4 linear experts   NLL -0.895   [-0.98 to -0.89]
   MDN, K = 2                         NLL -1.042   [-1.09 to -1.03]
   MDN, K = 3                         NLL -1.085   [-1.13 to -1.07]
   MDN, K = 5                         NLL -1.069   [-1.12 to -1.05]
   Ensembles and dropout do not help: every member learns the same blurry average.

   Point predictions (Figure 13.23d): RMSE, and how often the prediction is a value that can occur
   linear experts   mean: RMSE 0.217, possible 0.429    most active expert: RMSE 0.245, possible 0.982
   MDN, K = 3       mean: RMSE 0.205, possible 0.783    most active expert: RMSE 0.254, possible 0.999
   The mean always wins on squared error. The most active expert wins on landing where data can occur.

   Shortest 90% sets (HPD, eqs 4.193-4.194):
   one Gaussian: length 0.586, coverage 0.910
   MDN, K = 3:   length 0.330, coverage 0.896   (its central 90% interval has width 0.520)
   true density: length 0.332

3. 200 points and a hole at 10 < x < 25: marginalizing over models (eqs 13.96-13.97)
   dropout rate chosen on held-out likelihood: 0.003
   model            NLL outside the hole          NLL inside   90% coverage inside
   one network        0.817   [0.72-2.08]         1.13       0.725
   5-member ensemble  0.790   [0.71-1.72]         1.02       0.798
   MC dropout         0.759   [0.76-0.97]         1.66       0.643
   weight scaling     0.758   [0.76-0.95]         1.72       0.636
   Model uncertainty (standard deviation of the means), inside vs outside the hole:
   ensemble     0.103 vs 0.049, ratio 2.09
   MC dropout   0.104 vs 0.104, ratio 1.00
   Over 20 data seeds that ratio has median 3.2 for the ensemble (2.0-11.9) and 1.02 for MC dropout (0.83-1.18): only the ensemble notices the hole.
   Neither is honest there: on average their 90% intervals cover 65% (ensemble) and 58% (MC dropout) inside the hole.

study finished in 435 s
```
