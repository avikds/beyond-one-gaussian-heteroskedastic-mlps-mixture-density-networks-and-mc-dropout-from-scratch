"""
Beyond One Gaussian: Heteroskedastic MLPs, Mixture Density Networks and MC Dropout from Scratch scaffold.

Run this with: python scaffold.py
Uses functions defined in model.py.
"""

from model import *  # noqa: F401, F403 (pulls in your solution functions)

"""Beyond One Gaussian: Heteroskedastic MLPs, Mixture Density Networks and MC Dropout from Scratch.

The whole study for one data seed (seed 0). The bracketed ranges are the same numbers over data seeds
0-4 (0-19 for the small-data experiment), measured on this image when the project was calibrated. Lower NLL is better.
"""
import time

R = {
    "s_oracle": "0.67-0.69", "s_fixed": "0.93-0.97", "s_two": "0.70-0.72",
    "i_oracle": "-1.29 to -1.25", "i_gauss": "-0.62 to -0.57", "i_ens": "-0.62 to -0.57", "i_mc": "-0.58 to -0.54",
    "i_mole": "-0.93 to -0.89", "i_hme": "-0.98 to -0.89", "i_mdn2": "-1.09 to -1.03", "i_mdn3": "-1.13 to -1.07", "i_mdn5": "-1.12 to -1.05",
    "single": "0.72-2.08", "ensemble": "0.71-1.72", "mc": "0.76-0.97", "ws": "0.76-0.95",
}


def main() -> None:
    t0 = time.time()
    rep = beyond_one_gaussian_report(seed=0)
    s, i, m = rep["series"], rep["inverse"], rep["small"]

    print("1. Growing noise, Figure 13.6 (1000 training points, 5000 test points)")
    print(f"   true density                 NLL {s['oracle']:.3f}   [{R['s_oracle']}]")
    for key, label, r in (("fixed", "fixed sigma (13.6b)", "s_fixed"), ("two_head", "two heads (eq 13.21, 13.6a)", "s_two")):
        d = s[key]
        print(f"   {label:28s} NLL {d['nll']:.3f}   [{R[r]}]   overall 90% coverage {d['cov90']:.3f}, width {d['width90']:.2f}")
    print(f"   learned sigma vs the truth: RMSE {s['sigma_rmse']:.3f}   mean vs the truth: fixed {s['mean_rmse']['fixed']:.3f}, two heads {s['mean_rmse']['two_head']:.3f}")
    print("   90% interval coverage by region of x:")
    print("   region            fixed sigma   two heads   two-head width")
    for a, b in zip(s["fixed_regions"], s["two_head_regions"]):
        print(f"   [{a['lo']:>4.0f}, {a['hi']:>4.0f})       {a['coverage']:>8.3f}    {b['coverage']:>8.3f}    {b['width']:>8.2f}")
    print("   The fixed sigma looks calibrated overall and is wrong in every region: too wide on the left, too narrow on the right.")

    print("\n2. One-to-many inverse problem, Figure 13.23 (1000 training points, 5000 test points)")
    print(f"   true density                 NLL {i['oracle']:.3f}   [{R['i_oracle']}]")
    rows = (("gaussian", "one Gaussian (eq 13.21)", "i_gauss"), ("ensemble_gaussian", "3 Gaussians, deep ensemble", "i_ens"),
            ("mc_dropout_gaussian", "one Gaussian, MC dropout", "i_mc"), ("mole", "3 linear experts by EM", "i_mole"), ("hme", "2-level tree of 4 linear experts", "i_hme"))
    for key, label, r in rows:
        print(f"   {label:34s} NLL {i[key]['nll']:.3f}   [{R[r]}]")
    for K in (2, 3, 5):
        print(f"   {'MDN, K = ' + str(K):34s} NLL {i['mdn'][K]['nll']:.3f}   [{R['i_mdn' + str(K)]}]")
    print("   Ensembles and dropout do not help: every member learns the same blurry average.")

    print("\n   Point predictions (Figure 13.23d): RMSE, and how often the prediction is a value that can occur")
    for key, label in (("mole", "linear experts"), ("mdn3", "MDN, K = 3")):
        p = i["points"][key]
        print(f"   {label:16s} mean: RMSE {p['rmse_mean']:.3f}, possible {p['support_mean']:.3f}    most active expert: RMSE {p['rmse_mode']:.3f}, possible {p['support_mode']:.3f}")
    print("   The mean always wins on squared error. The most active expert wins on landing where data can occur.")

    print("\n   Shortest 90% sets (HPD, eqs 4.193-4.194):")
    print(f"   one Gaussian: length {i['sets']['gaussian']['length']:.3f}, coverage {i['sets']['gaussian']['coverage']:.3f}")
    print(f"   MDN, K = 3:   length {i['sets']['mdn3']['length']:.3f}, coverage {i['sets']['mdn3']['coverage']:.3f}   (its central 90% interval has width {i['mdn'][3]['width90']:.3f})")
    print(f"   true density: length {i['oracle_set_length']:.3f}")

    print("\n3. 200 points and a hole at 10 < x < 25: marginalizing over models (eqs 13.96-13.97)")
    print(f"   dropout rate chosen on held-out likelihood: {m['best_p']}")
    print("   model            NLL outside the hole          NLL inside   90% coverage inside")
    for key, label, r in (("single", "one network", "single"), ("ensemble", "5-member ensemble", "ensemble"), ("mc_dropout", "MC dropout", "mc"), ("weight_scaled", "weight scaling", "ws")):
        d = m[key]
        print(f"   {label:18s} {d['nll_out']:.3f}   [{R[r]:>9s}]    {d['nll_in']:>9.2f}    {d['cov90_in']:>8.3f}")
    print("   Model uncertainty (standard deviation of the means), inside vs outside the hole:")
    for key, label in (("ensemble", "ensemble"), ("mc_dropout", "MC dropout")):
        d = m[key]
        print(f"   {label:12s} {d['epistemic_in']:.3f} vs {d['epistemic_out']:.3f}, ratio {d['epistemic_in'] / d['epistemic_out']:.2f}")
    print("   Over 20 data seeds that ratio has median 3.2 for the ensemble (2.0-11.9) and 1.02 for MC dropout (0.83-1.18): only the ensemble notices the hole.")
    print("   Neither is honest there: on average their 90% intervals cover 65% (ensemble) and 58% (MC dropout) inside the hole.")

    print(f"\nstudy finished in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()

