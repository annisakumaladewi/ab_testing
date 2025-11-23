
# -----------------------------------------
# A/B Testing Automation Toolkit
# -----------------------------------------
# Contains reusable functions for SRM checks,
# A/A tests, proportion tests, mean tests,
# delta method, Simpson's paradox checks,
# and combined metric summarization.
# -----------------------------------------
# Notes:
# - Delta method (var_delta / ztest_delta) implementation is adapted from
#   public examples and articles on ratio metrics (e.g. Medium / course materials),
#   with modifications for this project.
# - Mean difference tests in my analysis notebooks use pingouin.pairwise_tests
#   directly (not re-implemented here).


import numpy as np
import pandas as pd
import scipy.stats as stats
from statsmodels.stats.proportion import proportions_ztest
from statsmodels.stats.power import FTestAnovaPower
import itertools
from itertools import combinations
from statsmodels.stats.power import NormalIndPower
from statsmodels.stats.proportion import proportion_effectsize
from statsmodels.stats.proportion import proportions_ztest, proportion_effectsize
from scipy.stats import norm


# SRM
def srm_check(df, assign_col, expected_props=None, alpha=0.05):
    """
    Performs a Sample Ratio Mismatch (SRM) chi-square test.

    Parameters
    ----------
    df : pandas.DataFrame
        Experiment data containing the assignment column.
    assign_col : str
        Column name describing group assignment.
    expected_props : dict or None
        Expected allocation proportions per group.
        If None, equal split is assumed.
    alpha : float
        Significance threshold.

    Returns
    -------
    dict
        Contains observed counts, expected counts,
        chi-square statistic, p-value, and significance flag.
    """
    # all groups, fixed order
    groups = sorted(df[assign_col].unique())

    # observed counts in that order
    obs = df[assign_col].value_counts().reindex(groups).values
    n = len(df)
    k = len(groups)

    # --- expected counts ---
    if expected_props is None:
        # equal split
        expected = np.repeat(n / k, k)
    else:
        # assume dict like {"A": 0.5, "B": 0.3, "C": 0.2}
        props = np.array([expected_props[g] for g in groups], dtype=float)
        # normalise in case they don't sum exactly to 1
        props = props / props.sum()
        expected = props * n

    chi2, pval = stats.chisquare(obs, expected)

    return {
        "groups": groups,
        "observed": obs,
        "expected": expected,
        "chi2": chi2,
        "p_value": pval,
        "significant": pval < alpha
    }

# Demographic Balance Test
def demographic_balance_tests(df, assign_col, covariates, dropna=False):
    """
    Chi-square balance checks for multiple covariates against the assignment column.

    Parameters
    ----------
    df : pd.DataFrame
    assign_col : str
        Column with variant assignment (e.g., 'checkout_page')
    covariates : list[str]
        Columns to test for balance (e.g., ['browser','gender'])
    dropna : bool
        If True, drop NA in covariate before testing; if False, NA is treated as a category.

    Returns
    -------
    pd.DataFrame with: covariate, chi2, dof, p_value, cramers_v, n_categories, min_expected, n_small_expected
    """
    rows = []
    for cov in covariates:
        sub = df[[assign_col, cov]].copy()
        if dropna:
            sub = sub.dropna(subset=[cov])
        # contingency table (NA as its own level if dropna=False)
        table = pd.crosstab(sub[assign_col], sub[cov], dropna=False)

        # chi-square test
        chi2, p, dof, expected = stats.chi2_contingency(table)
        n = table.values.sum()
        r, k = table.shape
        # Cramér's V (effect size)
        phi2 = chi2 / n
        cramers_v = np.sqrt(phi2 / max(1, min(r - 1, k - 1)))

        rows.append({
            "covariate": cov,
            "chi2": chi2,
            "dof": dof,
            "p_value": p,
            "cramers_v": cramers_v,
            "n_categories": k,
            "min_expected": float(expected.min()),
            "n_small_expected(<5)": int((expected < 5).sum())
        })
    return pd.DataFrame(rows).sort_values("p_value")

# A/A Test
def a_a_test(df, assign_col, variant, outcome_col, alpha=0.05, random_state=42):
  """
    Performs an A/A test by randomly splitting a single variant into two
    pseudo-groups and comparing their outcome rates using a two-sample
    proportion z-test.

    A/A testing is used to validate experiment setup, randomization
    quality, and statistical testing pipelines. Since both groups come
    from the same underlying variant, their conversion rates should be
    statistically similar. A significant difference may indicate data
    quality issues, implementation errors, or unintended bias.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataset containing experiment assignments and the outcome.
    assign_col : str
        Column name representing the experiment variant assignment
        (e.g., 'checkout_page').
    variant : str
        The specific variant label to extract and split into two subgroups
        (e.g., 'A').
    outcome_col : str
        Binary outcome variable used to compute conversion (e.g., 'purchased').
    alpha : float, optional (default=0.05)
        Significance threshold for the z-test.
    random_state : int, optional (default=42)
        Seed for reproducible random splitting.

    Returns
    -------
    dict
        Contains the statistical test results and group-level summaries:

        - 'variant' : the original variant selected for A/A splitting
        - 'groups' : list of generated subgroup names (e.g., ['A1', 'A2'])
        - 'z_stat' : z-statistic of the two-sample proportion test
        - 'p_value' : p-value corresponding to the z-statistic
        - 'alpha' : significance level used
        - 'significant' : boolean indicating if the test detects a difference
        - 'rates' : dictionary of conversion rates per subgroup
        - 'counts' : dictionary of number of successes per subgroup
        - 'nobs' : dictionary of subgroup sample sizes

    Notes
    -----
    - A non-significant result (p > alpha) suggests the A/A split behaves
      as expected.
    - A significant result (p < alpha) may indicate randomization issues,
      session contamination, or data pipeline inconsistencies.
    - This function assumes a binary outcome for the proportions z-test.
    """

  group = df[df[assign_col] == variant].copy()

  np.random.seed(random_state)
  group['aa_group'] = np.random.choice(['A1', 'A2'], size=len(group))

  AA_summary = group.groupby('aa_group')[outcome_col].agg(['sum', 'count'])

  count = AA_summary['sum'].values
  nobs = AA_summary['count'].values
  stat, pval = proportions_ztest(count, nobs)
  rates = (AA_summary['sum'] / AA_summary['count']).to_dict()

  return {
    "variant": variant,
    "groups": AA_summary.index.tolist(),      # like ['A1','A2']
    "z_stat": stat,
    "p_value": pval,
    "alpha": alpha,
    "significant": pval < alpha,
    "rates": rates,
    "counts": AA_summary['sum'].to_dict(),
    "nobs": AA_summary['count'].to_dict(),
}

# Simpson's Check
def simpsons_check(df, assign_col, segment_col, outcome_col):
    """
    Detects potential Simpson's paradox in an A/B test by comparing the
    overall ordering of variants to the ordering within each segment.

    Simpson's paradox occurs when the relationship between variants
    (e.g., conversion rates) reverses or changes once the data is
    stratified by a third variable. This function checks for such
    pattern changes.
    ...
    """

    overall_rates = df.groupby(assign_col)[outcome_col].mean()
    overall_sorted = overall_rates.sort_values(ascending=False)
    overall_order = overall_sorted.index.tolist()

    segment_rates = (
        df.groupby([segment_col, assign_col])[outcome_col]
        .mean()
        .unstack(assign_col)
    )

    rows = []
    for seg_value, row in segment_rates.iterrows():
        seg_sorted = row.sort_values(ascending=False)
        seg_order = seg_sorted.index.tolist()

        if seg_order == overall_order:
            pattern = "same_as_overall"
        elif seg_order == overall_order[::-1]:
            pattern = "reversed_overall"
        else:
            pattern = "different_order"

        rows.append({
            "segment": seg_value,
            "overall_order": overall_order,
            "segment_order": seg_order,
            "pattern": pattern,
            "rates": row.to_dict(),
        })

    result = pd.DataFrame(rows)
    return result

#ANOVA Power Test for Sample Size (Means)
def anova_power_from_data(df, group_col, metric_col, alpha=0.05, power=0.8):
    """
    Estimate ANOVA effect size (Cohen's f) from data and
    compute required sample size per group.
    """
    # 1. split data by group
    groups = [g[metric_col].dropna().values
              for _, g in df.groupby(group_col)]

    k = len(groups)  # number of groups

    # 2. group means
    means = np.array([g.mean() for g in groups])

    # 3. pooled standard deviation (across all groups)
    n_list = [len(g) for g in groups]
    pooled_var = sum((n-1)*g.var(ddof=1) for g, n in zip(groups, n_list)) / (sum(n_list) - k)
    sd = np.sqrt(pooled_var)

    # 4. Cohen's f for ANOVA
    grand_mean = means.mean()
    f = np.sqrt(((means - grand_mean)**2).mean()) / sd

    # 5. power analysis
    analysis = FTestAnovaPower()
    n_per_group = analysis.solve_power(effect_size=f,
                                       alpha=alpha,
                                       power=power,
                                       k_groups=k)

    return {
        "metric": metric_col,
        "groups": k,
        "effect_size_f": f,
        "required_n_per_group": n_per_group,
        "total_required_n": n_per_group * k
    }

# Sample Size for Proportion Test
def sample_size_proportion_multigroup(df, group_col, outcome_col, alpha=0.05, power=0.8):
    """
    Computes required sample size per group for detecting differences in proportions
    across 3+ variants in an A/B/n test.

    Parameters:
        df: dataframe
        group_col: column with A/B/C labels
        outcome_col: binary column (0/1)
        alpha: significance level
        power: desired power

    Returns:
        Dictionary with required sample sizes per pair + max needed
    """

    groups = df[group_col].unique()
    results = {}
    analysis = NormalIndPower()

    # Loop through all pairwise combinations (A-B, A-C, B-C)
    for g1, g2 in combinations(groups, 2):
        # Baseline proportion
        p1 = df[df[group_col] == g1][outcome_col].mean()
        # Treatment proportion
        p2 = df[df[group_col] == g2][outcome_col].mean()

        # Effect size (Cohen's h)
        h = proportion_effectsize(p1, p2)

        # Required sample size
        n = analysis.solve_power(effect_size=abs(h),
                                 power=power,
                                 alpha=alpha,
                                 alternative="two-sided")

        results[f"{g1} vs {g2}"] = n

    results["max_required_per_group"] = max(results.values())
    return results


# Pairwise Proportion Test

def pairwise_proportion_tests(df, group_col, outcome_col):

    grouped = df.groupby(group_col)
    successes = grouped[outcome_col].sum()
    nobs = grouped[outcome_col].count()
    p = successes / nobs
    groups = list(successes.index)

    results = []

    for g1, g2 in itertools.combinations(groups, 2):
        stat, pval = proportions_ztest([successes[g1], successes[g2]],
                                       [nobs[g1], nobs[g2]])
        h = proportion_effectsize(p[g1], p[g2])

        results.append({
            "comparison": f"{g1} vs {g2}",
            "p1": p[g1],
            "p2": p[g2],
            "z_stat": stat,
            "p_value": pval,
            "effect_size_h": h
        })

    return results

# FWER bonferroni correction
def add_bonferroni(results, alpha=0.05):
    m = len(results)                  # number of pairwise comparisons
    alpha_bonf = alpha / m            # adjusted significance threshold

    for r in results:
        p = r["p_value"]
        r["p_bonf"] = min(p * m, 1.0) # Bonferroni corrected p-value
        r["alpha_bonf"] = alpha_bonf  # store threshold for reference
        r["significant"] = r["p_bonf"] < alpha_bonf

    return results

# Delta Test
def var_delta(x, y):
    """
    Delta-method variance of a ratio-of-means metric: E[X]/E[Y]
    x, y are per-user numerator and denominator arrays for ONE variant.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    x_bar = np.mean(x)
    y_bar = np.mean(y)
    x_var = np.var(x, ddof=1)
    y_var = np.var(y, ddof=1)
    cov_xy = np.cov(x, y, ddof=1)[0, 1]

    # Same formula as in the slides / Medium article
    var_ratio = (
        x_var / y_bar**2
        + y_var * (x_bar**2 / y_bar**4)
        - 2 * cov_xy * (x_bar / y_bar**3)
    ) / len(x)

    return var_ratio

def ztest_delta(x_control, y_control, x_treatment, y_treatment, alpha=0.05):
    x_control = np.asarray(x_control, dtype=float)
    y_control = np.asarray(y_control, dtype=float)
    x_treatment = np.asarray(x_treatment, dtype=float)
    y_treatment = np.asarray(y_treatment, dtype=float)

    mean_control = x_control.sum() / y_control.sum()
    mean_treatment = x_treatment.sum() / y_treatment.sum()

    var_control = var_delta(x_control, y_control)
    var_treatment = var_delta(x_treatment, y_treatment)

    var_diff = var_control + var_treatment
    se = np.sqrt(var_diff)

    diff = mean_treatment - mean_control
    z_stat = diff / se

    p_value = 2 * (1 - norm.cdf(abs(z_stat)))

    z_crit = norm.ppf(1 - alpha / 2)
    lower = diff - z_crit * se
    upper = diff + z_crit * se

    return {
        "mean_control": mean_control,
        "mean_treatment": mean_treatment,
        "var_control": var_control,
        "var_treatment": var_treatment,
        "diff": diff,
        "diff_CI": (lower, upper),
        "z_stat": z_stat,
        "p_value": p_value,
    }

def pairwise_delta_test(df, col_var, num_col, denom_col, user_col):
    groups = df[col_var].unique()
    results = []

    for i in range(len(groups)):
        for j in range(i+1, len(groups)):
            g1 = groups[i]
            g2 = groups[j]

            df1 = df[df[col_var] == g1]
            df2 = df[df[col_var] == g2]

            per_user_1 = df1.groupby(user_col)[[num_col, denom_col]].sum()
            per_user_2 = df2.groupby(user_col)[[num_col, denom_col]].sum()

            x_control = per_user_1[num_col].values
            y_control = per_user_1[denom_col].values

            x_treatment = per_user_2[num_col].values
            y_treatment = per_user_2[denom_col].values

            delta_res = ztest_delta(x_control, y_control, x_treatment, y_treatment)

            # here adapt to how ztest_delta returns results
            diff = delta_res['diff']
            pval = delta_res['p_value']

            results.append({
                "comparison": f"{g1} vs {g2}",
                "diff": diff,
                "p_value": pval
            })

    return results

# Combined results
def combine_results(result_tables, test_types, alpha=0.05):
    """
    Combine multiple result tables (means, proportions, delta, etc.)
    into one standardized summary.

    Parameters
    ----------
    result_tables : list of pd.DataFrame
        Each table must at least have columns:
        - 'comparison'
        - 'p_value'
        - 'effect_size'
        Optionally:
        - 'metric' (e.g. 'order_value', 'purchased', 'ratio_metric')
    test_types : list of str
        Label for each table, same length as result_tables.
        Examples: 'mean', 'proportion', 'delta'
    alpha : float
        Significance threshold, default 0.05.

    Returns
    -------
    pd.DataFrame
        Combined table with columns:
        - comparison
        - metric (if present in inputs)
        - p_value
        - effect_size
        - test_type
        - significant
        - effect_size_label
    """
    if len(result_tables) != len(test_types):
        raise ValueError("result_tables and test_types must have the same length")

    all_tables = []

    for table, ttype in zip(result_tables, test_types):
        tmp = table.copy()

        # add which kind of test this came from
        tmp["test_type"] = ttype

        # significance flag
        tmp["significant"] = tmp["p_value"] < alpha

        # interpret effect size (if present)
        if "effect_size" in tmp.columns:
            if ttype in ["mean", "proportion"]:
                es_abs = tmp["effect_size"].abs()

                conditions = [
                    es_abs < 0.1,
                    (es_abs >= 0.1) & (es_abs < 0.3),
                    (es_abs >= 0.3) & (es_abs < 0.5),
                    es_abs >= 0.5,
                ]
                labels = [
                    "very small",
                    "small",
                    "medium",
                    "large",
                ]
                tmp["effect_size_label"] = np.select(
                    conditions, labels, default="unknown"
                )

            elif ttype == "delta":
                # raw difference on ratio metric – interpretation is context-specific
                tmp["effect_size_label"] = "raw_diff"

            else:
                tmp["effect_size_label"] = "unknown"
        else:
            tmp["effect_size_label"] = "missing"

        all_tables.append(tmp)

    combined = pd.concat(all_tables, ignore_index=True)

    # nice ordering if 'metric' exists
    sort_cols = ["metric", "comparison"] if "metric" in combined.columns else ["comparison"]
    combined = combined.sort_values(sort_cols)

    return combined

# Public API
__all__ = [
    "srm_check",
    "demographic_balance_tests",
    "a_a_test",
    "simpsons_check",
    "anova_power_from_data",
    "sample_size_proportion_multigroup",
    "pairwise_proportion_tests",
    "add_bonferroni",
    "var_delta",
    "ztest_delta",
    "pairwise_delta_test",
    "combine_results",
]
