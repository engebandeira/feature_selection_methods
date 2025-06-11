import pandas as pd
import numpy as np

def calculate_iv(df,num_vars,cat_vars,target):
    """
    Calculate the Information Value (IV) of numerical and categorical features 
    using supervised binning from the `optbinning` library.

    This function applies optimal binning to each variable, computes the IV, 
    and returns a sorted DataFrame with the IV and the strength classification.

    Args:
        df (pd.DataFrame): The input dataset containing the features and target.
        num_vars (list of str): List of names of numerical features to evaluate.
        cat_vars (list of str): List of names of categorical features to evaluate.
        target (str): Name of the binary target column (0 = "good", 1 = "bad").

    Returns:
        pd.DataFrame: A DataFrame with the following columns:
            - 'FEATURE': Name of the feature.
            - 'IV': Calculated Information Value.
            - 'STRENGTH': Qualitative strength of the variable based on IV:
                * IV < 0.02 → 'useless'
                * 0.02 ≤ IV < 0.1 → 'weak'
                * 0.1 ≤ IV < 0.3 → 'medium'
                * 0.3 ≤ IV < 0.5 → 'strong'
                * IV ≥ 0.5 → 'Suspiciously strong'
            - 'TYPE': Return the type of the feature (categorical or numerical)

    Notes:
        - Uses `monotonic_trend='descending'` in OptimalBinning for all features.
        - Assumes `df[target]` is binary with 0 and 1 values.
        - Uses supervised binning, so the IV reflects the separation power with respect to the target.

    Example:
        df_iv = calculate_iv(df, num_vars=['age', 'income'], cat_vars=['state'], target='default')
    """
    from optbinning import OptimalBinning

    dict_vars = {**{k:'numerical' for k in num_vars},
                 **{k:'categorical' for k in cat_vars}}

    iv_values = {}

    for k,i in dict_vars.items():
    
        binning = OptimalBinning(name=k, dtype=i, monotonic_trend='descending')
        binning.fit(df[k], df[target])

        binning.binning_table.build()
        iv = binning.binning_table.iv
        
        iv_values[k] = iv

    df_iv = pd.DataFrame.from_dict(iv_values, orient='index').reset_index()
    df_iv.columns = ['FEATURE','IV']

    df_iv = df_iv.sort_values('IV',ascending=False)

    df_iv['STRENGTH'] = df_iv['IV'].apply(lambda x: 'useless' if x < 0.02 else
                                                    'weak' if x < 0.1 else
                                                    'medium' if x < 0.3 else
                                                    'strong' if x < 0.5 else
                                                    'Suspiciously strong')
    df_iv['TYPE'] = df_iv['FEATURE'].map(dict_vars)

    return df_iv

def classification_mutual_information(df,num_vars,cat_vars,target):
    """
    Compute Mutual Information (MI) between features and a binary classification target.

    This function calculates the mutual information score for both numerical and 
    categorical features with respect to a binary classification target variable.
    It returns a DataFrame with MI values, variable types, and feature names sorted
    by importance.

    Args:
        df (pd.DataFrame): Input dataset containing features and target variable.
        num_vars (list of str): List of numerical feature column names.
        cat_vars (list of str): List of categorical feature column names.
        target (str): Name of the binary target column.

    Returns:
        pd.DataFrame: A DataFrame with the following columns:
            - 'FEATURE': Name of the feature.
            - 'MUTUAL_INFO': Calculated mutual information score.
            - 'TYPE': Feature type ('numerical' or 'categorical').

    Notes:
        - The target variable must be binary (e.g., 0 and 1).
        - Categorical variables are automatically encoded by scikit-learn.
        - Higher MI values indicate greater dependency between the feature and the target.

    Example:
        mi_df = classification_mutual_information(
            df=data,
            num_vars=['age', 'income'],
            cat_vars=['gender', 'region'],
            target='default'
        )
    """
    from sklearn.feature_selection import mutual_info_classif

    dict_vars = {
                 **{k:'numerical' for k in num_vars},
                 **{k:'categorical' for k in cat_vars}
                }
    
    X = df[[*num_vars,*cat_vars]]
    y = df[target]

    # Calcular MI
    mi = mutual_info_classif(X, y, discrete_features='auto')

    # Organizar em DataFrame
    mi_df = pd.DataFrame({'FEATURE': X.columns, 'MUTUAL_INFO': mi})
    mi_df = mi_df.sort_values('MUTUAL_INFO', ascending=False)
    mi_df['TYPE'] = mi_df['FEATURE'].map(dict_vars)

    return mi_df

def chi2_dependence(df, num_vars, cat_vars, target):
    """
    Perform Chi-Square (χ²) tests of independence between input variables and a categorical target,
    including Cramér's V to measure strength of association.

    Parameters
    ----------
    df : pandas.DataFrame
        The input dataset containing the features and target.

    num_vars : list of str
        List of numerical feature names.

    cat_vars : list of str
        List of categorical feature names.

    target : str
        Name of the target variable (categorical).

    Returns
    -------
    pandas.DataFrame
        DataFrame with columns:
        - 'FEATURE': Name of the feature.
        - 'CHI2_SCORE': Chi-Square statistic.
        - 'P_VALUE': p-value from the test.
         - 'CRAMERS_V': Strength of association (0 to 1).
        - 'TYPE': 'numerical' or 'categorical'.
        - 'SIGNIFICANCE': 'associated' if p < 0.05, otherwise 'not_associated'.
    """
    from optbinning import OptimalBinning
    from sklearn.feature_selection import SelectKBest, chi2
    from sklearn.preprocessing import LabelEncoder
    from scipy.stats import chi2_contingency

    dict_vars = {**{k: 'numerical' for k in num_vars},
                 **{k: 'categorical' for k in cat_vars}}

    chi_tests = []

    for k, i in dict_vars.items():
        # Binning
        binning = OptimalBinning(name=k, dtype=i, monotonic_trend='descending')
        binning.fit(df[k], df[target])
        X = binning.transform(df[k], metric='bins')

        # Label encoding
        X_encoded = LabelEncoder().fit_transform(X)
        y = df[target].astype(str)  # Ensure categorical

        # Chi2 test
        selector = SelectKBest(score_func=chi2, k='all')
        selector.fit(X_encoded.reshape(-1, 1), y)

        chi2_score = selector.scores_[0]
        p_value = selector.pvalues_[0]

        # Cramér's V
        contingency_table = pd.crosstab(X_encoded, y)
        chi2_stat, _, _, _ = chi2_contingency(contingency_table)
        n = contingency_table.sum().sum()
        r, k_ = contingency_table.shape
        min_dim = min(r - 1, k_ - 1) if min(r, k_) > 1 else 1  # prevent division by 0
        cramers_v = np.sqrt(chi2_stat / (n * min_dim)) if min_dim > 0 else 0

        # Results
        chi_tests.append({
            'FEATURE': k,
            'CHI2_SCORE': chi2_score,
            'P_VALUE': p_value,
            'CRAMERS_V': cramers_v,
            'TYPE': i,
            'SIGNIFICANCE': 'associated' if p_value < 0.05 else 'not_associated',
        })

    chi_table = pd.DataFrame(chi_tests).sort_values('CRAMERS_V', ascending=False)
    return chi_table

def testt_mann_whitney(df,num_vars,target,alpha):
    """
    Performs statistical tests to compare numerical variables between two groups defined by a binary target variable.

    For each numeric variable in `num_vars`, the function checks whether the distributions in the two groups 
    (defined by target == 0 and target == 1) are normal using the Shapiro-Wilk test. 
    If both groups are normally distributed, a Welch's t-test (t-test with unequal variances) is performed. 
    Otherwise, the non-parametric Mann-Whitney U test is used.

    Parameters:
    -----------
    df : pandas.DataFrame  
        The dataset containing the variables.
    num_vars : list of str  
        List of numerical variable names to be tested.
    target : str  
        Name of the binary target variable (0/1) defining the comparison groups.
    alpha : float  
        Significance level (e.g., 0.05) used to determine statistical significance.

    Returns:
    --------
    pandas.DataFrame  
        A DataFrame with the following columns for each variable tested:
        - FEATURE: name of the variable
        - TEST: type of test used ("t-test" or "Mann-Whitney")
        - P_VALUE: p-value from the statistical test
        - SIGNIFICANCE: boolean indicating whether the result is statistically significant (p < alpha)

    Notes:
    ------
    - The normality test is performed using a sample of up to 500 observations per group.

    Example:
    --------
    >>> testt_mann_whitney(df=data, num_vars=["age", "income"], target="defaulted", alpha=0.05)
    """
    from scipy.stats import ttest_ind, mannwhitneyu, shapiro

    results = []
    
    for var in num_vars:
        group0 = df[df[target] == 0][var].dropna()
        group1 = df[df[target] == 1][var].dropna()

        normal0 = shapiro(group0.sample(min(len(group0), 500)))[1] > 0.05
        normal1 = shapiro(group1.sample(min(len(group1), 500)))[1] > 0.05

        if normal0 and normal1:
            _, p = ttest_ind(group0, group1, equal_var=False)
            test = "t-test"
        else:
            _, p = mannwhitneyu(group0, group1, alternative='two-sided')
            test = "Mann-Whitney"

        results.append({
            'FEATURE': var,
            'TEST': test,
            'P_VALUE': p,
            'SIGNIFICANCE': p < alpha
        })

    return pd.DataFrame(results).sort_values('P_VALUE')


def forward_selection_rf_auc(df,vars, target, cv=5, random_state=42, verbose=True):
    """
    Perform Forward Feature Selection using Random Forest and AUC as the evaluation metric.

    Parameters
    ----------
    df : pd.DataFrame
        A DataFrame containing the feature columns and the target variable.
    vars: list of str
        the list of variables to be tested.
    target : str
        The name of the target column in the DataFrame (must be binary).
    cv : int, default=5
        Number of cross-validation folds to evaluate AUC.
    random_state : int, default=42
        Random state used for reproducibility in the Random Forest classifier.
    verbose : bool, default=True
        Whether to print progress messages at each step.

    Returns
    -------
    selected_features : list of str
        The list of selected features in the order they were added.
    auc_history : list of float
        AUC scores obtained after each feature addition.

    Notes
    -----
    - This function performs greedy forward selection: at each step, it adds the feature
      that leads to the highest increase in mean AUC across cross-validation folds.
    - The process stops when no feature addition improves the current best AUC.
    - Useful for feature ranking and dimensionality reduction based on model performance.
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import cross_val_score
    from IPython.display import clear_output

    def plot_auc_progress(selected_features, auc_history):
        """
        Plot the AUC score progression during forward feature selection.

        Parameters
        ----------
        selected_features : list of str
            The list of features selected at each step.
        auc_history : list of float
            The corresponding AUC score after each feature addition.
        """
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(auc_history) + 1), auc_history, marker='o', linestyle='--')
        plt.xticks(ticks=range(1, len(selected_features) + 1), labels=selected_features, rotation=45)
        plt.xlabel("Features Added")
        plt.ylabel("Cross-Validated AUC")
        plt.title("AUC Progression During Forward Selection")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=random_state,
        n_jobs=-1
    )

    X = df[vars]
    y = df[target]

    selected_features = []
    remaining_features = list(X.columns)
    best_overall_auc = 0.0
    auc_history = []

    while len(remaining_features) > 0:
        best_auc_this_iteration = 0.0
        best_feature_this_iteration = None

        if verbose:
            print(f"\nSelected Features: {selected_features}")
            print("Testing addition of remaining features:")

        for feature in remaining_features:
            features_to_test = selected_features + [feature]
            auc_scores = cross_val_score(
                model,
                X[features_to_test],
                y,
                cv=cv,
                scoring='roc_auc'
            )
            mean_auc = np.mean(auc_scores)
            
            if verbose:
                print(f"  - Testing '{feature}': Average AUC = {mean_auc:.4f}")
            
            if mean_auc > best_auc_this_iteration:
                best_auc_this_iteration = mean_auc
                best_feature_this_iteration = feature

        if best_auc_this_iteration > best_overall_auc:
            selected_features.append(best_feature_this_iteration)
            remaining_features.remove(best_feature_this_iteration)
            best_overall_auc = best_auc_this_iteration
            auc_history.append(best_overall_auc)

            clear_output(wait=True)
            if verbose:
                print(f"\n🎉 ADDED FEATURE: '{best_feature_this_iteration}'")
                print(f"New Best Overall AUC: {best_overall_auc:.4f}")
                print("-" * 30)
        else:
            if verbose:
                print("\n🛑 No AUC improvement. Stopping feature selection.")
            break

    plot_auc_progress(selected_features,auc_history)
    
    return selected_features, auc_history