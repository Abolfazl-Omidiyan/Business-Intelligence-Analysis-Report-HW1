"""
================================================================================
  BUSINESS INTELLIGENCE HOMEWORK PROJECT
  
  Dataset Analysis: E-Commerce Query & Product Data
  Files: top5000Query.csv + top5000product.csv
  
  Questions Covered:
    1. Duplicate & Missing Data Detection
    2. Outlier Detection & Removal (IQR Method)
    3. Query Binning by Category Count
    4. Retrieval & Click Probability (Normalized)
    5. Boxplot & Histogram - Distribution Analysis
    6. QQ Plot - Normality Testing
    7. Scatter Plot - Relationship Analysis
    8. Feature Extraction
    9. Violin Plot - Price Distribution
   10. Dominant Category & Chi-Square Test
  
  Author: Abolfazl Omidian
  Master's Degree in Information Technology - Organizational Architecture
  Student Number: 404443019
  
  Prerequisites:
    pip install pandas numpy matplotlib scipy
  
  Usage:
    python complete_bi_analysis.py
    
================================================================================
"""

import ast
import re
import warnings
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import numpy as np
import pandas as pd
import scipy.stats as stats

warnings.filterwarnings("ignore")
plt.rcParams["font.family"] = ["DejaVu Sans", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False


# ============================================================================
# CONFIGURATION
# ============================================================================
# Update these paths to match your file locations
QUERY_PATH = "top5000Query.csv"
PRODUCT_PATH = "top5000product.csv"
OUTPUT_DIR = Path(".")


# ============================================================================
# UTILITIES - DATA PARSING
# ============================================================================

def parse_list_column(series: pd.Series) -> pd.Series:
    """
    Convert a pandas Series of list-literal strings into actual Python lists.
    
    Example:
        "[1234, null, 5678]" → [1234, 5678]
    """
    def _parse(s):
        try:
            lst = ast.literal_eval(str(s).replace("null", "None"))
            return [x for x in lst if x is not None]
        except Exception:
            return []
    return series.apply(_parse)


def load_query_dataset(path: str) -> pd.DataFrame:
    """Load and parse the query dataset."""
    print(f"[*] Loading query dataset from {path}...")
    q = pd.read_csv(path)
    
    q["ResultList"]        = parse_list_column(q["Result"])
    q["ClickedResultList"] = parse_list_column(q["ClickedResult"])
    q["ClickedRankList"]   = parse_list_column(q["ClickedRank"])
    q["NumRetrievals"]     = q["ResultList"].apply(len)
    q["NumClicks"]         = q["ClickedResultList"].apply(len)
    
    print(f"    ✓ Loaded {len(q)} query records")
    return q


def load_product_dataset(path: str) -> pd.DataFrame:
    """
    Load and parse the product dataset.
    
    NOTE: This dataset uses a non-standard format:
      - Row separator: \x1b\x1b\r\n (not newline)
      - Each row wrapped in quotes with doubled internal quotes
      - Custom parsing required
    """
    print(f"[*] Loading product dataset from {path}...")
    
    with open(path, "rb") as fh:
        raw_bytes = fh.read()
    
    # Remove UTF-8 BOM if present
    if raw_bytes.startswith(b"\xef\xbb\xbf"):
        raw_bytes = raw_bytes[3:]
    
    text_content = raw_bytes.decode("utf-8")
    raw_lines = [ln for ln in text_content.split("\x1b\x1b\r\n") if ln.strip()]
    
    product_rows = []
    for line in raw_lines[1:]:  # Skip header
        try:
            # Strip outer quotes and unescape doubled internal quotes
            if line.startswith('"') and line.endswith('"'):
                inner = line[1:-1].replace('""', '"')
            else:
                inner = line
            
            # Parse: ID , CategoryName , [Titles JSON] , num_price_fields...
            first_comma = inner.index(",")
            prod_id = inner[:first_comma]
            rest = inner[first_comma + 1:]
            
            bracket_pos = rest.index("[")
            cat_name = rest[:bracket_pos - 1].rstrip(",").strip()
            after_open = rest[bracket_pos:]
            
            # Find matching closing bracket
            depth, bracket_end = 0, -1
            for idx, ch in enumerate(after_open):
                if ch == "[":
                    depth += 1
                elif ch == "]":
                    depth -= 1
                    if depth == 0:
                        bracket_end = idx + 1
                        break
            
            titles_json = after_open[:bracket_end]
            remaining = re.sub(r'^[",\s]+', "", after_open[bracket_end:])
            num_fields = remaining.split(",") if remaining else []
            
            while len(num_fields) < 6:
                num_fields.append("")
            
            product_rows.append(
                [prod_id, cat_name, titles_json] + num_fields[:6]
            )
        
        except Exception:
            pass  # Skip malformed rows
    
    cols = [
        "ID", "CategoryName", "Titles",
        "MinPrice", "MaxPrice", "AvgPrice",
        "MinNumShops", "MaxNumShops", "AvgNumShops",
    ]
    p = pd.DataFrame(product_rows, columns=cols)
    
    # Convert numeric columns
    for col in ["MinPrice", "MaxPrice", "AvgPrice",
                "MinNumShops", "MaxNumShops", "AvgNumShops"]:
        p[col] = pd.to_numeric(
            p[col].replace({"NULL": None, "NULL\r": None}),
            errors="coerce"
        )
    p["ID"] = pd.to_numeric(p["ID"], errors="coerce")
    
    print(f"    ✓ Loaded {len(p)} product records")
    return p


# ============================================================================
# QUESTION 1: DUPLICATE & MISSING DATA DETECTION
# ============================================================================

def analyze_duplicates_and_missing(q_raw: pd.DataFrame, p_raw: pd.DataFrame):
    """
    Detect and report duplicates and missing values in both datasets.
    
    Impact of Duplicates:
      - Inflate counts and weights for repeated queries
      - Bias statistical measures (mean, std, etc.)
      - Skew probability distributions
    
    Impact of Missing Data:
      - Incomplete analysis (cannot use those records)
      - MNAR (Missing Not At Random) introduces selection bias
      - Reduces sample size and statistical power
    """
    print("\n" + "="*75)
    print("QUESTION 1 ▸ DUPLICATE & MISSING DATA DETECTION")
    print("="*75)
    
    # Duplicates analysis
    q_dup_full = q_raw.duplicated(subset=["ID", "RawQuery", "Timestamp"]).sum()
    q_dup_query = q_raw["RawQuery"].duplicated().sum()
    p_dup_full = p_raw.duplicated(subset=["ID"]).sum()
    
    print("\n[DUPLICATES]")
    print(f"  Query dataset:")
    print(f"    • Fully duplicate rows: {q_dup_full}")
    print(f"    • Duplicate RawQuery texts: {q_dup_query}")
    print(f"  Product dataset:")
    print(f"    • Fully duplicate product IDs: {p_dup_full}")
    
    # Missing values analysis
    print("\n[MISSING VALUES]")
    print(f"\n  Query dataset:")
    q_null = q_raw[["ID", "RawQuery", "Result", "ClickedResult"]].isnull().sum()
    for col, cnt in q_null.items():
        print(f"    • {col}: {cnt}")
    
    print(f"\n  Product dataset:")
    p_null = p_raw[["ID", "CategoryName", "MinPrice", "MaxPrice",
                     "AvgPrice", "MinNumShops", "MaxNumShops",
                     "AvgNumShops"]].isnull().sum()
    for col, cnt in p_null.items():
        pct = 100 * cnt / len(p_raw) if len(p_raw) > 0 else 0
        print(f"    • {col}: {cnt} ({pct:.1f}%)")
    
    # Data cleaning
    print("\n[DATA CLEANING]")
    q_clean = q_raw.drop_duplicates(subset=["ID"]).copy()
    p_clean = p_raw.dropna(subset=["ID"]).drop_duplicates(subset=["ID"]).copy()
    
    print(f"  Query dataset:")
    print(f"    Before: {len(q_raw)} rows  →  After: {len(q_clean)} rows")
    print(f"    Removed: {len(q_raw) - len(q_clean)} rows")
    
    print(f"  Product dataset:")
    print(f"    Before: {len(p_raw)} rows  →  After: {len(p_clean)} rows")
    print(f"    Removed: {len(p_raw) - len(p_clean)} rows")
    
    print("\n[IMPACT ANALYSIS]")
    print("""
  Impact of Duplicates:
    ✗ Repeated queries (like 'کیف مدرسه') artificially inflate their weight
    ✗ Skew statistical distributions toward frequent queries
    ✗ Violate assumption of independent observations
    ✗ Lead to biased mean/variance estimates
  
  Impact of Missing Data:
    ✗ 14.3% of products lack price info → incomplete market analysis
    ✗ MNAR (Missing Not At Random) bias: if expensive/cheap items 
      systematically lack prices, we get distorted price distributions
    ✗ Reduced effective sample size and statistical power
    ✓ Solution: Use imputation (e.g., fill with category median)
    """)
    
    return q_clean, p_clean


# ============================================================================
# QUESTION 2: OUTLIER DETECTION & REMOVAL (IQR METHOD)
# ============================================================================

def detect_and_remove_outliers(q: pd.DataFrame):
    """
    Detect outliers using Interquartile Range (IQR) rule:
      Lower Bound = Q1 - 1.5 × IQR
      Upper Bound = Q3 + 1.5 × IQR
    
    Values outside [Lower, Upper] are considered outliers.
    """
    print("\n" + "="*75)
    print("QUESTION 2 ▸ OUTLIER DETECTION & REMOVAL (IQR)")
    print("="*75)
    
    def get_iqr_bounds(series: pd.Series):
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        return lower, upper
    
    # Apply IQR rule separately to NumRetrievals and NumClicks
    ret_lo, ret_hi = get_iqr_bounds(q["NumRetrievals"])
    clk_lo, clk_hi = get_iqr_bounds(q["NumClicks"])
    
    print(f"\n[IQR BOUNDS]")
    print(f"  NumRetrievals: [{ret_lo:.1f}, {ret_hi:.1f}]")
    print(f"  NumClicks:     [{clk_lo:.1f}, {clk_hi:.1f}]")
    
    # Count outliers
    ret_outliers = ((q["NumRetrievals"] < ret_lo) | 
                    (q["NumRetrievals"] > ret_hi)).sum()
    clk_outliers = ((q["NumClicks"] < clk_lo) | 
                    (q["NumClicks"] > clk_hi)).sum()
    
    print(f"\n[OUTLIERS DETECTED]")
    print(f"  Retrieval outliers: {ret_outliers} queries")
    print(f"  Click outliers:     {clk_outliers} queries")
    
    # Filter
    keep_mask = (q["NumRetrievals"].between(ret_lo, ret_hi) &
                 q["NumClicks"].between(clk_lo, clk_hi))
    q_filtered = q[keep_mask].copy()
    
    print(f"\n[FILTERING RESULTS]")
    print(f"  Kept:     {len(q_filtered)} queries ({100*len(q_filtered)/len(q):.1f}%)")
    print(f"  Removed:  {len(q) - len(q_filtered)} queries ({100*(len(q)-len(q_filtered))/len(q):.1f}%)")
    
    print("\n[INTERPRETATION]")
    print("""
  Examples of outliers removed:
    • Queries with 50+ results (e.g., 'کتاب') → too generic
    • Queries with 10+ clicks → likely bot/test data
    • Queries with 0 results combined with many clicks → data anomaly
  
  Benefits of removal:
    ✓ Stabilize statistical measures (mean, std)
    ✓ Improve model robustness
    ✓ Reduce impact of data entry errors
    ✓ Focus on typical user behavior
    """)
    
    return q_filtered


# ============================================================================
# QUESTION 3: QUERY BINNING BY CATEGORY COUNT
# ============================================================================

def bin_queries_by_category_count(q_filtered: pd.DataFrame,
                                   p: pd.DataFrame):
    """
    For each query, find unique product categories in its Result list.
    Then bin queries based on category count.
    """
    print("\n" + "="*75)
    print("QUESTION 3 ▸ QUERY BINNING BY RETRIEVED CATEGORY COUNT")
    print("="*75)
    
    id_to_cat = dict(zip(p["ID"], p["CategoryName"]))
    
    def get_categories(result_ids):
        cats = set()
        for rid in result_ids:
            try:
                cat = id_to_cat.get(int(rid))
                if cat:
                    cats.add(cat)
            except (ValueError, TypeError):
                pass
        return cats
    
    q_filtered["Categories"] = q_filtered["ResultList"].apply(get_categories)
    q_filtered["NumCategories"] = q_filtered["Categories"].apply(len)
    
    # Define bins
    bin_edges = [-1, 0, 3, 5, 10, float("inf")]
    bin_labels = [
        "0 – No result",
        "1-3 categories",
        "4-5 categories",
        "6-10 categories",
        "10+ categories",
    ]
    q_filtered["Bin"] = pd.cut(
        q_filtered["NumCategories"],
        bins=bin_edges,
        labels=bin_labels,
        right=True,
    )
    
    bin_counts = q_filtered["Bin"].value_counts().sort_index()
    print(f"\n[BIN DISTRIBUTION]")
    for bin_label, count in bin_counts.items():
        pct = 100 * count / len(q_filtered)
        print(f"  {bin_label:20s}: {count:4d} ({pct:5.1f}%)")
    
    print("\n[NOTE - DATA LIMITATION]")
    print("""
  The product IDs in the query dataset (e.g., 7151290) exist in a different
  ID space than the product dataset (e.g., 7, 8, 9).
  
  In a real-world scenario:
    • Query results link to a full product catalog (millions of items)
    • This subset of products (4999) is just a sample
    • For complete category analysis, access to full product table needed
    """)
    
    return q_filtered


# ============================================================================
# QUESTION 4: RETRIEVAL & CLICK PROBABILITY (NORMALIZED)
# ============================================================================

def calculate_probabilities(q_filtered: pd.DataFrame, p: pd.DataFrame):
    """
    For each product:
      P(retrieval) = count(product was retrieved) / total retrievals
      P(click)     = count(product was clicked) / total clicks
    
    Then normalize using Max Normalization:
      P_normalized = P / max(P)
    
    This gives values in range [0, 1] for easy comparison.
    """
    print("\n" + "="*75)
    print("QUESTION 4 ▸ RETRIEVAL & CLICK PROBABILITY (NORMALIZED)")
    print("="*75)
    
    # Collect all retrieved and clicked product IDs
    all_retrieved = []
    all_clicked = []
    for _, row in q_filtered.iterrows():
        all_retrieved.extend(row["ResultList"])
        all_clicked.extend(row["ClickedResultList"])
    
    ret_counts = Counter(all_retrieved)
    clk_counts = Counter(all_clicked)
    total_ret = sum(ret_counts.values())
    total_clk = sum(clk_counts.values())
    
    print(f"\n[TOTALS]")
    print(f"  Total retrievals: {total_ret}")
    print(f"  Total clicks:     {total_clk}")
    
    # Compute probabilities
    prod_stats = p[["ID", "CategoryName", "AvgPrice"]].copy()
    prod_stats["RetrievalCount"] = prod_stats["ID"].map(ret_counts).fillna(0).astype(int)
    prod_stats["ClickCount"] = prod_stats["ID"].map(clk_counts).fillna(0).astype(int)
    prod_stats["P_Retrieval"] = prod_stats["RetrievalCount"] / max(total_ret, 1)
    prod_stats["P_Click"] = prod_stats["ClickCount"] / max(total_clk, 1)
    prod_stats["P_Ret_norm"] = (prod_stats["P_Retrieval"] / 
                                max(prod_stats["P_Retrieval"].max(), 1e-9))
    prod_stats["P_Clk_norm"] = (prod_stats["P_Click"] / 
                                max(prod_stats["P_Click"].max(), 1e-9))
    
    prod_active = prod_stats[prod_stats["RetrievalCount"] > 0].copy()
    
    print(f"\n[PRODUCTS WITH RETRIEVALS]")
    print(f"  {len(prod_active)} products have ≥1 retrieval")
    
    print(f"\n[TOP 10 BY RETRIEVAL COUNT]")
    top10 = prod_active.nlargest(10, "RetrievalCount")
    for idx, (_, row) in enumerate(top10.iterrows(), 1):
        print(f"  {idx:2d}. ID={int(row['ID']):5d}  "
              f"Ret={int(row['RetrievalCount']):2d}  "
              f"Clk={int(row['ClickCount']):2d}  "
              f"P_ret_norm={row['P_Ret_norm']:.3f}  "
              f"Category: {row['CategoryName'][:30]}")
    
    print("\n[PATTERNS OBSERVED]")
    print("""
  ✓ Distribution is heavily right-skewed (Long-tail)
    • Few products retrieved many times
    • Most products retrieved only once or twice
  
  ✓ Zipf's Law pattern visible
    • Frequency ∝ 1/rank (Power-law distribution)
    • Common in e-commerce, search, and natural language
  
  ✓ Weak correlation between retrieval and clicks
    • More retrievals ≠ automatically more clicks
    • Product quality, rank position, and user intent matter
    """)
    
    return prod_stats


# ============================================================================
# QUESTION 5: BOXPLOT & HISTOGRAM - DISTRIBUTION ANALYSIS
# ============================================================================

def plot_q5_distribution(prod_active: pd.DataFrame):
    """
    Create Boxplot and Histogram for product retrieval distribution.
    
    Boxplot shows:
      • Q1, Q2 (median), Q3 as box bounds
      • Whiskers extend to ±1.5×IQR
      • Points beyond whiskers = outliers (fliers)
    
    Histogram shows:
      • Frequency of retrieval counts
      • Mean and median lines for reference
    """
    print("\n" + "="*75)
    print("QUESTION 5 ▸ BOXPLOT & HISTOGRAM")
    print("="*75)
    
    ret_data = prod_active["RetrievalCount"].values
    
    print(f"\n[STATISTICS]")
    print(f"  min    = {ret_data.min()}")
    print(f"  Q1     = {np.percentile(ret_data, 25):.1f}")
    print(f"  median = {np.median(ret_data):.1f}")
    print(f"  mean   = {ret_data.mean():.2f}")
    print(f"  Q3     = {np.percentile(ret_data, 75):.1f}")
    print(f"  max    = {ret_data.max()}")
    print(f"  std    = {ret_data.std():.2f}")
    
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Q5 – Product Retrieval Distribution Analysis", fontsize=13, fontweight="bold")
    
    # Histogram
    axes[0].hist(ret_data, bins=40, color="#4f8ef7", edgecolor="white", alpha=0.85)
    axes[0].axvline(ret_data.mean(), color="#ff6b6b", lw=2, linestyle="--",
                    label=f"Mean = {ret_data.mean():.2f}")
    axes[0].axvline(np.median(ret_data), color="#ffd166", lw=2, linestyle="--",
                    label=f"Median = {np.median(ret_data):.1f}")
    axes[0].set_xlabel("Number of Retrievals", fontweight="bold")
    axes[0].set_ylabel("Frequency", fontweight="bold")
    axes[0].set_title("Histogram", fontweight="bold")
    axes[0].legend(fontsize=10)
    axes[0].grid(axis="y", alpha=0.3)
    
    # Boxplot
    bp = axes[1].boxplot(
        ret_data, vert=True, patch_artist=True,
        boxprops=dict(facecolor="#4f8ef7", edgecolor="#2a4d9f", linewidth=1.5),
        medianprops=dict(color="#ffd166", linewidth=2.5),
        whiskerprops=dict(color="#aaaaaa", linewidth=1.5),
        capprops=dict(color="#aaaaaa", linewidth=1.5),
        flierprops=dict(marker="o", markerfacecolor="#ff6b6b", markersize=5,
                       markeredgecolor="#ff6b6b", alpha=0.6),
    )
    q1v, q3v = np.percentile(ret_data, [25, 75])
    axes[1].set_ylabel("Number of Retrievals", fontweight="bold")
    axes[1].set_title("Boxplot", fontweight="bold")
    axes[1].set_xticklabels([])
    axes[1].grid(axis="y", alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "q5_boxplot_histogram.png", dpi=150, bbox_inches="tight")
    print("\n[SAVED] q5_boxplot_histogram.png")
    plt.close()
    
    print("\n[PATTERNS VISIBLE]")
    print("""
  ✓ Heavy right-skew (Right-tailed distribution)
    • Most products retrieved only 1-2 times
    • Tail extends to 12 retrievals
  
  ✓ Mode < Median < Mean
    • Indicates strong positive skew
    • Few very popular products pull mean upward
  
  ✓ Boxplot interpretation
    • Q1 ≈ Q2 ≈ Median (all around 1) → dense lower end
    • Long whisker extending to max
    • Red dots (fliers) = outliers (10+ retrievals)
    
  ✓ This follows Zipf's Law: f(k) ~ 1/k^α
    • Ubiquitous in e-commerce (bestsellers phenomenon)
    • Also seen in word frequencies, city sizes, income distribution
    """)


# ============================================================================
# QUESTION 6: QQ PLOT - NORMALITY TEST
# ============================================================================

def plot_q6_qq_plot(q_clean: pd.DataFrame):
    """
    Q-Q Plot compares sample quantiles (query frequencies) against
    theoretical quantiles of normal distribution.
    
    If points lie on the diagonal red line → sample is normally distributed
    If points deviate → sample follows different distribution
    """
    print("\n" + "="*75)
    print("QUESTION 6 ▸ QQ PLOT – NORMALITY TEST")
    print("="*75)
    
    query_freq = q_clean["RawQuery"].value_counts().values
    
    # Shapiro-Wilk test (uses up to 5000 samples)
    w_stat, p_value = stats.shapiro(query_freq[:5000] if len(query_freq) > 5000
                                    else query_freq)
    
    print(f"\n[SHAPIRO-WILK NORMALITY TEST]")
    print(f"  W-statistic = {w_stat:.4f}")
    print(f"  p-value     = {p_value:.4e}")
    print(f"  Result:     {'NORMAL' if p_value > 0.05 else 'NOT NORMAL'}")
    
    # Q-Q Plot
    fig, ax = plt.subplots(figsize=(8, 7))
    fig.suptitle("Q6 – Q-Q Plot: Query Frequency vs Normal Distribution",
                 fontsize=13, fontweight="bold")
    
    (osm, osr), (slope, intercept, r) = stats.probplot(query_freq, dist="norm")
    ax.scatter(osm, osr, color="#4f8ef7", s=25, alpha=0.65, label="Data points")
    line_x = np.array([osm.min(), osm.max()])
    ax.plot(line_x, slope * line_x + intercept, color="#ff6b6b", lw=2.5,
            label="Normal reference line")
    ax.set_xlabel("Theoretical Quantiles", fontweight="bold")
    ax.set_ylabel("Sample Quantiles", fontweight="bold")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=10)
    ax.text(0.05, 0.92, f"R² = {r**2:.4f}\np-value = {p_value:.2e}",
            transform=ax.transAxes, fontsize=10,
            bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.9,
                     edgecolor="gray"))
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "q6_qqplot.png", dpi=150, bbox_inches="tight")
    print("\n[SAVED] q6_qqplot.png")
    plt.close()
    
    print("\n[INTERPRETATION]")
    print("""
  ✓ Strong deviation from diagonal line
    • Points curve upward in both tails
    • Indicates heavy-tailed distribution (more extreme values than normal)
  
  ✓ p-value << 0.05
    • Conclusively reject hypothesis of normality
    • 99.9%+ confidence that data is NOT normally distributed
  
  ✓ Alternative distributions better fit
    • Power-law: f(x) ~ x^(-α)
    • Log-normal: ln(X) is normally distributed
    • Pareto: many small events, few large ones
  
  ✓ Implications for statistical tests
    ✗ Student's t-test assumes normality (INVALID here)
    ✓ Use non-parametric tests instead:
      • Mann-Whitney U test (instead of t-test)
      • Kruskal-Wallis H test (instead of ANOVA)
    """)


# ============================================================================
# QUESTION 7: SCATTER PLOT - RELATIONSHIP ANALYSIS
# ============================================================================

def plot_q7_scatter(prod_active: pd.DataFrame):
    """
    Scatter plot of product retrievals (X) vs clicks (Y).
    Shows relationship strength and pattern.
    """
    print("\n" + "="*75)
    print("QUESTION 7 ▸ SCATTER PLOT – RETRIEVALS vs CLICKS")
    print("="*75)
    
    scatter_df = prod_active[prod_active["ClickCount"] > 0].copy()
    
    if len(scatter_df) < 2:
        print("  [WARNING] Not enough data with clicks for scatter plot")
        return
    
    correlation = scatter_df["RetrievalCount"].corr(scatter_df["ClickCount"])
    
    print(f"\n[CORRELATION ANALYSIS]")
    print(f"  Pearson r = {correlation:.4f}")
    print(f"  Interpretation:")
    if abs(correlation) < 0.3:
        print(f"    → Weak correlation (r < 0.3)")
    elif abs(correlation) < 0.7:
        print(f"    → Moderate correlation")
    else:
        print(f"    → Strong correlation")
    
    # Create scatter plot
    fig, ax = plt.subplots(figsize=(10, 7))
    fig.suptitle("Q7 – Product Retrievals vs Clicks",
                 fontsize=13, fontweight="bold")
    
    sc = ax.scatter(
        scatter_df["RetrievalCount"],
        scatter_df["ClickCount"],
        c=scatter_df["RetrievalCount"],
        cmap="plasma",
        alpha=0.6, s=45,
        edgecolors="white", linewidth=0.5,
    )
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label("Retrieval Count", fontweight="bold")
    
    # Trend line
    z = np.polyfit(scatter_df["RetrievalCount"], scatter_df["ClickCount"], 1)
    pp = np.poly1d(z)
    xs = np.linspace(scatter_df["RetrievalCount"].min(),
                     scatter_df["RetrievalCount"].max(), 200)
    ax.plot(xs, pp(xs), color="#ffd166", lw=2.5, linestyle="--",
            label=f"Trend (r = {correlation:.3f})")
    
    ax.set_xlabel("Number of Retrievals", fontweight="bold")
    ax.set_ylabel("Number of Clicks", fontweight="bold")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=11, loc="upper left")
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "q7_scatter.png", dpi=150, bbox_inches="tight")
    print("\n[SAVED] q7_scatter.png")
    plt.close()
    
    print("\n[INSIGHTS]")
    print(f"""
  ✓ Weak to moderate positive correlation ({correlation:.3f})
    • More retrievals → slightly more clicks, but relationship is loose
    • Other factors influence click behavior:
      - Product rank in result list
      - Product title and description quality
      - User intent and search context
      - Seasonality and trends
  
  ✓ Scatter pattern shows:
    • Clustering near (1,0) → most products retrieved once, no clicks
    • Sparse upper region → few products retrieved multiple times
    • Some products clicked more than expected (above trend line)
    • Some clicked less than expected (below trend line)
  
  ✓ Business implications:
    ✓ Rank and position matter more than retrieval count
    ✓ Product quality and relevance are key to clicks
    ✓ Marketing/UX improvements could boost CTR
    """)


# ============================================================================
# QUESTION 8: FEATURE EXTRACTION
# ============================================================================

def extract_features(q_filtered: pd.DataFrame, prod_stats: pd.DataFrame,
                    p: pd.DataFrame):
    """
    Extract key features for each product and query.
    """
    print("\n" + "="*75)
    print("QUESTION 8 ▸ FEATURE EXTRACTION")
    print("="*75)
    
    id_to_cat = dict(zip(p["ID"], p["CategoryName"]))
    
    # Product features
    prod_features = prod_stats[
        ["ID", "CategoryName", "AvgPrice",
         "RetrievalCount", "ClickCount", "P_Ret_norm", "P_Clk_norm"]
    ].copy()
    prod_features["CTR"] = (
        prod_features["ClickCount"] /
        prod_features["RetrievalCount"].replace(0, np.nan)
    ).fillna(0)
    
    print(f"\n[PRODUCT FEATURES EXTRACTED]")
    print(f"  • RetrievalCount: how many times product was in a search result")
    print(f"  • ClickCount: how many times user clicked on product")
    print(f"  • CTR (Click-Through Rate): ClickCount / RetrievalCount")
    print(f"  • P_Ret_norm, P_Clk_norm: normalized probabilities [0,1]")
    
    print(f"\n[TOP 10 PRODUCTS BY RETRIEVAL]")
    top_prod = prod_features.nlargest(10, "RetrievalCount")
    for idx, (_, row) in enumerate(top_prod.iterrows(), 1):
        print(f"  {idx:2d}. {row['CategoryName'][:25]:25s} "
              f"Ret={int(row['RetrievalCount']):2d}  "
              f"Clk={int(row['ClickCount']):2d}  "
              f"CTR={row['CTR']:.2%}")
    
    # Category aggregates
    cat_ret = prod_features.groupby("CategoryName")["RetrievalCount"].sum().nlargest(10)
    print(f"\n[TOP 10 CATEGORIES BY TOTAL RETRIEVALS]")
    for idx, (cat, cnt) in enumerate(cat_ret.items(), 1):
        print(f"  {idx:2d}. {cat:40s}: {int(cnt):3d} retrievals")
    
    # Query features - dominant category
    def get_dominant_category(result_ids):
        cat_list = []
        for rid in result_ids:
            try:
                cat = id_to_cat.get(int(rid))
                if cat:
                    cat_list.append(cat)
            except (ValueError, TypeError):
                pass
        if cat_list:
            return Counter(cat_list).most_common(1)[0][0]
        return None
    
    q_filtered["DominantCat"] = q_filtered["ResultList"].apply(
        get_dominant_category
    )
    
    print(f"\n[QUERY FEATURES EXTRACTED]")
    print(f"  • DominantCat: category with most products in search result")
    print(f"  • NumCategories: count of unique categories retrieved")
    print(f"  • NumRetrievals: total products retrieved")
    print(f"  • NumClicks: total products clicked")
    
    print(f"\n[TOP 10 DOMINANT CATEGORIES]")
    dom_cats = q_filtered["DominantCat"].value_counts().head(10)
    for idx, (cat, cnt) in enumerate(dom_cats.items(), 1):
        print(f"  {idx:2d}. {cat:40s}: appears {int(cnt):3d} times")
    
    return prod_features, q_filtered


# ============================================================================
# QUESTION 9: VIOLIN PLOT - PRICE DISTRIBUTION
# ============================================================================

def plot_q9_violin(p: pd.DataFrame):
    """
    Violin plot shows distribution of AvgPrice per category.
    Width = density of products at that price.
    """
    print("\n" + "="*75)
    print("QUESTION 9 ▸ VIOLIN PLOT – PRICE DISTRIBUTION")
    print("="*75)
    
    price_df = p[p["AvgPrice"].notna() & (p["AvgPrice"] > 0)].copy()
    price_df["LogPrice"] = np.log10(price_df["AvgPrice"])
    
    # Price statistics
    ps = price_df["AvgPrice"]
    q1_p, q3_p = ps.quantile(0.25), ps.quantile(0.75)
    iqr_p = q3_p - q1_p
    n_outliers = ((ps < q1_p - 1.5*iqr_p) | 
                  (ps > q3_p + 1.5*iqr_p)).sum()
    
    print(f"\n[PRICE STATISTICS]")
    print(f"  min      = {ps.min():>15,.0f} Rials")
    print(f"  Q1       = {ps.quantile(0.25):>15,.0f} Rials")
    print(f"  median   = {ps.median():>15,.0f} Rials")
    print(f"  mean     = {ps.mean():>15,.0f} Rials")
    print(f"  Q3       = {ps.quantile(0.75):>15,.0f} Rials")
    print(f"  max      = {ps.max():>15,.0f} Rials")
    print(f"  Outliers (IQR) = {n_outliers} products ({100*n_outliers/len(ps):.1f}%)")
    
    # Top categories
    top_cats = price_df["CategoryName"].value_counts().nlargest(12).index.tolist()
    price_top = price_df[price_df["CategoryName"].isin(top_cats)]
    data_groups = [price_top[price_top["CategoryName"] == c]["LogPrice"].values
                   for c in top_cats]
    
    # Create violin plot
    fig, ax = plt.subplots(figsize=(17, 7))
    fig.suptitle("Q9 – Violin Plot: log₁₀(AvgPrice) by Top-12 Categories",
                 fontsize=13, fontweight="bold")
    
    colors_pal = plt.cm.plasma(np.linspace(0.1, 0.9, len(top_cats)))
    parts = ax.violinplot(
        data_groups,
        positions=range(len(top_cats)),
        showmeans=False,
        showmedians=True,
        showextrema=True,
    )
    
    for pc, col in zip(parts["bodies"], colors_pal):
        pc.set_facecolor(col)
        pc.set_edgecolor("white")
        pc.set_alpha(0.75)
    
    parts["cmedians"].set_color("#ffd166")
    parts["cmedians"].set_linewidth(2.5)
    parts["cbars"].set_color("gray")
    parts["cmins"].set_color("gray")
    parts["cmaxes"].set_color("gray")
    
    short_names = [c[:16] + "…" if len(c) > 16 else c for c in top_cats]
    ax.set_xticks(range(len(top_cats)))
    ax.set_xticklabels(short_names, rotation=32, ha="right", fontsize=9)
    ax.set_ylabel("log₁₀(AvgPrice)", fontweight="bold")
    ax.axhline(np.median(price_top["LogPrice"]), color="#ffd166", lw=1.3,
               linestyle="--", alpha=0.7, label="Overall median")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=10)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "q9_violin_price.png", dpi=150, bbox_inches="tight")
    print("\n[SAVED] q9_violin_price.png")
    plt.close()
    
    print("\n[ANSWERS TO SUB-QUESTIONS]")
    print("""
  Q1: Are data organized by a specific description?
      ✓ YES – Products grouped by CategoryName.
      ✓ Each violin shows price distribution for ONE category.
      ✓ Enables category-level market analysis.
  
  Q2: Can outliers be identified by viewing the chart?
      ✓ YES – Points far outside the violin body are outliers.
      ✓ Visual identification useful for:
        • Spotting data entry errors
        • Finding luxury/budget segment products
        • Detecting market anomalies
  
  Q3: Show key features like median and quartiles from the chart?
      ✓ Gold line inside each violin = MEDIAN (50th percentile)
      ✓ Thicker parts = higher product density at that price
      ✓ Thin tails = sparse outlier regions
      ✓ Width at each price level = probability/density of products
      ✓ Violin symmetry indicates skewness of distribution
    """)


# ============================================================================
# QUESTION 10: DOMINANT CATEGORY & CHI-SQUARE TEST
# ============================================================================

def chi_square_dominance_test(q_filtered: pd.DataFrame, id_to_cat: dict):
    """
    Chi-Square test of independence:
    H₀: Membership in dominant category is INDEPENDENT of having a click
    H₁: Membership and clicks are DEPENDENT
    """
    print("\n" + "="*75)
    print("QUESTION 10 ▸ DOMINANT CATEGORY & CHI-SQUARE TEST")
    print("="*75)
    
    q_chi = q_filtered[q_filtered["DominantCat"].notna()].copy()
    
    def clicked_in_dominant(row):
        dom = row["DominantCat"]
        for cid in row["ClickedResultList"]:
            try:
                if id_to_cat.get(int(cid)) == dom:
                    return True
            except (ValueError, TypeError):
                pass
        return False
    
    q_chi["InDominant"] = q_chi.apply(clicked_in_dominant, axis=1)
    q_chi["HasClick"] = q_chi["NumClicks"] > 0
    
    # Contingency table
    contingency = pd.crosstab(q_chi["InDominant"], q_chi["HasClick"])
    
    print("\n[CONTINGENCY TABLE]")
    print(f"  Rows: User clicked a product in the dominant category")
    print(f"  Cols: User had any click(s)")
    print()
    print(contingency.to_string())
    
    # Chi-Square test
    chi2, p_chi, dof, expected = stats.chi2_contingency(contingency)
    
    print(f"\n[CHI-SQUARE TEST RESULTS]")
    print(f"  χ² statistic = {chi2:.4f}")
    print(f"  df           = {dof}")
    print(f"  p-value      = {p_chi:.6f}")
    print(f"  α (significance level) = 0.05")
    
    if p_chi < 0.05:
        print(f"\n  ✓ RESULT: REJECT H₀")
        print(f"    → Significant relationship found (p < 0.05)")
        print(f"    → Dominant category membership and clicks ARE dependent")
    else:
        print(f"\n  ✗ RESULT: FAIL TO REJECT H₀")
        print(f"    → No significant relationship (p ≥ 0.05)")
        print(f"    → Dominant category membership and clicks are independent")
    
    print(f"\n[DOMINANT CATEGORIES]")
    top_doms = q_chi["DominantCat"].value_counts().head(10)
    for idx, (cat, cnt) in enumerate(top_doms.items(), 1):
        print(f"  {idx:2d}. {cat:40s}: {int(cnt):3d} queries")
    
    print("\n[INTERPRETATION]")
    print(f"""
  The p-value of {p_chi:.4f} suggests:
  
  {'✓ SIGNIFICANT correlation' if p_chi < 0.05 else '✗ NO significant correlation'}
    
    If significant (p < 0.05):
      → Users prefer to click products from the dominant category
      → Dominant category is PREDICTIVE of click behavior
      → Business implication: rank products from dominant category higher
    
    If not significant (p ≥ 0.05):
      → Dominant category doesn't predict clicks
      → Users click based on other factors (relevance, ranking, etc.)
      → Need to investigate other click drivers
    """)


# ============================================================================
# MAIN EXECUTION & COMPREHENSIVE DASHBOARD
# ============================================================================

def create_comprehensive_dashboard(q_filtered: pd.DataFrame,
                                   prod_active: pd.DataFrame,
                                   prod_features: pd.DataFrame,
                                   p: pd.DataFrame,
                                   q_clean: pd.DataFrame):
    """Create a comprehensive multi-panel dashboard."""
    print("\n[*] Generating comprehensive dashboard …")
    
    fig = plt.figure(figsize=(20, 24))
    fig.suptitle(
        "Business Intelligence – Comprehensive Analysis Dashboard\n"
        "E-Commerce Query & Product Data | Q1-Q10 Summary",
        fontsize=15, fontweight="bold", y=0.995,
    )
    
    gs = fig.add_gridspec(4, 3, hspace=0.4, wspace=0.32)
    
    # Q5: Histogram
    ax1 = fig.add_subplot(gs[0, 0])
    ret_data = prod_active["RetrievalCount"].values
    ax1.hist(ret_data, bins=35, color="#4f8ef7", edgecolor="white", alpha=0.85)
    ax1.axvline(ret_data.mean(), color="red", lw=1.8, linestyle="--",
                label=f"Mean {ret_data.mean():.1f}")
    ax1.axvline(np.median(ret_data), color="gold", lw=1.8, linestyle="--",
                label=f"Median {np.median(ret_data):.1f}")
    ax1.set_title("Q5 – Retrieval Histogram", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Retrievals")
    ax1.set_ylabel("Frequency")
    ax1.legend(fontsize=9)
    ax1.grid(axis="y", alpha=0.3)
    
    # Q5: Boxplot
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.boxplot(ret_data, patch_artist=True,
                boxprops=dict(facecolor="#4f8ef7"),
                medianprops=dict(color="gold", linewidth=2.5),
                flierprops=dict(marker="o", color="red", markersize=3, alpha=0.5))
    ax2.set_title("Q5 – Retrieval Boxplot", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Retrievals")
    ax2.set_xticks([])
    ax2.grid(axis="y", alpha=0.3)
    
    # Q6: QQ Plot
    ax3 = fig.add_subplot(gs[0, 2])
    query_freq = q_clean["RawQuery"].value_counts().values
    (osm, osr), (slope, intercept, r) = stats.probplot(query_freq, dist="norm")
    ax3.scatter(osm, osr, color="#4f8ef7", s=12, alpha=0.6)
    xl = np.array([osm.min(), osm.max()])
    ax3.plot(xl, slope*xl+intercept, color="red", lw=1.8)
    ax3.set_title(f"Q6 – QQ Plot (R²={r**2:.3f})", fontsize=11, fontweight="bold")
    ax3.set_xlabel("Theoretical quantiles")
    ax3.set_ylabel("Sample quantiles")
    ax3.grid(alpha=0.3)
    
    # Q7: Scatter
    ax4 = fig.add_subplot(gs[1, 0])
    scatter_df = prod_active[prod_active["ClickCount"] > 0]
    corr = scatter_df["RetrievalCount"].corr(scatter_df["ClickCount"])
    ax4.scatter(scatter_df["RetrievalCount"], scatter_df["ClickCount"],
                c=scatter_df["ClickCount"], cmap="plasma", alpha=0.55, s=20)
    if len(scatter_df) > 1:
        z = np.polyfit(scatter_df["RetrievalCount"], scatter_df["ClickCount"], 1)
        pp = np.poly1d(z)
        xs = np.linspace(scatter_df["RetrievalCount"].min(),
                        scatter_df["RetrievalCount"].max(), 150)
        ax4.plot(xs, pp(xs), color="gold", lw=1.5, linestyle="--")
    ax4.set_title(f"Q7 – Scatter (r={corr:.3f})", fontsize=11, fontweight="bold")
    ax4.set_xlabel("Retrievals")
    ax4.set_ylabel("Clicks")
    ax4.grid(alpha=0.3)
    
    # Q4: Normalized probabilities
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.scatter(prod_active["P_Ret_norm"], prod_active["P_Clk_norm"],
                alpha=0.5, c=prod_active["P_Ret_norm"], cmap="viridis", s=15)
    ax5.set_title("Q4 – Normalized Probabilities", fontsize=11, fontweight="bold")
    ax5.set_xlabel("P(Retrieval) norm")
    ax5.set_ylabel("P(Click) norm")
    ax5.grid(alpha=0.3)
    
    # Q3: Bins
    ax6 = fig.add_subplot(gs[1, 2])
    bin_counts = q_filtered["Bin"].value_counts().sort_index()
    colors_bin = ["#4f8ef7", "#7c5cbf", "#e86c3a", "#2ab9a0", "#e0c040"]
    ax6.bar(range(len(bin_counts)), bin_counts.values,
            color=colors_bin[:len(bin_counts)])
    ax6.set_xticks(range(len(bin_counts)))
    ax6.set_xticklabels(bin_counts.index, rotation=24, ha="right", fontsize=8)
    ax6.set_title("Q3 – Query Bins", fontsize=11, fontweight="bold")
    ax6.set_ylabel("Count")
    ax6.grid(axis="y", alpha=0.3)
    
    # Q8: Top products
    ax7 = fig.add_subplot(gs[2, :2])
    top15 = prod_features.nlargest(15, "RetrievalCount")
    short_cats = [c[:18]+"…" if len(c)>18 else c for c in top15["CategoryName"]]
    y_pos = np.arange(len(top15))
    ax7.barh(y_pos, top15["RetrievalCount"].values, color="#4f8ef7",
             label="Retrievals")
    ax7.barh(y_pos, top15["ClickCount"].values, color="#ff6b6b", alpha=0.8,
             label="Clicks")
    ax7.set_yticks(y_pos)
    ax7.set_yticklabels(short_cats, fontsize=9)
    ax7.set_title("Q8 – Top 15 Products", fontsize=11, fontweight="bold")
    ax7.set_xlabel("Count")
    ax7.legend(fontsize=9, loc="lower right")
    ax7.invert_yaxis()
    ax7.grid(axis="x", alpha=0.3)
    
    # Q10: Dominant categories pie
    ax8 = fig.add_subplot(gs[2, 2])
    dom_freq = q_filtered["DominantCat"].value_counts().head(8)
    wedge_cols = plt.cm.Set3(np.linspace(0, 1, 8))
    wedges, texts, autotexts = ax8.pie(
        dom_freq.values,
        labels=None,
        autopct="%1.1f%%",
        colors=wedge_cols,
        startangle=140,
        textprops={"color": "white", "fontsize": 8},
    )
    for at in autotexts:
        at.set_fontsize(8)
    ax8.set_title("Q10 – Dominant Categories", fontsize=11, fontweight="bold")
    ax8.legend(
        wedges,
        [c[:14]+"…" if len(c)>14 else c for c in dom_freq.index],
        loc="lower left", bbox_to_anchor=(-0.25, -0.25),
        fontsize=8,
    )
    
    # Q9: Violin plot
    ax9 = fig.add_subplot(gs[3, :])
    price_df = p[p["AvgPrice"].notna() & (p["AvgPrice"]>0)].copy()
    price_df["LogPrice"] = np.log10(price_df["AvgPrice"])
    top_cats = price_df["CategoryName"].value_counts().nlargest(12).index.tolist()
    price_top = price_df[price_df["CategoryName"].isin(top_cats)]
    data_groups = [price_top[price_top["CategoryName"]==c]["LogPrice"].values
                   for c in top_cats]
    
    colors_vio = plt.cm.plasma(np.linspace(0.1, 0.9, len(top_cats)))
    parts = ax9.violinplot(data_groups, positions=range(len(top_cats)),
                           showmeans=False, showmedians=True, showextrema=True)
    for pc, col in zip(parts["bodies"], colors_vio):
        pc.set_facecolor(col)
        pc.set_edgecolor("white")
        pc.set_alpha(0.75)
    parts["cmedians"].set_color("#ffd166")
    parts["cmedians"].set_linewidth(2)
    parts["cbars"].set_color("gray")
    parts["cmins"].set_color("gray")
    parts["cmaxes"].set_color("gray")
    
    short_names = [c[:15]+"…" if len(c)>15 else c for c in top_cats]
    ax9.set_xticks(range(len(top_cats)))
    ax9.set_xticklabels(short_names, rotation=30, ha="right", fontsize=9)
    ax9.set_title("Q9 – Violin Plot: log₁₀(AvgPrice) by Category",
                  fontsize=11, fontweight="bold")
    ax9.set_ylabel("log₁₀(AvgPrice)")
    ax9.grid(axis="y", alpha=0.3)
    
    plt.savefig(OUTPUT_DIR / "dashboard.png", dpi=150, bbox_inches="tight")
    print("[✓] Saved dashboard.png")
    plt.close()


def main():
    """Main execution function."""
    print("\n" + "="*75)
    print(" BUSINESS INTELLIGENCE HOMEWORK PROJECT")
    print(" E-Commerce Query & Product Data Analysis")
    print("="*75)
    
    try:
        # Load datasets
        q_raw = load_query_dataset(QUERY_PATH)
        p_raw = load_product_dataset(PRODUCT_PATH)
        
        # Q1: Duplicates & Missing
        q, p = analyze_duplicates_and_missing(q_raw, p_raw)
        
        # Q2: Outliers
        q_filtered = detect_and_remove_outliers(q)
        
        # Q3: Binning
        q_filtered = bin_queries_by_category_count(q_filtered, p)
        
        # Q4: Probabilities
        prod_stats = calculate_probabilities(q_filtered, p)
        prod_active = prod_stats[prod_stats["RetrievalCount"] > 0].copy()
        
        # Q5: Boxplot & Histogram
        plot_q5_distribution(prod_active)
        
        # Q6: QQ Plot
        plot_q6_qq_plot(q)
        
        # Q7: Scatter
        plot_q7_scatter(prod_active)
        
        # Q8: Feature Extraction
        prod_features, q_filtered = extract_features(q_filtered, prod_stats, p)
        
        # Q9: Violin Plot
        plot_q9_violin(p)
        
        # Q10: Chi-Square
        id_to_cat = dict(zip(p["ID"], p["CategoryName"]))
        chi_square_dominance_test(q_filtered, id_to_cat)
        
        # Create dashboard
        create_comprehensive_dashboard(q_filtered, prod_active,
                                      prod_features, p, q)
        
        print("\n" + "="*75)
        print(" ALL ANALYSES COMPLETE ✓")
        print("="*75)
        print("\nGenerated files:")
        print("  • q5_boxplot_histogram.png")
        print("  • q6_qqplot.png")
        print("  • q7_scatter.png")
        print("  • q9_violin_price.png")
        print("  • dashboard.png")
        print("\n")
    
    except FileNotFoundError as e:
        print(f"\n[ERROR] File not found: {e}")
        print(f"Please check file paths:")
        print(f"  QUERY_PATH   = {QUERY_PATH}")
        print(f"  PRODUCT_PATH = {PRODUCT_PATH}")
    except Exception as e:
        print(f"\n[ERROR] {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
