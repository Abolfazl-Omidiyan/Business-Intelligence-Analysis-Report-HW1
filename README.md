# Business Intelligence Homework Project
## E-Commerce Query & Product Data Analysis

**Author:** Abolfazl Omidiyan  
**Degree:** Master's Student in Information Technology - Enterprise Architecture  
**Student Number:** 404443019

---

## 📁 Project Files

### 1. **complete_bi_analysis.py** (Main Python Script)
The comprehensive analysis script that covers all 10 questions.

**Features:**
- Custom CSV parsing for both datasets (handles special encoding)
- All 10 questions fully implemented
- Detailed console output with section headers
- Automatic chart generation
- Professional documentation and comments

**Requirements:**
```bash
pip install pandas numpy matplotlib scipy
```

**Usage:**
```bash
python complete_bi_analysis.py
```

**Output:** Generates 4 chart images (q5, q6, q7, q9) + 1 dashboard

---

### 2. **BI_Analysis_Report_English.html** (English Report)
Comprehensive HTML report with:
- ✓ All 10 questions covered in detail
- ✓ Dark theme UI (professional GitHub-inspired design)
- ✓ Interactive table of contents
- ✓ Embedded visualizations
- ✓ Statistical analysis and interpretations
- ✓ Author information at bottom
- ✓ Business insights and recommendations

**How to view:**
- Open in any web browser
- Fully responsive design
- Dark mode by default (easier on eyes)

---

### 3. **Visualization Files**

#### dashboard.png
- Comprehensive multi-panel dashboard
- All questions visualized on one page
- Quick overview of key findings

#### q5_boxplot_histogram.png
- Boxplot: Product retrieval distribution with quartiles
- Histogram: Frequency of retrieval counts
- Shows right-skewed distribution and outliers

#### q6_qqplot.png
- Q-Q plot comparing query frequency to normal distribution
- Shapiro-Wilk test results
- Demonstrates non-normality

#### q7_scatter.png
- Scatter plot: Retrievals (X) vs Clicks (Y)
- Trend line showing weak correlation
- Color-coded by retrieval count

#### q9_violin_price.png
- Violin plots for price distribution across top-12 categories
- Shows density, median (gold line), and outliers
- Log scale for better visibility of wide range

---

## 🎯 What Each Question Covers

| Q | Title | Key Findings |
|---|-------|--------------|
| 1 | Duplicates & Missing Data | 1,765 duplicate queries; 14.3% missing prices |
| 2 | Outlier Detection (IQR) | 833 queries removed (16.7%); bounds [-47, 105] |
| 3 | Query Binning | Categorized by retrieved category diversity |
| 4 | Probabilities (Normalized) | Weak retrieval-click correlation; Zipf's Law pattern |
| 5 | Boxplot & Histogram | Right-skewed distribution; median=1, max=12 |
| 6 | QQ Plot (Normality) | **NOT normal** (p < 0.0001); Heavy-tailed distribution |
| 7 | Scatter Plot | Weak correlation (r=0.33) between retrievals & clicks |
| 8 | Feature Extraction | 127 products with ≥1 retrieval; 8 key features |
| 9 | Violin Plot (Price) | 654 outliers (15.3%); range 1 to 680M Rials |
| 10 | Chi-Square Test | No significant association (p=1.0) with category |

---

## 🔧 How to Run the Python Script

### Step 1: Prepare Environment
```bash
# Install dependencies
pip install pandas numpy matplotlib scipy

# Verify installations
python -c "import pandas, numpy, matplotlib, scipy; print('✓ All packages installed')"
```

### Step 2: Place Data Files
Ensure these CSV files are in the same directory as the script:
- `top5000Query.csv`
- `top5000product.csv`

### Step 3: Run Analysis
```bash
python complete_bi_analysis.py
```

### Step 4: View Results
- Console output shows detailed analysis for each question
- Generated charts appear in current directory:
  - `q5_boxplot_histogram.png`
  - `q6_qqplot.png`
  - `q7_scatter.png`
  - `q9_violin_price.png`
  - `dashboard.png`

---

## 📊 Understanding the Data

### Query Dataset (top5000Query.csv)
- **5,000 rows** of search queries
- Columns: ID, RawQuery, Timestamp, Result, ClickedResult, ClickedRank
- Result/ClickedResult are Python-like list literals
- Must be parsed to extract product IDs

### Product Dataset (top5000product.csv)
- **4,999 rows** of products
- Columns: ID, CategoryName, Titles, MinPrice, MaxPrice, AvgPrice, MinNumShops, MaxNumShops, AvgNumShops
- **Special encoding:** Uses \x1b\x1b as row separator (not standard CSV)
- Requires custom parsing (handled by script)

### Data Limitation
Product IDs in queries (e.g., 7151290) exist in different namespace than product DB (e.g., 7, 8, 9). Only ~127 products match between datasets. In real scenario, full product catalog would be available.

---

## 📈 Key Insights from Analysis

### 🔴 Critical Findings
1. **Non-Normal Distribution:** Query frequency and retrieval counts are heavily right-skewed, NOT normally distributed
   - Implication: Use non-parametric statistical tests
   - Better fits: Power-law, Pareto, or Log-normal distributions

2. **Weak Correlations:** More retrievals don't guarantee more clicks
   - Product rank, title quality, and UX matter more
   - Category membership alone doesn't predict clicks

3. **Outliers Found:** 
   - 833 queries with extreme behavior (16.7%)
   - 654 products with extreme prices (15.3%)

### 🟢 Positive Findings
1. **Data Quality Generally Good:** No full duplicates, most values present
2. **Clear Patterns:** Zipf's Law evident (80/20 rule applies)
3. **Books High-Performing:** 92% CTR for Books category

### 💡 Business Recommendations
1. Focus on improving product relevance ranking, not just visibility
2. Optimize product titles, descriptions, and images (affect CTR)
3. Investigate high-retrieval/low-click products for UX issues
4. Implement featured placement for long-tail products
5. Use non-parametric statistical methods for analysis

---

## 🎓 Learning Outcomes

After completing this project, you will understand:
- ✓ Data cleaning and validation techniques
- ✓ Outlier detection using statistical methods (IQR)
- ✓ Statistical testing for normality (Shapiro-Wilk, QQ plots)
- ✓ Correlation and relationship analysis (scatter plots, chi-square)
- ✓ Distribution visualization (histograms, boxplots, violin plots)
- ✓ Feature engineering for data analysis
- ✓ Non-parametric statistics
- ✓ Data-driven business insights

---

## 📝 Python Script Structure

```
complete_bi_analysis.py
├── Utilities (Data Loading & Parsing)
│   ├── parse_list_column()
│   ├── load_query_dataset()
│   └── load_product_dataset()
│
├── Question 1 (Duplicates & Missing)
│   └── analyze_duplicates_and_missing()
│
├── Question 2 (Outliers)
│   └── detect_and_remove_outliers()
│
├── Question 3 (Binning)
│   └── bin_queries_by_category_count()
│
├── Question 4 (Probabilities)
│   └── calculate_probabilities()
│
├── Question 5 (Boxplot & Histogram)
│   └── plot_q5_distribution()
│
├── Question 6 (QQ Plot)
│   └── plot_q6_qq_plot()
│
├── Question 7 (Scatter)
│   └── plot_q7_scatter()
│
├── Question 8 (Features)
│   └── extract_features()
│
├── Question 9 (Violin Plot)
│   └── plot_q9_violin()
│
├── Question 10 (Chi-Square)
│   └── chi_square_dominance_test()
│
├── Dashboard Creation
│   └── create_comprehensive_dashboard()
│
└── main()
```

---

## 🐛 Troubleshooting

### "FileNotFoundError: top5000Query.csv"
**Solution:** Place CSV files in the same directory as the script

### "No module named 'pandas'"
**Solution:** Install dependencies
```bash
pip install pandas numpy matplotlib scipy --break-system-packages
```

### Charts not saving
**Solution:** Ensure script has write permissions in current directory

### Special characters in output
**Solution:** Already handled by script's encoding configuration

---

## 📚 References & Concepts

### Statistical Concepts Used
- **IQR (Interquartile Range):** Q3 - Q1, for outlier detection
- **Shapiro-Wilk Test:** Tests if data is normally distributed
- **Pearson Correlation:** Measures linear relationship strength
- **Chi-Square Test:** Tests independence of categorical variables
- **Zipf's Law:** Frequency ∝ 1/rank, common in natural phenomena

### Visualization Techniques
- **Boxplot:** Shows quartiles, median, whiskers, and outliers
- **Histogram:** Shows frequency distribution
- **QQ Plot:** Compares sample quantiles to theoretical quantiles
- **Scatter Plot:** Shows bivariate relationships
- **Violin Plot:** Combines boxplot with kernel density estimate

---

## 📧 Contact & Support

For questions about this project:
- Review the HTML report for detailed explanations
- Check Python script comments for code-level documentation
- Examine the insight boxes in HTML for interpretations
- See recommendations sections for actionable insights

---

**Last Updated:** 2024-2025  
**Status:** Complete ✓  
**All 10 Questions Covered:** Yes ✓  
**English Report Included:** Yes ✓  
**Author Information Added:** Yes ✓

https://linkedin.com/in/abolfazl-omidiyan
