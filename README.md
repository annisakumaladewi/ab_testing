## The A/B Testing Automation Module

The `ab_testing_functions.py` file provides reusable tools for:

### **Internal Validity**
- Sample Ratio Mismatch (SRM) checks  
- Demographic / covariate balance tests  
- A/A test validation  
- Simpson’s paradox detection (via hashed segments)

### **Hypothesis Testing**
- Mean difference tests (Cohen’s d)
- Proportion Z-tests (Cohen’s h)
- Delta method for ratio metrics

### **Power & Sample Size**
- ANOVA power analysis (means)
- Multi-group proportion sample size estimation

### **Utilities**
- Pairwise comparison helpers  
- Bonferroni correction  
- Standardized results combination  

You can import and use the module like:

```python
import ab_testing_functions as abt

abt.srm_check(df, "variant")
abt.pairwise_proportion_tests(df, "variant", "converted")
abt.pairwise_delta_test(df, "variant", "clicks", "views", "user_id")
