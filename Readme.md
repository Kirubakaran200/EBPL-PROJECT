This project uses smart regression models (Ridge, Random Forest, XGBoost) to forecast house prices using the **California Housing** dataset.

- Built-in dataset (no upload required)
- Data preprocessing and scaling
- Cross-validation for model evaluation
- Comparison of Ridge, Random Forest, and XGBoost
- Actual vs Predicted plot for visual validation

- Source: [`sklearn.datasets.fetch_california_housing`](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_california_housing.html)
- Features: 8 numerical features
- Target: Median House Value (converted to `SalePrice`)

Install all dependencies using:

```bash
pip install -r requirements.txt
