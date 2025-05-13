# 🏠 Feature Engineering for House Price Prediction

This document outlines the transformations applied to the original dataset to enhance the performance of regression models (`XGBoost` and `Random Forest`). These features were empirically validated, improving the model's R² score to 0.85.

---

## 📌 Original Features (13)

```
['sqft_living', 'sqft_lot', 'bedrooms', 'bathrooms', 'floors',
 'waterfront', 'view', 'condition', 'grade', 'sqft_above',
 'sqft_basement', 'yr_built', 'yr_renovated']
```

---

## 🧠 Strategic decisions by type of variable

| Variable                    | EDA Observation                          | Recommended Action             |
| --------------------------- | ---------------------------------------- | ------------------------------ |
| `sqft_living`             | Possibly skewed                          | Log-transform                  |
| `sqft_lot`                | High variance, heavy tail                | Log-transform                  |
| `sqft_above`              | High correlation with `sqft_living`    | Keep only one or combine       |
| `sqft_basement`           | Many zeros + collinearity with others    | Add binary feature + log1p     |
| `bedrooms`, `bathrooms` | Little range                             | Keep, no transformation        |
| `floors`                  | Can be considered ordinal or categorical | Review distribution / binarize |
| `view`, `condition`     | Ordinal but subjective                   | Treat as ordinal / categorize  |
| `grade`                   | Strongly correlated with `price`       | Keep as ordinal                |
| `yr_built`                | Lower correlation with `price`         | Calculate age (2025 - year)    |
| `yr_renovated`            | Many zeros                               | Binary feature + age           |
| `waterfront`              | Binary                                   | Keep unchanged                 |

<hr>

## ⚙️ Applied Transformations

### 1. `age` (Property Age)

- **Formula:** `age = 2025 - yr_built`
- **Rationale:** Represents the actual age more effectively than the raw year built.

### 2. `was_renovated` (Renovation Indicator)

- **Formula:** `was_renovated = 1 if yr_renovated > 0 else 0`
- **Rationale:** Indicates whether the property has undergone modernization.

### 3. `total_rooms` (Functional Room Count)

- **Formula:** `total_rooms = bedrooms + bathrooms`
- **Rationale:** Better represents the usable size of the home.

### 4. `sqft_ratio` (Living Area to Lot Size Ratio)

- **Formula:** `sqft_ratio = sqft_living / sqft_lot`
- **Rationale:** Measures building density in relation to the land size.

### 5. `has_basement` (Basement Indicator)

- **Formula:** `has_basement = 1 if sqft_basement > 0 else 0`
- **Rationale:** Captures the presence of a valuable feature in the property.

---

## ✅ Final Selected Features(14)

```
['sqft_living', 'grade', 'bathrooms', 'view', 'sqft_above',
 'sqft_basement', 'floors', 'waterfront', 'condition', 
 'age', 'was_renovated', 'total_rooms', 'sqft_ratio', 'has_basement']
```

---

## 📈 Results

- **Model Used:** `XGBoostRegressor`
- **R² Score:** `0.85` on the test set.
- **Conclusion:** These feature transformations allow the model to capture nonlinear relationships and implicit effects, significantly enhancing predictive power.
