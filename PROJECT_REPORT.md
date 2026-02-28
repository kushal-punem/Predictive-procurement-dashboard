# University Bulk Order & Predictive Procurement Analytics System  

## 1. Overview

This project implements an end‑to‑end **Predictive Procurement Analytics System** for university textbook ordering.  
The goal is to replace naïve “order for 100% of enrollment” rules with a **behavioral, ML‑driven forecast** of which students will actually purchase the university bundle.

The solution has three major layers:

- **Data Layer (Snowflake‑style schema)** – fact/dimension model for enrollments, students, adoptions, and sections.
- **Analytics & ML Layer** – feature engineering, demand and opt‑out modeling, and feature importance.
- **Decision Dashboard (Streamlit + Plotly)** – interactive UI for procurement officers with filters, KPIs, and risk views.

---

## 2. Data Architecture & Schema

The logical schema is implemented in `schema.sql` and follows a **snowflake** pattern.

### 2.1 Fact Table – `ENROLLMENTS`

- **Grain**: one row per **student–course section** registration.
- **Key fields**:
  - `enrollment_id` – surrogate PK.
  - `sis_user_id` – FK to `STUDENT_MASTER`.
  - `section_id` – FK to `SECTIONS`.
  - `enrollment_date` – registration date.
  - `fee_included_flag` – whether the course uses “inclusive access” billing.
  - `actual_purchase_flag` – derived from transaction data (target label).
  - `predicted_purchase_prob` – model‑generated probability.
  - `term_code` – normalized term identifier (e.g. `2026FA`).

This table captures **behavioral outcomes** (buy / opt‑out) joined to all necessary context via dimensions.

### 2.2 Dimension – `STUDENT_MASTER`

Represents a student’s **financial and demographic profile**:

- `sis_user_id` (PK)
- `financial_condition` – e.g., Low / Medium / High or custom buckets.
- `region` – local vs out‑of‑state, metro vs rural, etc.
- `gpa` – continuous academic performance indicator.
- `commuter_distance_km` – distance from campus.
- `scholarship_type` – merit, need‑based, athletic, etc.
- `housing_status` – on‑campus, off‑campus, commuter.

These attributes feed **economic sensitivity** and **commuter friction** features.

### 2.3 Dimension – `ADOPTIONS`

Represents the **product being ordered** (book or digital code):

- `isbn` (PK)
- `title`
- `publisher`
- `retail_new_price`
- `retail_rent_price`
- `is_digital` – 1 for access codes / platforms.
- `is_required` – required vs recommended.
- `course_level` – lower‑div, upper‑div, grad, etc.

Used to compute **Arbitrage Index**, **Digital Lock Flag**, and price attributes.

### 2.4 Dimension – `SECTIONS`

Represents the **context** in which the adoption is used:

- `section_id` (PK)
- `course_code`
- `dept_code` – CSE, HUM, ART, etc.
- `term_year`
- `instructor`
- `modality` – online / hybrid / in‑person.

This dimension connects enrollment behavior back to **departmental risk** and term scenarios.

### 2.5 Relationship Summary

- `ENROLLMENTS.sis_user_id → STUDENT_MASTER.sis_user_id`
- `ENROLLMENTS.section_id → SECTIONS.section_id`
- Section adoptions (course_code + isbn) join through **adoption mapping** (not explicitly modeled here but assumed in ETL).

---

## 3. ETL & Feature Table

The synthetic ETL is implemented in `etl_pipeline.py` via `load_feature_table()`.  
In production, this function would:

1. Ingest raw CSVs / warehouse tables for:
   - Student master data
   - Enrollment transactions
   - Adoption catalog and pricing
   - Section / timetable data
2. Apply data quality checks (null handling, type casting, normalization).
3. Join facts and dimensions into a **feature table** at the `ENROLLMENTS` grain.
4. Persist this table as a view or materialized table for modeling and dashboard use.

The synthetic implementation generates a DataFrame with these key columns:

- `Term` – academic term label.
- `Dept_Code` – department code.
- `Publisher` – publisher name.
- `Student_Type` – Full‑Time / Part‑Time.
- `Format` – Digital / Physical.
- `Rental_to_Retail_Ratio` – rental price ÷ new retail price.
- `Wallet_Pressure_Score` – synthetic 0–1 score approximating total semester expense.
- `Digital_Lock_Flag` – 1 if the title is an access code or similar non‑resellable asset.
- `Major_Alignment_Score` – 0–1; higher when course is in the student’s own major.
- `Commuter_Friction` – 0–1; higher for long‑distance commuters or off‑campus students.
- `Arbitrage_Index` – `1 - Rental_to_Retail_Ratio`.
- `Opt_Out_Probability` – pseudo‑ground‑truth probability of opting out (for demo).
- `Predicted_Purchase_Prob` – `1 - Opt_Out_Probability`.
- `Predicted_Demand_Units` – expected units per record (1–3).
- `Unit_Price` – synthetic per‑unit price.
- `Projected_Spend` – `Predicted_Demand_Units * Unit_Price * Predicted_Purchase_Prob`.

This table is the **core input** for both modeling and the dashboard.

---

## 4. Feature Engineering Strategy

Feature engineering captures **economic pressure**, **behavioral lock‑in**, and **logistical frictions** that influence purchase behavior.

### 4.1 Economic Sensitivity Features

- **Rental‑to‑Retail Ratio** (`Rental_to_Retail_Ratio`)
  - Definition: \(\text{rent price} / \text{new retail price}\).
  - Interpretation:
    - Values **≪ 1.0** → renting is much cheaper; students are more likely to opt out of the bundle.
  - Used to derive **Arbitrage Index**.

- **Arbitrage Index** (`Arbitrage_Index`)
  - Definition: \(1 - \text{Rental_to_Retail_Ratio}\).
  - High values indicate **strong arbitrage opportunities** (big savings from external options).
  - Strong positive correlation with `Opt_Out_Probability`.

- **Wallet Pressure Score** (`Wallet_Pressure_Score`)
  - Composite measure of:
    - Course load.
    - Total required material cost.
    - Financial condition / aid.
  - Modeled here as a synthetic 0–1 value where higher = more price sensitive.

**Economic Hypothesis:**  
Higher `Arbitrage_Index` and `Wallet_Pressure_Score` → higher probability of opting out.

### 4.2 Behavioral Lock‑In Features

- **Digital Lock Flag** (`Digital_Lock_Flag`)
  - 1 if the adoption is a **digital access code / platform** that cannot be rented or resold.
  - These items are usually only available through the official channel.
  - Expected effect: strongly **reduces** opt‑out (students must obtain them).

- **Major Alignment Score** (`Major_Alignment_Score`)
  - Measures alignment between student’s major and course department.
  - 1 for core/major courses, 0 for unrelated electives (synthetic in this demo).
  - Expectation: students keep or invest more in books for their **major**, but are more cost‑sensitive for electives.

### 4.3 Logistical Features

- **Commuter Friction** (`Commuter_Friction`)
  - A function of `commuter_distance_km` and `housing_status`.
  - Higher values indicate more friction in handling physical books (transport, storage).
  - Leads to preference for **digital format** and may change opt‑out behavior if digital bundles are convenient.

---

## 5. Machine Learning Methodology

### 5.1 Target Variable

- **`Actual_Purchase_Flag`** (not explicitly present in synthetic data, but conceptually):
  - 1 → Student obtained the bundle through the university program.
  - 0 → Student opted out or sourced materials elsewhere.

In a real pipeline, this would be derived from bookstore sales or inclusive‑access billing records.

### 5.2 Model Choice

- Intended algorithms:
  - **Random Forest Classifier**
  - Or **XGBoost** (gradient boosted trees)

Reasons:

- Handle **non‑linear relationships** (e.g., threshold effects on price).
- Naturally work with **tabular** mixed‑scale data.
- Provide **feature importance** values to explain predictions to procurement teams.

### 5.3 Feature Importance

`feature_engine.py` currently exposes `compute_feature_importance_example()` which returns a static table used by the dashboard. In a full implementation:

1. Train a tree‑based model on the engineered features.
2. Extract `model.feature_importances_`.
3. Map importances back to feature names.
4. Persist this as a table or JSON object for the dashboard.

The dashboard’s **“Feature Importance (Model Explainability)”** chart then answers:

> “Which factors (price ratios, wallet pressure, digital lock, etc.) contribute most to opt‑out risk?”

### 5.4 Evaluation Metrics

When training the model, focus on:

- **Precision** on the positive class (predicting purchase):
  - High precision ↓ over‑ordering cost (fewer false positives where we order but students don’t buy).
- **Recall** on the positive class:
  - High recall ↓ stockouts (we don’t miss students who actually will buy).
- **AUC‑ROC / PR‑AUC**:
  - For overall ranking quality across probability thresholds.

---

## 6. Dashboard Design (Streamlit + Plotly)

The UI is implemented in `dashboard_app.py` and consumes the feature table from `load_feature_table()`.

### 6.1 Left Filter Bar – Term & Segments

The left vertical bar includes:

- `Term`
- `Department`
- `Publisher`
- `Student Type`
- `Format`

These filters operate on the in‑memory DataFrame and drive all KPIs and visuals on the right.

### 6.2 Executive KPI Row

Functions implemented in `render_top_kpis`:

- **Total Predicted Demand** – \(\sum \text{Predicted_Demand_Units}\).
- **Total Projected Spend** – \(\sum \text{Projected_Spend}\).
- **Digital vs Physical Split** – share of predicted units by `Format`.
- **High‑Risk Opt‑Out Rate** – fraction of records with `Opt_Out_Probability > 0.6`.

These cards give a **one‑glance view** of demand, spending, and risk.

### 6.3 Feature Engineering & Key Indicators Row

Three side‑by‑side charts:

1. **Price Sensitivity & Opt‑Out Threshold**
   - Scatter of `Rental_to_Retail_Ratio` vs `Opt_Out_Probability`.
   - Colored by `Dept_Code`.
   - Shows how stronger arbitrage opportunities (low rent‑to‑retail) correlate with opt‑outs.

2. **Feature Importance (Model Explainability)**
   - Horizontal bar chart of top engineered features and their relative importances.
   - Helps justify model recommendations to non‑technical stakeholders.

3. **Format Preference by Segment**
   - Grouped bar chart by `Student_Type` and `Format`.
   - Helps answer: “For which segments should we increase digital capacity or maintain physical inventory?”

### 6.4 Procurement Planning & Strategy Row

1. **Funding Source Planning & Strategy (Sankey Diagram)**
   - Simplified flow from funding sources (Financial Aid / Self‑Pay / Scholarship) to Opt‑In / Opt‑Out.
   - Intended to be replaced by real payment‑source breakdowns.

2. **Procurement Risk by Department**
   - Bar chart of average `Opt_Out_Probability` by department (top 5).
   - Highlights **departments where over‑ordering risk is highest**.

3. **Recommended Actions Panel**
   - Textual recommendations generated from high‑risk department metrics.
   - Example actions:
     - Negotiate better pricing in those departments.
     - Increase digital options for high commuter‑friction segments.
     - Re‑evaluate “required” status for high‑price, low‑utilization titles.

---

## 7. How the Pieces Work Together

1. **Data → Feature Table**
   - ETL merges student, section, adoption, and transaction data.
   - Feature engineering computes economic, behavioral, and logistical signals.

2. **Feature Table → Model**
   - ML model estimates `Predicted_Purchase_Prob` for each enrollment row.
   - Aggregations produce demand and spend forecasts by segment.

3. **Model Output → Dashboard**
   - Streamlit app loads the feature table (and, in a full build, model scores + importances).
   - Users slice by term, department, publisher, student type, and format.
   - Visuals expose both **what** the demand looks like and **why** the model predicts it.

This closes the loop from **raw data → ML insights → procurement decisions**.

---

## 8. Future Enhancements

- **Live Data Integration**
  - Replace synthetic data in `etl_pipeline.py` with warehouse queries or API calls.
  - Schedule nightly or intra‑day refreshes.

- **Real Model Training**
  - Implement full training pipeline (`train_model.py`) with:
    - Train/validation/test splits.
    - Hyperparameter tuning.
    - Model registry.

- **Scenario Planning**
  - Add a “What‑If” control panel:
    - Adjust prices, discounts, or adoption policies.
    - Recompute projected spend and demand on the fly.

- **Power BI / Tableau Layer**
  - Optional business‑facing layer that consumes the same feature table for more pixel‑perfect reporting.

---

## 9. Files Summary

- `schema.sql` – DDL for fact and dimension tables.
- `etl_pipeline.py` – synthetic ETL + feature table generator.
- `feature_engine.py` – placeholder for model‑driven feature importance.
- `dashboard_app.py` – Streamlit UI and Plotly charts.
- `requirements.txt` – Python dependencies.
- `README.md` – quick‑start instructions.
- `PROJECT_REPORT.md` – this in‑depth technical and conceptual documentation.

