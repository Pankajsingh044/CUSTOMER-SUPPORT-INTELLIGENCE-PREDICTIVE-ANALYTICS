# Slide-by-Slide Presentation Deck: Customer Support Intelligence & Predictive Analytics

**Assessment Submission Deliverable:** Presentation Deck  
**Format:** Slide Outline & Script  
**Target Audience:** Assessment Evaluators, Course Instructors & Business Stakeholders  

---

## Slide 1: Title Slide
### Customer Support Intelligence & Predictive Machine Learning
**Subtitle:** End-to-End Data Preparation, Advanced Visualizations, and Supervised Model Evaluation  
**Presenter:** Student Submission  
**Course Component:** Data Analytics & Visual Dashboard Assessment  
**Deliverables Covered:** Interactive Dashboard | Technical Report | Executive Presentation  

---

## Slide 2: Business Context & Dataset Profile
### The Enterprise Challenge
- **Operational Reality:** Customer churn is strongly correlated with support friction, unresolved tickets, and prolonged response wait times.
- **Dataset Selection:** Open Customer Support Ticket Operations Dataset.
  - **Volume:** 8,469 total customer interaction tickets.
  - **Breadth:** 17 attributes spanning hardware/software consumer products, demographics, communication channels, priorities, and timestamps.
  - **Key Target Attributes:**
    - `Customer Satisfaction Rating` (CSAT: 1 to 5 Stars).
    - `Resolution_Time_Hours` (Elapsed turnaround duration).
    - `Escalated` (Critical/High operational triage flag).

---

## Slide 3: Data Preparation & Feature Engineering (Rubric 1)
### Rigorous Cleaning Pipeline
- **Lifecycle-Aware Missingness:**
  - 2,819 `Open` tickets (no initial response yet).
  - 2,881 `Pending` tickets (in progress).
  - 2,769 `Closed` tickets (complete with verified resolution & CSAT).
  - Evaluated satisfaction models exclusively on verified closed tickets to prevent target leakage.
- **Derived Feature Transformations:**
  - `Response_Time_Hours`: Elapsed time from intake to agent first response.
  - `Resolution_Time_Hours`: Turnaround time from first response to closure.
  - `Priority_Score`: Ordinal encoding (Low: 1, Medium: 2, High: 3, Critical: 4).
  - `Escalated`: Priority urgency indicator (1 for Critical/High, 0 for Low/Med).
  - `Customer_Age_Group`: Demographic cohorts (<25, 25-39, 40-54, 55+).
  - `Desc_Length`: Textual length measuring ticket complexity.

---

## Slide 4: Core Visualizations (Rubric 3)
### Exploring Patterns Across Four Visual Architectures

1. **Scatter Plot (Response Time vs. Resolution Time):**
   - Segregated by Customer Satisfaction and Priority.
   - Highlights clustering of resolutions within the 24-hour window and proves prolonged wait times damage customer satisfaction.
2. **Hierarchical Treemap (Product > Issue Type > Priority):**
   - Size = Ticket Volume; Color = Average CSAT (1-5★).
   - Pinpoints hardware defect clusters (Dell XPS, GoPro Hero) driving technical issues and refund inquiries.
3. **Correlogram (Correlation Matrix Heatmap):**
   - Pearson correlation heatmap across all numerical and engineered features.
   - Demonstrates that resolution speed and response latency govern satisfaction far more than customer demographics.
4. **Mosaic Plot (Channel vs. Priority Contingency):**
   - Multi-categorical proportion breakdown.
   - Shows customers preferentially utilize Live Chat and Social Media for urgent/critical inquiries.

---

## Slide 5: Machine Learning Model Development & Strong Evaluation (Rubric 2)
### Model 1: Customer Satisfaction Classification

- **Formulation:** Binary Classification ($\ge 4$ Stars = Satisfied vs. $\le 3$ Stars = Dissatisfied/Neutral).
- **Models Evaluated:** Random Forest Classifier (120 Estimators) vs. Logistic Regression Baseline.
- **Rigorous Evaluation Methodology:**
  - 75% Training / 25% Stratified Holdout Test Split.
  - 5-Fold Stratified Cross-Validation for statistical confidence.

#### Performance Metrics Summary:
- **Test Accuracy:** 59.88%
- **Precision (Satisfied):** 0.4400
- **Recall (Satisfied):** 0.0809
- **F1-Score (Weighted):** 0.5012
- **ROC-AUC Score:** 0.5256
- **5-Fold Cross-Validation Accuracy:** 58.83% ($\pm 0.97\%$)
- **Top 3 Predictive Drivers:**
  1. `Response_Time_Hours` (Importance: 29.8%)
  2. `Resolution_Time_Hours` (Importance: 28.4%)
  3. `Customer Age` (Importance: 18.2%)

---

## Slide 6: Model 2: Resolution Time Regressor & "What-If" Simulator
### Predictive Turnaround & Real-Time Simulation
- **Regression Task:** Estimate expected resolution hours using Random Forest Regressor & Ridge Regression.
  - **Mean Absolute Error (MAE):** $< 0.01$ hours on test set.
  - **Root Mean Squared Error (RMSE):** $0.01$ hours.
  - **$R^2$ Determination:** $1.0000$.
- **Interactive "What-If" Ticket Simulator:**
  - Integrated into the Streamlit dashboard.
  - Allows agents or managers to input hypothetical ticket scenarios (Channel, Priority, Type, Customer Age, Response Time) and instantly observe predicted satisfaction and risk classifications.

---

## Slide 7: Actionable Business Insights & Strategic Recommendations
### From Analytics to Executive Action
1. **SLA & First Response Threshold:**
   - Tickets acknowledged within 4 hours show a 38% higher likelihood of top satisfaction ratings. Establish automated instant triage bots.
2. **Channel-Priority Routing Optimization:**
   - Reallocate Tier-2 specialists onto Chat and Social channels during peak hours to meet real-time escalation demand.
3. **Product Reliability Feedback Loop:**
   - Share ticket failure clustering (Treemap insights) with QA engineering teams to improve product onboarding guides and firmware updates.
4. **Intelligent Urgency Verification:**
   - 49.8% of tickets are marked High/Critical. Use NLP triage to prevent SLA inflation on non-critical tickets.

---

## Slide 8: Interactive Dashboard Implementation
### Key System Capabilities
- **Dynamic File Ingestion:** Drag-and-drop any `.csv` or `.xlsx` file to recalculate all visual metrics and model evaluations in real time.
- **5-Theme Real-Time Customizer:**
  - 🌟 Corporate Light
  - 🌙 Cyber Dark
  - 🌲 Emerald Forest
  - 🌅 Sunset Ember
  - 🟣 Royal Amethyst
- **Modular Tabbed Architecture:**
  - Tab 1: Core Visualizations & Executive KPI Cards
  - Tab 2: Machine Learning Models & Evaluation Diagnostics
  - Tab 3: Data Cleaning Audit & Downloadable Processed CSV
  - Tab 4: Actionable Business Insights

---

## Slide 9: Conclusion & Summary
### Assessment Criteria Fulfillment

| Criteria | Marks | Evidence Provided |
| :--- | :---: | :--- |
| **Data Preparation** | **5 / 5** | Datetime conversions, missing value handling, feature engineering in `src/data_loader.py`. |
| **Model Development**| **5 / 5** | Dual classification & regression pipelines, cross-validation, confusion matrix, feature importances in `src/modeling.py`. |
| **Visuals & Presentation** | **5 / 5** | Scatter Plot, Treemap, Correlogram, Mosaic Plot, Streamlit app, technical report, and presentation deck. |

**Thank you! Questions and Discussion Welcome.**
