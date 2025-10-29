Task Overview
--------------
The classification label is located in the column “COHORT”, which includes the following diagnostic groups:

- PD Participant — diagnosed Parkinson’s disease patients  
- Healthy Control — participants with no Parkinson’s symptoms  
- SWEDD — participants initially suspected of PD but showing Scans Without Evidence of Dopaminergic Deficit  
- Prodromal — individuals exhibiting early, pre-diagnostic symptoms  


Modeling Options
----------------
You may choose one of the following two modeling approaches:

1. **Random Forest Classification** with feature importance analysis.  
2. **TabTransformer-based Classification** using the implementation available at:  
   https://github.com/lucidrains/tab-transformer-pytorch


Notes and Guidelines
--------------------
- The selected feature columns are already specified in `Selected Features.docx`.  
  No additional feature selection is required.

- When splitting the data into training and validation sets, consider the **“EVENT_ID”** column:  
  - Each patient may have multiple visits.  
  - All visits from the same patient must be placed entirely in either the training or validation set.  
  - The first visit of each patient is labeled **“BL” (baseline)** in the “EVENT_ID” column.  
  - Refer to the dataset documentation for further details about “EVENT_ID”.


Evaluation Metrics
------------------
Use the following metrics to assess model performance:

- Overall accuracy  
- Per-class precision, recall, and F1-score  

If the Random Forest option is selected, also identify and rank the **top 10 most important features** contributing to the classification (e.g., motor scores, demographic variables, imaging metrics, etc.).
