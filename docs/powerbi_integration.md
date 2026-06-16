# Power BI Integration Guide (End-to-End)

This guide provides micro-level, step-by-step instructions to connect your Power BI environment to the Healthcare Capacity Intelligence Platform (HCIP) machine learning predictions, construct the relational schema, and build insightful visual dashboards.

---

## Phase 1: Ingest Data via Power Query

Since our training pipeline automatically outputs a bulk parquet file of predictions, loading the `.parquet` file directly is the most efficient method for Power BI.

### Baby Steps to Load Data:
1. Open **Power BI Desktop**.
2. Close the splash screen if it appears.
3. On the Home ribbon, click **Get Data** > **More...**
4. In the search box, type `Parquet` and select the **Parquet** connector, then click **Connect**.
5. A prompt will appear asking for a **URL**. This refers to the absolute file path on your machine.
6. Paste the full path to the local file (e.g., `F:\healthcare-capacity-intelligence-platform\data\processed\fact_predictions.parquet`) and click **OK**.
7. A preview window will appear.
8. Click **Transform Data** (Do NOT click Load yet). This opens the **Power Query Editor**.

### Baby Steps to Format Data:
1. In Power Query Editor, look at the `fact_predictions` table. We need to ensure data types are correct.
2. Right-click the `provider_code` column header -> **Change Type** -> **Text**.
3. Right-click the `specialty_code` column header -> **Change Type** -> **Text**.
4. Right-click the `period_date` column header -> **Change Type** -> **Date**.
5. Right-click `pred_breach_prob` -> **Change Type** -> **Decimal Number**.
6. Right-click `pred_total_next` -> **Change Type** -> **Whole Number**.
7. Right-click `pred_pct_next` -> **Change Type** -> **Decimal Number**.
8. In the top-left corner of the ribbon, click **Close & Apply**. The data will now load into your model.

> [!NOTE]
> *Repeat the steps above to also load your other tables from the `data/gold/` folder: `dim_hospital.parquet`, `dim_specialty.parquet`, `dim_date.parquet`, and your historical data `fact_waiting_list.parquet`.*

---

## Phase 2: Establish the Semantic Model

Power BI needs to know how your predictions link to your hospital and specialty names.

### Baby Steps to Link Data:
1. On the far-left sidebar, click the **Model view** icon (it looks like three connected boxes).
2. You should see all your tables (`fact_predictions`, `fact_waiting_list`, `dim_hospital`, `dim_specialty`, `dim_date`).
3. Link **`fact_predictions`**:
   - Drag `provider_code` to `dim_hospital[provider_code]`.
   - Drag `specialty_code` to `dim_specialty[specialty_code]`.
   - Drag `period_date` to `dim_date[period_date]`.
4. Link **`fact_waiting_list`** (historical data):
   - Drag `provider_code` to `dim_hospital[provider_code]`.
   - Drag `specialty_code` to `dim_specialty[specialty_code]`.
   - Drag `period_date` to `dim_date[period_date]`.
5. Your complete Star Schema is now established!

---

## Phase 3: Create DAX Measures

Measures are formulas that calculate on the fly when you click different filters.

### Baby Steps to Create Measures:
1. On the far-left sidebar, click the **Report view** icon (it looks like a bar chart).
2. On the right-side **Data pane**, right-click the `fact_predictions` table and select **New measure**.
3. A formula bar will appear at the top. Copy and paste the following DAX formula:
   ```dax
   Predicted Total Waiting = SUM(fact_predictions[pred_total_next])
   ```
4. Press **Enter**. You will see the measure appear in the Data pane with a calculator icon.
5. Right-click `fact_predictions` again > **New measure**, and paste:
   ```dax
   Risk Status = 
   VAR AvgProb = AVERAGE(fact_predictions[pred_breach_prob])
   RETURN
   SWITCH(
       TRUE(),
       ISBLANK(AvgProb), BLANK(),
       AvgProb >= 0.85, "🔴 Critical Risk",
       AvgProb >= 0.60, "🟡 Elevated Risk",
       "🟢 Stable"
   )
   ```
6. Press **Enter**. 
7. Create one last measure:
   ```dax
   Forecasted % Within 18 Weeks = AVERAGE(fact_predictions[pred_pct_next])
   ```

---

## Phase 4: Constructing the Dashboards

Now you will build the actual visual charts. Here is a curated list of suggested charts and exactly how to configure them.

### Chart 1: The "Headline" KPI Cards
*What it shows: High-level snapshot of predicted performance.*
1. Click the **Card** visual icon in the Visualizations pane.
2. Drag the `Predicted Total Waiting` measure into the **Fields** well.
3. Create a second **Card** visual.
4. Drag the `Forecasted % Within 18 Weeks` measure into the **Fields** well.
5. Format the second card to display as a Percentage (click the measure in the Data pane, then click the `%` icon in the top ribbon).

### Chart 2: Predicted Breach Risk Heatmap
*What it shows: A grid identifying exactly which hospitals and specialties are going to fail the NHS 18-week standard next month.*
1. Click the **Matrix** visual icon.
2. Drag `dim_hospital[provider_name]` into the **Rows** well.
3. Drag `dim_specialty[specialty_name]` into the **Columns** well.
4. Drag the `fact_predictions[pred_breach_prob]` field into the **Values** well. Ensure it is aggregating as **Average** (Right-click it in the Values well -> Average).
5. Format as Percentage: Click `pred_breach_prob` in the Data pane, then click the `%` icon on the top ribbon.
6. Go to the **Format your visual** pane (paintbrush icon) > **Cell elements**.
7. Turn on **Background color**. A gradient will automatically apply.
8. *Pro Tip (True Heatmap)*: If you want to hide the numbers and just show solid color blocks, turn on **Font color** in the Cell elements pane and apply the exact same gradient rules as the background.

### Chart 3: Demand Forecast Trendline
*What it shows: The historical growth of the waiting list compared to our ML forecast.*
1. Click the **Line and clustered column chart** visual icon.
2. Drag `dim_date[period_date]` into the **X-axis**.
3. Drag your historical actuals (e.g., `fact_waiting_list[total_waiting]`) into the **Column y-axis**.
4. Drag the `Predicted Total Waiting` measure into the **Line y-axis**.
5. Now you can visually see the historical bars transitioning into the predicted ML line.

### Chart 4: Worst-Offenders Scatter Plot
*What it shows: Outliers that have both a massive waiting list AND a high probability of breaching.*
1. Click the **Scatter chart** visual icon.
2. Drag `Predicted Total Waiting` into the **X-axis**.
3. Drag `fact_predictions[pred_breach_prob]` into the **Y-axis**.
4. Drag `dim_hospital[provider_name]` into the **Values** well.
5. Drag `dim_specialty[specialty_name]` into the **Legend**.
6. *Interpretation*: Any dot in the top-right corner is a massive bottleneck. Hovering over it will reveal the exact hospital and specialty.
