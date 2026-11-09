# AI-Powered KPI Monitoring Dashboard for Enterprise BI

Run: `pip install -r requirements.txt` then `python app.py` and open http://127.0.0.1:5000

| Project module | Where |
|---|---|
| 3. Dataset & preprocessing | `kpi_engine.generate()` / `load()` (CSV in `data/`, duplicates and nulls dropped) |
| 4. KPI calculation | `kpi_engine.monthly()` / `kpi_cards()` (value, MoM, QoQ, YoY) |
| 5. Dashboard | `app.py`, `templates/index.html` (Flask + Chart.js, six Power BI-style slicers) |
| 6. AI/ML | `anomalies()` Isolation Forest on trend+seasonality residuals, with a driver explanation; `forecast()` linear trend + seasonality, 6 months, 80% range |
| Insights | `insights()` auto-generated plain-language summary |

To use real data, replace `data/business_data.csv` with the columns:
date, region, category, revenue, cost, profit, customers, employee_score, efficiency.

## Slicers (Power BI slicer types)

All slicers are AND-ed together; an empty selection means "no filter". Use *Clear all slicers* to reset.

| Slicer | Type | Filters |
|---|---|---|
| Category | List (checkboxes, multi-select) | `category` |
| Year | Dropdown | calendar year of `date` |
| Month range | Date (dual-handle range) | `date`, by month |
| Revenue per record | Numeric (dual-handle range) | `revenue` of each region/category/month row |
| Region | Chiclet (toggle buttons, multi-select) | `region` |
| Region > Category | Hierarchy (expandable tree) | specific region + category pairs |

Notes
- Forecast and anomaly detection need at least 12 months of history in the selection (`MIN_MONTHS`); with less, the
  dashboard still shows KPIs and trends and says why the forecast is off. YoY needs 13 months, QoQ 6, MoM 2.
- The Numeric slicer removes individual rows, so monthly totals for a narrowed range cover only the rows that match.
- API: `/api/dashboard?regions=North,East&categories=Apparel&pairs=East|Apparel&year=2024&start=2024-01&end=2024-12&rmin=50000&rmax=90000`
  (all optional; the old `?region=` and `?category=` parameters still work).
