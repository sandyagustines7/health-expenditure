**Mapping Global Health Expenditure Trends and the Future: Insights from 192 Countries (2000-2020)**

**Project Description**

This project aims to analyze global health spending across 192 countries over two decade, wtih a focus on understanding how expenditure patterns have evolved, what drives them and what they might look like in the future. 

Using PostGreSQL for structured data amanagement and Python for modeling and analysis, the project combines descriptive statistics, predictive modeling (ARIMA), regression analysis and unsupervised learning (clustering) to extract actionable insights. TWhe results are surfaced through an interactive Tableau dashboard for policy and investment stakeholders. 

**Tools and Technologies**
- PostgreSQL: data storage, cleaning and transformation
- Python (Pandas, statsmodels, scikit-learn): EDA, ARIMA forecasting, regression modeling, clustering
- Tableau: interactive dashboard for exploration and visualization

**Data Sources:**
- Global Health Expenditure Database (GHED): https://apps.who.int/nha/database/Home/Index/en
- Global Health Observatory (GHO): https://www.who.int/data/gho 

These are the WHO's public databases, which contain a list of indicators and health outcomes that are easily filterable depending on what you are looking for. 

**Research Questions**

These research questions were created in order to better understand where and how countries have invested into global healthcare, and whether these trends are subject to change in the forseeable future. 

1. Spending Trends: How has health expenditure evolved globally from 2000-2020?
2. Forecasting: What do predictive models suggest about future spending trajectories?
3. Impact: How do different types of spending (e.g. public vs private) relate to health outcomes such as life expectancy and mortality rates?
4. Comparative Insights: How do countries cluster by spending profiles? What regional or economic similarities emerge?

**Current Outputs** 
- Cleaned and normalized database with metadata on regions, countries and indicators
- Exploratory Data Analysis (e.g. KPI summaries, correlation metrics)
- ARIMA-based forecasts on future spending
- Regression models exploring relationship between expenditure and health outcomes
- Clustering analysis to identify similar country profiles
- Tableau dashboard for interactive visual exploration (in progress)

**Status**

This project is in its final stages. All core analyses is complete and the dashboard is being finalized for publication. 
