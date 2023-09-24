# Data Science Salary Predictor: Overview

* Created a model that estimates data science salaries (MAE ~ $ 25K) in the USA.
* Cleaned the Glassdoor data science salary dataset
* Engineered features to quantify the demand for data skills (python, excel, spark, etc.)
* Trained & Optimized Linear, Lasso, and Random Forest Regressors using GridsearchCV to find the best parameters
* Built a client-facing API using flask

## Code and Resources used
**Python Version:** 3.10.6 <br>
**Packages:** pandas, numpy, sklearn, matplotlib, seaborn, selenium, flask, json, pickle <br>
**For Web Framework Requirements:** pip install -r requirements.txt <br>
**Flask productionization:** https://towardsdatascience.com/productionize-a-machine-learning-model-with-flask-and-heroku-8201260503d2 <br>
**Dataset:** https://www.kaggle.com/datasets/rashikrahmanpritom/data-science-job-posting-on-glassdoor/code?resource=download <br>

# Data Cleaning
Cleaned the data to make it suitable for analysis and modeling. Performed the following actions: <br>

* Created columns for min, max, and avg salary, parsing from the salary estimate
* Made columns for employer-provided salary and hourly wages
* Removed rows without salary
* Parsed rating out of company text
* Added a column for if the job was at the company’s headquarters
* Transformed the founded date into the age of the company
* Made columns for if different skills were listed in the job description:
* Python
* Excel
* AWS
* Spark
* Tableau
* Added a column for simplified job titles and Seniority
* Added a column for description length

# Exploratory data analysis
* Analyzed the cleaned data to generate insights & trends between average salary and age of company, location, skills, seniority, industry, job title
* Generated boxplots to visualize descriptive statistics of numerical columns (rating, age, average salary, description length)
![Alt text](https://github.com/cipher499/ds_salary_project/blob/master/barchart.png)
![Alt text](https://github.com/cipher499/ds_salary_project/blob/master/heatmap.png)
![Alt text](https://github.com/cipher499/ds_salary_project/blob/master/pivottable.png)
![Alt text](https://github.com/cipher499/ds_salary_project/blob/master/word_cloud.png)

# Model Building
* One-hot encoded the categorical variables using the get_dummy method of pandas.
* Split the dataset into train & test sets
* Tried three different models & chose MAE (Mean Absolute Error) as the evaluation metric, mainly because it is relatively easier to interpret & handles outliers decently.
* Models used: <br>
  1. **Linear Regression:** Baseline for the model <br>
  2. **Lasso Regression:** Because of the sparse data from many categorical variables <br>
  3. **Random Forest Regressor:** Again, with the sparsity associated with the data, I thought this would be a good fit <br>


# Productionization

Built a Flask API endpoint that was hosted on a local web server by following along with the TDS tutorial in the reference section above. The API endpoint takes in a request with a list of values from a job listing and returns an estimated salary.


