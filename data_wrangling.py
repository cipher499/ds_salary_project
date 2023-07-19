import pandas as pd

# read the dataset into a dataframe
df = pd.read_csv('Uncleaned_DS_jobs.csv')
df.head()
df.columns

# drop the rows in which the salary estimate is -1
df = df[df['Salary Estimate'] != '-1']

# create separate columns for hourly and employer provided salaries
df['hourly'] = df['Salary Estimate'].apply(lambda x: 1 if 'per hour' in x.lower() else 0)
df['employer_provided'] = df['Salary Estimate'].apply(lambda x: 1 if 'employer provided salary' in x.lower() else 0)

# parse numeric data out of salary
salary = df['Salary Estimate']
salary = df['Salary Estimate'].apply(lambda x: x.split('(')[0])
minus_Kd = salary.apply(lambda x: x.replace('K', '').replace('$', ''))
min_hr = minus_Kd.apply(lambda x: x.replace('per hour', '').replace('employer provided salary', ''))

#create new columns for min & max salary
df['min_salary'] = min_hr.apply(lambda x: int(x.split('-')[0]))
df['max_salary'] = min_hr.apply(lambda x: int(x.split('-')[1]))

#create a column for avg salary
df['avg_salary'] = (df.min_salary + df.max_salary)/2
df.head()

# clean the rows in Company name
df['company_txt'] = df.apply(lambda x: x['Company Name'] if x['Rating'] < 0 else x['Company Name'][:-3], axis=1)
df['company_txt'] = df['company_txt'].apply(lambda x: x.replace('\n', ' '))
df.head()

# create separate columns for the city and the state
df['city'] = df['Location'].apply(lambda x: x.split(',')[0])
df['state'] = df['Location'].apply(lambda x: x.split(',')[-1])
df.head()

# create a binary column to check for location = headquarters
df['same_state'] = df.apply(lambda x: 1 if x['Location'] == x['Headquarters'] else 0, axis=1)
df.head()

#Age of the company
df['age'] = df['Founded'].apply(lambda x: x if x<1 else 2023 - x)
df.head()

# parse the skills  out of Job description
df['python'] = df['Job Description'].apply(lambda x: 1 if 'python' in x.lower() else 0)
df['excel'] = df['Job Description'].apply(lambda x: 1 if 'excel' in x.lower() else 0)
df['aws'] = df['Job Description'].apply(lambda x: 1 if 'aws' in x.lower() else 0)
df['tableau'] = df['Job Description'].apply(lambda x: 1 if 'tableau' in x.lower() else 0)
df['spark'] = df['Job Description'].apply(lambda x: 1 if 'spark' in x.lower() else 0)
df.head()

# download the cleaned dataset
df.to_csv('salary_cleaned_set.csv', index=False)
