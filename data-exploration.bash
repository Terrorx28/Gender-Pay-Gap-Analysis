python3 -c "
import pandas as pd
df = pd.read_csv('/mnt/user-data/uploads/gender_pay_gap_india.csv')
print('Shape:', df.shape)
print()
print(df.describe())
print()
print('Gender counts:')
print(df['gender'].value_counts())
print()
print('Unique industries:', df['industry'].unique())
print('Unique domains:', df['domain'].unique())
print('Unique job_levels:', df['job_level'].unique())
print('Unique city_tier:', df['city_tier'].unique())
print('Unique company_size:', df['company_size'].unique())
print()
print('Avg salary by gender:')
print(df.groupby('gender')['monthly_salary_inr'].mean())
"
Output

Shape: (2000, 19)

       years_of_experience  negotiated_salary  ...          age  monthly_salary_inr
count          2000.000000        2000.000000  ...  2000.000000         2000.000000
mean              9.015750           0.507500  ...    31.561500        82421.450000
std               4.296909           0.500069  ...     4.751579        45015.155083
min               0.500000           0.000000  ...    22.000000        14000.000000
25%               6.100000           0.000000  ...    28.000000        48450.000000
50%               8.800000           1.000000  ...    31.000000        74285.000000
75%              11.925000           1.000000  ...    35.000000       106421.500000
max              26.500000           1.000000  ...    48.000000       283576.000000

[8 rows x 10 columns]

Gender counts:
gender
Male      1100
Female     900
Name: count, dtype: int64

Unique industries: <StringArray>
[         'Consulting',                'FMCG',          'Technology',
          'Healthcare',   'Finance & Banking',       'Manufacturing',
           'Education',               'Media',              'Retail',
 'NGO / Social Sector']
Length: 10, dtype: str
Unique domains: <StringArray>
['Customer Support',          'Finance',      'Engineering',
        'Marketing',            'Sales',               'HR',
       'Operations',            'Legal', 'Data & Analytics',
            'Admin']
Length: 10, dtype: str
Unique job_levels: <StringArray>
['Senior', 'Mid', 'Junior', 'Manager', 'Director']
Length: 5, dtype: str
Unique city_tier: <StringArray>
['Tier 3', 'Tier 2', 'Tier 1']
Length: 3, dtype: str
Unique company_size: <StringArray>
['Large (500-2000)', 'MNC (2000+)', 'Startup (<50)', 'SME (50-500)']
Length: 4, dtype: str

Avg salary by gender:
gender
Female    67010.257778
Male      95030.607273
Name: monthly_salary_inr, dtype: float64