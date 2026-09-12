#!/usr/bin/env python
# coding: utf-8

# # Analyzing Customer Satisfaction Rating

# ### Business Task:
# Identify factors influence customer satisfaction and service time response
# 
# ### Question need to be answered:
# 1. What is the Average resolution time for different types of issues ?
# 2. How does resolution time correlate with customer satisfaction Rate ?
# 3. What is the most type of issue lead to the most customer dissatisfaction?

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

customers = pd.read_csv("/kaggle/input/customer-support-ticket-dataset/customer_support_tickets.csv")
customers.head(3)


# ### First Data Exploration

# In[2]:


customers.isnull().sum()


# In[3]:


customers.dtypes


# In[4]:


customers.nunique()


# In[5]:


customers.duplicated().sum()


# ### Second Data Cleaning and Manipulation

# In[6]:


# Convert a column to datetime
customers['Date of Purchase'] = pd.to_datetime(customers['Date of Purchase'], format='%Y-%m-%d')
customers['First Response Time'] = pd.to_datetime(customers['First Response Time'], format='%Y-%m-%d %H:%M:%S')
customers['Time to Resolution'] = pd.to_datetime(customers['Time to Resolution'], format='%Y-%m-%d %H:%M:%S')

customers.dtypes


# In[7]:


# Creating New Column Resolution Time
customers['Resolution Time'] = customers['Time to Resolution'] - customers['First Response Time']
customers['Resolution Time'] = customers['Resolution Time'].apply(lambda x: x if x >= pd.Timedelta(0) else pd.NaT)
customers['Resolution Time'].nunique()


# In[8]:


# What is the Average resolution time for different types of issues

Average_Resolution_Time = customers.groupby('Ticket Type').agg({'Resolution Time': 'mean'})
Average_Resolution_Time['Resolution Time Formatted'] = Average_Resolution_Time['Resolution Time'].apply(
    lambda x: f"{int(x.total_seconds() // 3600):02}:{int((x.total_seconds() % 3600) // 60):02}:{int(x.total_seconds() % 60):02}" if pd.notnull(x) else None
)
Average_Resolution_Time.drop('Resolution Time', axis=1, inplace=True)
Average_Resolution_Time = Average_Resolution_Time.sort_values(by='Resolution Time Formatted', ascending=True)

# Reset index to make 'Ticket Type' a column
Average_Resolution_Time = Average_Resolution_Time.reset_index()

# Convert 'Resolution Time Formatted' to numerical time in hours for plotting
def convert_to_hours(time_str):
    h, m, s = map(int, time_str.split(':'))
    return h + m / 60 + s / 3600

Average_Resolution_Time['Resolution Time (hours)'] = Average_Resolution_Time['Resolution Time Formatted'].apply(convert_to_hours)

plt.style.use('dark_background')
plt.figure(figsize=(12, 6))  # Adjusted width to 12 for more space
sns.barplot(
    x='Resolution Time (hours)',
    y='Ticket Type',
    data=Average_Resolution_Time,
    palette="viridis"
)
plt.title('Average Resolution Time by Ticket Type')
plt.xlabel('')
plt.ylabel('')

for index, row in Average_Resolution_Time.iterrows():
    plt.text(row['Resolution Time (hours)'] - 0.2, index, row['Resolution Time Formatted'], color='white', va="center", ha="right")

plt.show()


# It’s noticeable that the average resolution time varies slightly across ticket types, with billing inquiries taking around 7 minutes and refund requests about 8 minutes. This raises the question: could this small difference in average resolution time impact customer satisfaction rates?

# In[9]:


#how does resolution time correlate with customer satisfaction rate

# Convert 'Resolution Time' to delta time then to total seconds
customers['Customer Satisfaction Rating'] = customers['Customer Satisfaction Rating'].astype(float)
customers['Resolution Time'] = pd.to_timedelta(customers['Resolution Time'], errors='coerce')
customers['Resolution Time in Seconds'] = customers['Resolution Time'].dt.total_seconds()
correlation = customers['Resolution Time in Seconds'].corr(customers['Customer Satisfaction Rating'])
print(correlation)


# Upon examining the correlation between average resolution time and customer satisfaction rate, I found a correlation of -0.00354. While the negative sign suggests that an increase in resolution time might lead to a slight decrease in customer satisfaction—an intuitively logical relationship—the value is so close to zero that it indicates a nearly negligible effect. This suggests that average resolution time does not significantly influence customer satisfaction rates.

# In[10]:


# What is the most type of issue lead to customer dissatisfaction
result = customers.groupby('Ticket Type').agg({'Customer Satisfaction Rating':'mean'})

result_sorted = result.sort_values(by='Customer Satisfaction Rating', ascending=True)

result_sorted = result_sorted.reset_index()

plt.style.use('dark_background')
plt.figure(figsize=(12, 6))
sns.barplot(
    x='Customer Satisfaction Rating',
    y='Ticket Type',
    data=result_sorted,
    palette="viridis"
)
plt.title('Average Customer Satisfaction Rating by Ticket Type')
plt.xlabel('')
plt.ylabel('')

for index, row in result_sorted.iterrows():
    plt.text(row['Customer Satisfaction Rating'] - 0.05, index, f"{row['Customer Satisfaction Rating']:.2f}", 
             color='white', va="center", ha="right")

plt.show()


# 
# Refund requests have the lowest customer satisfaction ratings, so it's important to identify which specific subjects within refund requests contribute to these low ratings.

# In[11]:


# Filter data to only include 'refund request' ticket type
refund_requests = customers[customers['Ticket Type'] == 'Refund request']
refund_subjects = refund_requests.groupby('Ticket Subject').agg({'Customer Satisfaction Rating': 'mean'})
refund_subjects_sorted = refund_subjects.sort_values(by='Customer Satisfaction Rating', ascending=True)

plt.style.use('dark_background')
plt.figure(figsize=(10, 8))
ax = sns.barplot(x=refund_subjects_sorted['Customer Satisfaction Rating'], y=refund_subjects_sorted.index, palette='viridis')

for p in ax.patches:
    ax.text(p.get_width() -0.2 , p.get_y() + p.get_height() / 2,
            f'{p.get_width():.2f}', fontsize=10, color='white', ha='left', va='center')

plt.title("Average Customer Satisfaction Rating by Ticket Subject for 'Refund request' ticket", color='white')
plt.xlabel('', color='white')
plt.ylabel('', color='white')

plt.show()


# 
# As shown, customer satisfaction ratings drop significantly for subjects related to display issues, installation support, and product compatibility.

# ## **Key Insights:**
# 1. **Average Resolution Time by Issue Type:**
# Analysis indicates that average resolution times differ slightly by ticket type, with billing inquiries resolved in approximately 7 minutes and refund requests taking around 8 minutes. This small variation raises the question of whether resolution time impacts customer satisfaction.
# 2. **Correlation Between Resolution Time and Customer Satisfaction:**
# The correlation analysis reveals a very weak, negative correlation (-0.00354) between resolution time and customer satisfaction. Although this result theoretically suggests that higher resolution times may decrease satisfaction, the correlation is so close to zero that resolution time likely has minimal impact on customer satisfaction.
# 3. **Issue Types Leading to Customer Dissatisfaction:**
# Among the ticket types, refund requests consistently received the lowest customer satisfaction ratings. Within this category, specific subjects like display issues, installation support, and product compatibility particularly stand out as areas with notable dissatisfaction.
# 

# ## **Conclusion:**
# The analysis suggests that while resolution time varies by ticket type, it does not significantly affect customer satisfaction. Instead, dissatisfaction appears more closely tied to specific issues within refund requests, particularly display, installation, and compatibility problems. This insight highlights areas where improving service quality, particularly for refund-related issues, could have a stronger impact on enhancing overall customer satisfaction than focusing on reducing resolution times alone.

# In[ ]:




