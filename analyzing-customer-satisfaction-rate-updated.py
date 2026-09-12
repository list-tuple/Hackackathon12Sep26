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
#
# ### Extended with three ML tasks:
# 4. Customer Satisfaction Prediction
# 5. Ticket Resolution Time Prediction
# 6. Customer Segmentation

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

# ## **Key Insights (Descriptive Analysis):**
# 1. **Average Resolution Time by Issue Type:**
# Analysis indicates that average resolution times differ slightly by ticket type, with billing inquiries resolved in approximately 7 minutes and refund requests taking around 8 minutes. This small variation raises the question of whether resolution time impacts customer satisfaction.
# 2. **Correlation Between Resolution Time and Customer Satisfaction:**
# The correlation analysis reveals a very weak, negative correlation (-0.00354) between resolution time and customer satisfaction. Although this result theoretically suggests that higher resolution times may decrease satisfaction, the correlation is so close to zero that resolution time likely has minimal impact on customer satisfaction.
# 3. **Issue Types Leading to Customer Dissatisfaction:**
# Among the ticket types, refund requests consistently received the lowest customer satisfaction ratings. Within this category, specific subjects like display issues, installation support, and product compatibility particularly stand out as areas with notable dissatisfaction.

# ---
# # Machine Learning Extension
#
# The descriptive analysis above shows resolution time alone barely explains satisfaction. The three tasks below move from "what happened" to "what can we predict / what groups exist", using ticket-level attributes as features.

# In[12]:


from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    mean_absolute_error, mean_squared_error, r2_score, silhouette_score
)
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# Candidate categorical / numeric features shared across the three tasks.
# These are all attributes known about a ticket independent of its outcome.
BASE_CATEGORICAL = ['Ticket Type', 'Ticket Subject', 'Ticket Priority', 'Ticket Channel', 'Customer Gender']
BASE_NUMERIC = ['Customer Age']

BASE_CATEGORICAL = [c for c in BASE_CATEGORICAL if c in customers.columns]
BASE_NUMERIC = [c for c in BASE_NUMERIC if c in customers.columns]

def encode_categoricals(df, categorical_cols):
    """Label-encode a copy of df's categorical columns, returning (encoded_df, encoders)."""
    df = df.copy()
    encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le
    return df, encoders


# ## Task 1: Customer Satisfaction Prediction
#
# Goal: predict the `Customer Satisfaction Rating` (1-5) a ticket will receive, using
# information available about the ticket. The rating only exists for tickets that were
# actually rated, so we first drop rows with a missing rating. `Resolution Time in Seconds`
# is included as a feature since it's known once a ticket is resolved and we want to test
# whether it, combined with the other attributes, helps predict satisfaction.

# In[13]:


csat_df = customers.dropna(subset=['Customer Satisfaction Rating']).copy()

csat_feature_cols = BASE_CATEGORICAL + BASE_NUMERIC + ['Resolution Time in Seconds']
csat_feature_cols = [c for c in csat_feature_cols if c in csat_df.columns]

csat_model_df = csat_df[csat_feature_cols + ['Customer Satisfaction Rating']].dropna()

csat_encoded, csat_encoders = encode_categoricals(
    csat_model_df, [c for c in BASE_CATEGORICAL if c in csat_model_df.columns]
)

X_csat = csat_encoded[csat_feature_cols]
y_csat = csat_encoded['Customer Satisfaction Rating'].astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X_csat, y_csat, test_size=0.2, random_state=42, stratify=y_csat
)

csat_clf = RandomForestClassifier(
    n_estimators=300, max_depth=8, random_state=42, class_weight='balanced'
)
csat_clf.fit(X_train, y_train)
y_pred_csat = csat_clf.predict(X_test)

print("Customer Satisfaction Prediction")
print("Accuracy:", round(accuracy_score(y_test, y_pred_csat), 4))
print(classification_report(y_test, y_pred_csat))


# In[14]:


# Confusion matrix
plt.style.use('dark_background')
plt.figure(figsize=(6, 5))
cm = confusion_matrix(y_test, y_pred_csat, labels=sorted(y_csat.unique()))
sns.heatmap(cm, annot=True, fmt='d', cmap='viridis',
            xticklabels=sorted(y_csat.unique()), yticklabels=sorted(y_csat.unique()))
plt.title('Confusion Matrix - Customer Satisfaction Rating')
plt.xlabel('Predicted Rating')
plt.ylabel('Actual Rating')
plt.show()


# In[15]:


# Feature importance
csat_importance = pd.Series(csat_clf.feature_importances_, index=X_csat.columns).sort_values(ascending=True)

plt.style.use('dark_background')
plt.figure(figsize=(8, 5))
csat_importance.plot(kind='barh', color=plt.cm.viridis(np.linspace(0.2, 0.9, len(csat_importance))))
plt.title('Feature Importance - Customer Satisfaction Prediction')
plt.xlabel('Importance')
plt.show()


# A near-chance accuracy here (consistent with the earlier ~0 correlation finding) would confirm
# that ticket metadata alone -- type, subject, priority, channel, resolution time -- carries very
# little signal about the rating a customer ultimately leaves. That itself is a useful business
# finding: satisfaction is likely driven by factors not captured in this dataset (e.g. agent
# behavior, communication quality, issue actually being fixed).

# ## Task 2: Ticket Resolution Time Prediction
#
# Goal: predict `Resolution Time in Seconds` from factors known when a ticket is opened
# (type, subject, priority, channel, customer demographics). `Customer Satisfaction Rating`
# is deliberately excluded as a feature since it is only known *after* resolution and would
# leak information about the outcome we're trying to predict.

# In[16]:


resolution_df = customers.dropna(subset=['Resolution Time in Seconds']).copy()

resolution_feature_cols = BASE_CATEGORICAL + BASE_NUMERIC
resolution_feature_cols = [c for c in resolution_feature_cols if c in resolution_df.columns]

resolution_model_df = resolution_df[resolution_feature_cols + ['Resolution Time in Seconds']].dropna()

resolution_encoded, resolution_encoders = encode_categoricals(
    resolution_model_df, [c for c in BASE_CATEGORICAL if c in resolution_model_df.columns]
)

X_res = resolution_encoded[resolution_feature_cols]
y_res = resolution_encoded['Resolution Time in Seconds']

X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
    X_res, y_res, test_size=0.2, random_state=42
)

res_reg = RandomForestRegressor(n_estimators=300, max_depth=8, random_state=42)
res_reg.fit(X_train_r, y_train_r)
y_pred_res = res_reg.predict(X_test_r)

mae = mean_absolute_error(y_test_r, y_pred_res)
rmse = mean_squared_error(y_test_r, y_pred_res, squared=False)
r2 = r2_score(y_test_r, y_pred_res)

print("Ticket Resolution Time Prediction")
print(f"MAE (seconds): {mae:,.1f}  (~{mae/60:.1f} minutes)")
print(f"RMSE (seconds): {rmse:,.1f}  (~{rmse/60:.1f} minutes)")
print(f"R^2: {r2:.4f}")


# In[17]:


# Predicted vs actual
plt.style.use('dark_background')
plt.figure(figsize=(6, 6))
plt.scatter(y_test_r / 60, y_pred_res / 60, alpha=0.4, color='#21918c')
lims = [0, max((y_test_r / 60).max(), (y_pred_res / 60).max())]
plt.plot(lims, lims, color='white', linestyle='--', linewidth=1)
plt.xlabel('Actual Resolution Time (minutes)')
plt.ylabel('Predicted Resolution Time (minutes)')
plt.title('Predicted vs Actual Resolution Time')
plt.show()


# In[18]:


# Feature importance
res_importance = pd.Series(res_reg.feature_importances_, index=X_res.columns).sort_values(ascending=True)

plt.style.use('dark_background')
plt.figure(figsize=(8, 5))
res_importance.plot(kind='barh', color=plt.cm.viridis(np.linspace(0.2, 0.9, len(res_importance))))
plt.title('Feature Importance - Resolution Time Prediction')
plt.xlabel('Importance')
plt.show()


# A low R^2 would echo the descriptive result from Question 1 (resolution times only vary
# slightly by ticket type): if these coarse categorical attributes don't explain much variance,
# resolution time is likely driven more by case-specific complexity or agent workload than by
# ticket metadata.

# ## Task 3: Customer Segmentation
#
# Goal: group tickets/customers into segments based on ticket type, priority, channel, and
# satisfaction level, to see whether natural clusters emerge (e.g. "high priority / low
# satisfaction" vs "low priority / high satisfaction" segments).

# In[19]:


segment_cols = BASE_CATEGORICAL + BASE_NUMERIC + ['Resolution Time in Seconds', 'Customer Satisfaction Rating']
segment_cols = [c for c in segment_cols if c in customers.columns]

segment_df = customers[segment_cols].dropna().copy()

segment_encoded, segment_encoders = encode_categoricals(
    segment_df, [c for c in BASE_CATEGORICAL if c in segment_df.columns]
)

scaler = StandardScaler()
X_segment_scaled = scaler.fit_transform(segment_encoded)


# In[20]:


# Elbow method to pick a reasonable number of clusters
inertias = []
sil_scores = []
k_range = range(2, 9)

for k in k_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_segment_scaled)
    inertias.append(km.inertia_)
    sil_scores.append(silhouette_score(X_segment_scaled, labels))

plt.style.use('dark_background')
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(list(k_range), inertias, marker='o', color='#21918c')
axes[0].set_title('Elbow Method (Inertia)')
axes[0].set_xlabel('Number of Clusters (k)')
axes[0].set_ylabel('Inertia')

axes[1].plot(list(k_range), sil_scores, marker='o', color='#fde725')
axes[1].set_title('Silhouette Score by k')
axes[1].set_xlabel('Number of Clusters (k)')
axes[1].set_ylabel('Silhouette Score')

plt.tight_layout()
plt.show()


# In[21]:


# Fit final KMeans model - adjust N_CLUSTERS after inspecting the elbow/silhouette plot above
N_CLUSTERS = 4

kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=10)
segment_df['Cluster'] = kmeans.fit_predict(X_segment_scaled)

# 2D projection for visualization
pca = PCA(n_components=2, random_state=42)
pca_coords = pca.fit_transform(X_segment_scaled)
segment_df['PCA1'] = pca_coords[:, 0]
segment_df['PCA2'] = pca_coords[:, 1]

plt.style.use('dark_background')
plt.figure(figsize=(8, 6))
sns.scatterplot(
    x='PCA1', y='PCA2', hue='Cluster', data=segment_df,
    palette='viridis', alpha=0.7
)
plt.title(f'Customer Segments (k={N_CLUSTERS}) - PCA Projection')
plt.show()


# In[22]:


# Profile each cluster: what makes each segment distinct?
profile_cols = [c for c in segment_cols]
cluster_profile = segment_df.groupby('Cluster')[profile_cols].mean(numeric_only=True)
cluster_sizes = segment_df['Cluster'].value_counts().sort_index()
cluster_profile['Cluster Size'] = cluster_sizes
cluster_profile


# In[23]:


# Decode the categorical columns back to labels for a human-readable profile
readable_profile = cluster_profile.copy()
for col, le in segment_encoders.items():
    if col in readable_profile.columns:
        rounded = readable_profile[col].round().clip(0, len(le.classes_) - 1).astype(int)
        readable_profile[col] = rounded.map(lambda i: le.classes_[i])

readable_profile


# The resulting clusters can be read as customer/ticket segments -- for example, one cluster
# might combine high-priority tickets on a specific channel with low satisfaction, flagging a
# segment worth prioritizing for service improvements, while another might represent low-priority,
# high-satisfaction "easy" tickets.

# ## **Updated Key Insights:**
# 1. **Descriptive findings (original analysis):** resolution time varies only slightly by
# ticket type and shows almost no correlation with satisfaction; refund requests -- especially
# around display, installation, and compatibility issues -- show the lowest satisfaction.
# 2. **Satisfaction prediction:** ticket metadata (type, subject, priority, channel, resolution
# time) has limited predictive power for satisfaction rating, reinforcing that satisfaction is
# likely driven by factors outside this dataset (e.g. agent quality, actual issue resolution).
# 3. **Resolution time prediction:** similarly, resolution time is only weakly explained by
# ticket metadata, suggesting it depends more on case complexity or staffing than on ticket type.
# 4. **Customer segmentation:** clustering surfaces natural groupings of tickets (e.g. by
# priority/channel/satisfaction combinations) that can guide targeted service interventions,
# even where a single predictive feature doesn't stand out on its own.

# ## **Conclusion:**
# The extended analysis reinforces the original finding: individual ticket attributes explain
# little of the variance in resolution time or satisfaction on their own. The added ML tasks
# turn that into something actionable -- flagging that better data (e.g. agent-level or
# conversation-level features) would likely be needed to meaningfully predict satisfaction or
# resolution time, while segmentation still offers a practical way to group tickets for targeted
# service improvements today.

# In[ ]:
