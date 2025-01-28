# Task 1: Exploratory Data Analysis (EDA) and Business Insights

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Read the CSV files
transactions_df = pd.read_csv('Transactions.csv')
products_df = pd.read_csv('Products.csv')
customers_df = pd.read_csv('Customers.csv')

# Convert date columns to datetime
transactions_df['TransactionDate'] = pd.to_datetime(transactions_df['TransactionDate'])
customers_df['SignupDate'] = pd.to_datetime(customers_df['SignupDate'])

# Merge dataframes for analysis
sales_analysis = transactions_df.merge(products_df, on='ProductID')
full_analysis = sales_analysis.merge(customers_df, on='CustomerID')

# 1. Sales by Category
category_sales = sales_analysis.groupby('Category')['TotalValue'].sum().sort_values(ascending=False)
plt.figure(figsize=(10, 6))
category_sales.plot(kind='bar')
plt.title('Total Sales by Product Category')
plt.xlabel('Category')
plt.ylabel('Total Sales (USD)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 2. Regional Analysis
regional_sales = full_analysis.groupby('Region')['TotalValue'].sum().sort_values(ascending=False)
plt.figure(figsize=(10, 6))
regional_sales.plot(kind='bar')
plt.title('Total Sales by Region')
plt.xlabel('Region')
plt.ylabel('Total Sales (USD)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 3. Monthly Sales Trend
monthly_sales = full_analysis.groupby(full_analysis['TransactionDate'].dt.to_period('M'))['TotalValue'].sum()
plt.figure(figsize=(12, 6))
monthly_sales.plot(kind='line')
plt.title('Monthly Sales Trend')
plt.xlabel('Month')
plt.ylabel('Total Sales (USD)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 4. Customer Purchase Frequency
customer_frequency = transactions_df['CustomerID'].value_counts()
plt.figure(figsize=(10, 6))
plt.hist(customer_frequency, bins=30)
plt.title('Distribution of Customer Purchase Frequency')
plt.xlabel('Number of Purchases')
plt.ylabel('Number of Customers')
plt.tight_layout()
plt.show()

# 5. Average Order Value by Category
avg_order_value = sales_analysis.groupby('Category')['TotalValue'].mean().sort_values(ascending=False)
plt.figure(figsize=(10, 6))
avg_order_value.plot(kind='bar')
plt.title('Average Order Value by Category')
plt.xlabel('Category')
plt.ylabel('Average Order Value (USD)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Print key metrics
print("\nKey Metrics:")
print(f"Total Revenue: ${full_analysis['TotalValue'].sum():,.2f}")
print(f"Average Order Value: ${full_analysis['TotalValue'].mean():,.2f}")
print(f"Total Number of Transactions: {len(transactions_df)}")
print(f"Total Number of Unique Customers: {len(transactions_df['CustomerID'].unique())}")
print(f"Most Popular Category: {category_sales.index[0]}")