
# E-commerce Furniture Dataset Project
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer

# Load dataset
df = pd.read_csv('ecommerce_furniture_dataset_2024.csv')

# Preprocessing
df.dropna(subset=['price', 'tagText'], inplace=True)
df['price'] = df['price'].replace('[\$,]', '', regex=True).astype(float)
df.drop(['originalPrice'], axis=1, inplace=True)
df['tagText'] = df['tagText'].apply(lambda x: x if x in ['Free shipping', '+Shipping: $5.09'] else 'others')
le = LabelEncoder()
df['tagText'] = le.fit_transform(df['tagText'])

# TF-IDF on productTitle
tfidf = TfidfVectorizer(max_features=50)
productTitle_tfidf = tfidf.fit_transform(df['productTitle'])
productTitle_df = pd.DataFrame(productTitle_tfidf.toarray(), columns=tfidf.get_feature_names_out())

df_final = pd.concat([df.reset_index(drop=True), productTitle_df], axis=1).drop('productTitle', axis=1)

X = df_final.drop('sold', axis=1)
y = df_final['sold']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

lr = LinearRegression()
rf = RandomForestRegressor(n_estimators=100, random_state=42)

lr.fit(X_train, y_train)
rf.fit(X_train, y_train)

y_pred_lr = lr.predict(X_test)
y_pred_rf = rf.predict(X_test)

mse_lr = mean_squared_error(y_test, y_pred_lr)
r2_lr = r2_score(y_test, y_pred_lr)

mse_rf = mean_squared_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)

print(f"Linear Regression - MSE: {mse_lr}, R²: {r2_lr}")
print(f"Random Forest - MSE: {mse_rf}, R²: {r2_rf}")
