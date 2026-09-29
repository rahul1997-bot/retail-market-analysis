importnumpyas np
importpandasas pd
fromsklearn.linear_model importLinearRegression

# 1. Retail Sales Sample Data
np.random.seed(42)
dates = pd.date_range(start="2023-01-01", periods=180, freq="D")
categories = ["Electronics", "Clothing", "Home & Kitchen", "Groceries"]

data = {
    "Date": dates,
    "Category": np.random.choice(categories, size=180),
    "Sales_Amount": np.random.uniform(500, 5000, size=180).round(2),
}

df = pd.DataFrame(data)

# 2. Daily Aggregation & Trend Forecasting
daily_sales = (
    df.groupby("Date")["Sales_Amount"].sum().reset_index().sort_values("Date")
)
daily_sales["Day_Index"] = np.arange(len(daily_sales))

X = daily_sales[["Day_Index"]]
y = daily_sales["Sales_Amount"]

model = LinearRegression()
model.fit(X, y)

print("Retail Market Analysis & Forecasting Model Trained Successfully.")
