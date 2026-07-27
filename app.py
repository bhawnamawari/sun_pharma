import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="Sun Pharma Financial Dashboard", layout="wide")

COMPANY_NAME = "Sun Pharmaceutical Industries Ltd"

st.title(f"Financial Dashboard: {COMPANY_NAME}")
st.caption("Data source: Screener.in (Quarterly Results, consolidated figures in Rs Crores)")

df = pd.read_csv("company_financials_clean.csv")

# ---- KPI cards ----
latest = df.iloc[-1]
c1, c2, c3 = st.columns(3)
c1.metric("Sales (Latest Qtr)", f"₹{latest['Sales']:,.0f} Cr")
c2.metric("Net Profit (Latest Qtr)", f"₹{latest['Net_Profit']:,.0f} Cr")
c3.metric("OPM %", f"{latest['OPM_Percent']:.0f}%")

st.divider()

# ---- Charts ----
st.subheader("Sales & Net Profit Trend")
fig1, ax1 = plt.subplots(figsize=(10, 4))
ax1.plot(df["Period"], df["Sales"], marker="o", label="Sales (Rs Cr)")
ax1.plot(df["Period"], df["Net_Profit"], marker="o", label="Net Profit (Rs Cr)")
ax1.set_xlabel("Period")
ax1.set_ylabel("Rs Crores")
ax1.legend()
plt.xticks(rotation=45)
st.pyplot(fig1)

st.subheader("Operating Profit Margin (OPM %) by Quarter")
fig2, ax2 = plt.subplots(figsize=(10, 4))
ax2.bar(df["Period"], df["OPM_Percent"], color="teal")
ax2.set_xlabel("Period")
ax2.set_ylabel("OPM %")
plt.xticks(rotation=45)
st.pyplot(fig2)

st.subheader("Net Profit Distribution by Profit Trend")
plot_df = df.dropna(subset=["Profit_Trend"])
groups = [plot_df.loc[plot_df["Profit_Trend"] == g, "Net_Profit"] for g in ["Profit Grew", "Profit Declined"]]
fig3, ax3 = plt.subplots(figsize=(6, 4))
ax3.boxplot(groups, labels=["Profit Grew", "Profit Declined"])
ax3.set_ylabel("Net Profit (Rs Cr)")
st.pyplot(fig3)

st.divider()

# ---- Model comparison ----
st.subheader("Model Comparison: Predicting Profit_Trend")
st.caption("Logistic Regression vs Decision Tree (default settings). Small-sample caveat: only 12 labeled quarters.")

model_df = df.dropna(subset=["Profit_Trend", "Sales_Growth"]).copy()
X = model_df[["Sales_Growth", "OPM_Percent", "Interest", "Other_Income"]]
y = model_df["Profit_Trend"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y if y.nunique() > 1 else None
)

log_reg = LogisticRegression(max_iter=1000).fit(X_train, y_train)
dtree = DecisionTreeClassifier(random_state=42).fit(X_train, y_train)

comparison = pd.DataFrame({
    "Model": ["Logistic Regression", "Decision Tree (default)"],
    "Test Accuracy": [
        round(accuracy_score(y_test, log_reg.predict(X_test)), 2),
        round(accuracy_score(y_test, dtree.predict(X_test)), 2),
    ],
    "Test Set Size": [len(y_test), len(y_test)],
})
st.table(comparison)

st.divider()
with st.expander("View underlying cleaned data"):
    st.dataframe(df)
