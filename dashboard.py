# dashboard.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re

sns.set_theme(style="whitegrid")
st.set_page_config(page_title="Vehicle Dashboard", layout="wide")

st.title("Vehicle Dataset Dashboard")

# ----------------------
# Load data
# ----------------------
st.sidebar.header("Load Data")
file = st.sidebar.file_uploader("Upload CSV", type=["csv"])
if file is not None:
    df = pd.read_csv(file)
else:
    df = pd.read_csv("car_dataset.csv")

# ----------------------
# Price cleaning (inline)
# ----------------------
raw = df['Ex-Showroom_Price'].astype(str)

s = raw.str.replace("₹", "", regex=False)
s = s.str.replace("Rs.", "", regex=False)
s = s.str.replace("Rs", "", regex=False)
s = s.str.replace("INR", "", regex=False)
s = s.str.replace("/-", "", regex=False)
s = s.str.replace(",", "", regex=False)
s = s.str.strip()
s_low = s.str.lower()

df['price_numeric'] = np.nan

mask_range = s.str.contains(r'[-–—]| to ')
if mask_range.any():
    parts = s[mask_range].str.split(r'[-–—]| to ', expand=True)
    left = parts[0].str.extract(r'([\d\.]+)')[0].astype(float)
    right = parts[1].str.extract(r'([\d\.]+)')[0].astype(float)
    df.loc[mask_range, 'price_numeric'] = (left + right) / 2

mask_cr = s_low.str.contains("cr") | s_low.str.contains("crore")
if mask_cr.any():
    df.loc[mask_cr, 'price_numeric'] = (
        s[mask_cr].str.extract(r'([\d\.]+)')[0].astype(float) * 10000000
    )

mask_lakh = (
    s_low.str.contains("lakh") |
    s_low.str.contains("lakhs") |
    s_low.str.contains("lac") |
    s.str.endswith(("L","l"))
)
if mask_lakh.any():
    df.loc[mask_lakh, 'price_numeric'] = (
        s[mask_lakh].str.extract(r'([\d\.]+)')[0].astype(float) * 100000
    )

mask_k = s_low.str.contains("k")
if mask_k.any():
    df.loc[mask_k, 'price_numeric'] = (
        s[mask_k].str.extract(r'([\d\.]+)')[0].astype(float) * 1000
    )

mask_m = s_low.str.contains("m")
if mask_m.any():
    df.loc[mask_m, 'price_numeric'] = (
        s[mask_m].str.extract(r'([\d\.]+)')[0].astype(float) * 1000000
    )

mask_plain = s.str.match(r'^\d+(\.\d+)?$')
if mask_plain.any():
    df.loc[mask_plain, 'price_numeric'] = s[mask_plain].astype(float)

fallback = df['price_numeric'].isna()
if fallback.any():
    num = s[fallback].str.extract(r'([\d\.]+)')[0]
    df.loc[fallback, 'price_numeric'] = pd.to_numeric(num, errors='coerce')

df['price_lakh'] = df['price_numeric'] / 100000

# clean numeric columns
for c in ['Displacement','Cylinders','Fuel_Tank_Capacity','Height','Length','Width','Seating_Capacity','Power.1','Torque.1']:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors='coerce')

for c in ['Make','Model','Variant','Fuel_Type','Body_Type','Type']:
    if c in df.columns:
        df[c] = df[c].astype(str)

# ----------------------
# Sidebar filters as dropdowns
# ----------------------
st.sidebar.header("Filters")

# Make dropdown (All option)
makes = sorted(df['Make'].unique())
makes_options = ["All"] + makes
sel_make = st.sidebar.selectbox("Make", makes_options, index=0)

# Body Type dropdown
bodies = sorted(df['Body_Type'].unique())
bodies_options = ["All"] + bodies
sel_body = st.sidebar.selectbox("Body Type", bodies_options, index=0)

# Fuel Type dropdown
fuels = sorted(df['Fuel_Type'].unique())
fuels_options = ["All"] + fuels
sel_fuel = st.sidebar.selectbox("Fuel Type", fuels_options, index=0)

# Price slider (Lakhs)
min_l = float(df['price_lakh'].min(skipna=True)) if df['price_lakh'].notna().any() else 0.0
max_l = float(df['price_lakh'].max(skipna=True)) if df['price_lakh'].notna().any() else min_l + 1.0
price_range = st.sidebar.slider("Price Range (Lakhs)", min_value=float(round(min_l,1)), max_value=float(round(max_l,1)), value=(float(round(min_l,1)), float(round(max_l,1))))

# ----------------------
# Apply filters
# ----------------------
mask = pd.Series(True, index=df.index)
if sel_make != "All":
    mask = mask & (df['Make'] == sel_make)
if sel_body != "All":
    mask = mask & (df['Body_Type'] == sel_body)
if sel_fuel != "All":
    mask = mask & (df['Fuel_Type'] == sel_fuel)

df_work = df[mask].copy()
df_work = df_work[(df_work['price_lakh'] >= price_range[0]) & (df_work['price_lakh'] <= price_range[1])]

# ----------------------
# KPIs
# ----------------------
col1, col2, col3, col4 = st.columns(4)
col1.metric("Rows", len(df_work))
col2.metric("Unique Makes", df_work['Make'].nunique())
col3.metric("Avg Price (L)", f"{df_work['price_lakh'].mean():.2f}")
col4.metric("Median Power", f"{df_work['Power.1'].median():.0f}" if df_work['Power.1'].notna().any() else "N/A")

st.markdown("---")

# ----------------------
# Body Type & Fuel Type counts (use barh to control palette)
# ----------------------
r1c1, r1c2 = st.columns(2)

with r1c1:
    counts = df_work['Body_Type'].value_counts()
    colors = sns.color_palette("pastel", len(counts))
    fig = plt.figure(figsize=(8,4))
    plt.barh(counts.index, counts.values, color=colors)
    plt.title("Vehicle Count by Body Type")
    plt.xlabel("Count")
    plt.tight_layout()
    st.pyplot(fig)
    plt.clf()

with r1c2:
    counts2 = df_work['Fuel_Type'].value_counts()
    colors2 = sns.color_palette("pastel", len(counts2))
    fig = plt.figure(figsize=(8,4))
    plt.barh(counts2.index, counts2.values, color=colors2)
    plt.title("Fuel Type Distribution")
    plt.xlabel("Count")
    plt.tight_layout()
    st.pyplot(fig)
    plt.clf()

st.markdown("---")

# ----------------------
# Price charts
# ----------------------
r2c1, r2c2 = st.columns(2)

with r2c1:
    fig = plt.figure(figsize=(8,4))
    sns.histplot(df_work['price_lakh'].dropna(), bins=30, kde=True, color="skyblue")
    plt.title("Price Distribution (Lakhs)")
    plt.xlabel("Price (Lakhs)")
    plt.tight_layout()
    st.pyplot(fig)
    plt.clf()

with r2c2:
    # boxplot with palette dict to avoid future warning
    cats = df_work['Body_Type'].unique()
    pal = dict(zip(cats, sns.color_palette("pastel", len(cats))))
    fig = plt.figure(figsize=(8,4))
    sns.boxplot(data=df_work, x='Body_Type', y='price_lakh', palette=pal)
    plt.xticks(rotation=45)
    plt.title("Price by Body Type (Lakhs)")
    plt.tight_layout()
    st.pyplot(fig)
    plt.clf()

st.markdown("---")

# ----------------------
# Scatterplots
# ----------------------
r3c1, r3c2 = st.columns(2)

with r3c1:
    fig = plt.figure(figsize=(8,4))
    # use Set2 palette only when number of categories > 0
    hue_vals = df_work['Body_Type']
    unique_h = hue_vals.unique()
    palette_map = dict(zip(unique_h, sns.color_palette("Set2", len(unique_h))))
    sns.scatterplot(data=df_work, x='Power.1', y='Torque.1', hue='Body_Type', palette=palette_map, legend='brief')
    plt.title("Power vs Torque")
    plt.tight_layout()
    st.pyplot(fig)
    plt.clf()

with r3c2:
    fig = plt.figure(figsize=(8,4))
    hue_vals2 = df_work['Fuel_Type']
    unique_h2 = hue_vals2.unique()
    palette_map2 = dict(zip(unique_h2, sns.color_palette("Set2", len(unique_h2))))
    sns.scatterplot(data=df_work, x='Power.1', y='price_lakh', hue='Fuel_Type', palette=palette_map2, legend='brief')
    plt.title("Price (Lakhs) vs Power")
    plt.tight_layout()
    st.pyplot(fig)
    plt.clf()

st.markdown("---")

# ----------------------
# Displacement vs Price with safe binning
# ----------------------
fig = plt.figure(figsize=(10,4))
hue_vals3 = df_work['Body_Type']
unique_h3 = hue_vals3.unique()
palette_map3 = dict(zip(unique_h3, sns.color_palette("Set2", len(unique_h3))))
sns.scatterplot(data=df_work, x='Displacement', y='price_lakh', hue='Body_Type', palette=palette_map3, alpha=0.7, legend='brief')

tmp = df_work.dropna(subset=['Displacement','price_lakh'])
if not tmp.empty:
    tmp = tmp.copy()
    tmp['bin'] = pd.cut(tmp['Displacement'], bins=8)
    avg = tmp.groupby('bin', observed=False)['price_lakh'].mean().reset_index()
    avg['mid'] = avg['bin'].apply(lambda x: x.mid)
    plt.plot(avg['mid'], avg['price_lakh'], color="black", linewidth=2)

plt.title("Displacement vs Price (Lakhs)")
plt.tight_layout()
st.pyplot(fig)
plt.clf()

st.markdown("---")

# ----------------------
# Heatmap
# ----------------------
num = df_work.select_dtypes(include=[np.number])
fig = plt.figure(figsize=(10,6))
if num.shape[1] > 1:
    sns.heatmap(num.corr(), annot=True, fmt=".2f", cmap="Blues", linewidths=0.5)
    plt.title("Correlation Heatmap")
st.pyplot(fig)
plt.clf()

st.markdown("---")

# ----------------------
# Data table + download
# ----------------------
st.subheader("Filtered Data")
st.dataframe(df_work.head(200))

csv = df_work.to_csv(index=False).encode('utf-8')
st.download_button("Download CSV", csv, "filtered_data.csv", "text/csv")

