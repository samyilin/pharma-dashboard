import streamlit as st
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

(st.set_page_config(layout="wide"),)
data = pd.read_csv("discount_data.csv", header=0)

countries = st.multiselect("Choose Countries", list(data["Country "].unique()))
countries = list(data["Country "].unique()) if countries == [] else countries
brands = st.multiselect("Choose Brands", list(data["Brand "].unique()))
brands = list(data["Brand "].unique()) if brands == [] else brands
categories = st.multiselect("Choose Categories", list(data["Category"].unique()))
categories = list(data["Category"].unique()) if categories == [] else categories
channels = st.multiselect("Choose Channels", list(data["Channel"].unique()))
channels = list(data["Channel"].unique()) if channels == [] else channels

cm = sns.color_palette("vlag", as_cmap=True)
cn = sns.color_palette("light:r", as_cmap=True)

filtered_data = data[
    data["Country "].isin(countries)
    & data["Brand "].isin(brands)
    & data["Category"].isin(categories)
    & data["Channel"].isin(channels)
    & data["Year"].isin([2021, 2022, 2023])
]
totals_reference_brands = pd.pivot_table(
    data=data,
    index=["Year"],
    columns=["Brand "],
    values="Value (in Maple Dollars)",
    aggfunc="sum",
)
totals_reference_brands["Year"] = totals_reference_brands.index
totals_reference_brands = totals_reference_brands[
    totals_reference_brands.index.isin([2021, 2022, 2023])
].T
totals_reference_brands = totals_reference_brands.style.background_gradient(
    cmap=cn, axis=1
)
totals_reference_country = pd.pivot_table(
    data=data,
    index=["Year"],
    columns=["Country "],
    values="Value (in Maple Dollars)",
    aggfunc="sum",
)
totals_reference_country = totals_reference_country[
    totals_reference_country.index.isin([2021, 2022, 2023])
].T
totals_reference_country = totals_reference_country.style.background_gradient(
    cmap=cn, axis=1
)
totals_reference_both = pd.pivot_table(
    data=data,
    index=["Year"],
    columns=["Country ", "Brand "],
    values="Value (in Maple Dollars)",
    aggfunc="sum",
)
totals_reference_both = totals_reference_both[
    totals_reference_both.index.isin([2021, 2022, 2023])
].T
totals_reference_both = totals_reference_both.style.background_gradient(cmap=cn, axis=1)
col1, col2, col3 = st.columns(3)
col1.header("Annual Gross Sales by Brand")
col1.dataframe(totals_reference_brands)

col2.header("Annual Gross Sales by Country")
col2.dataframe(totals_reference_country)

col3.header("Annual Gross Sales")
col3.dataframe(totals_reference_both)

pivoted_data = pd.pivot_table(
    data=filtered_data,
    index=["Year"],
    columns="Detailed G2N Classification",
    values="Value (in Maple Dollars)",
    aggfunc="sum",
)
pivoted_data["net sales"] = pivoted_data["gross sales"] - pivoted_data.drop(
    "gross sales", axis=1
).sum(axis=1)
pivoted_data = pivoted_data.reindex(sorted(pivoted_data.columns, reverse=False), axis=1)
fig, ax = plt.subplots()
df_corr = pivoted_data.corr()
df_corr_viz = df_corr
df_corr_viz = df_corr_viz.style.background_gradient(cmap=cm, axis=1).hide(axis=1)
pivoted_data["total discount"] = pivoted_data.drop(
    ["gross sales", "net sales"], axis=1
).sum(axis=1)
pivoted_data["net sales growth"] = (
    pivoted_data["net sales"]
    .rolling(window=2)
    .apply(lambda x: (x.iloc[1] - x.iloc[0]) / x.iloc[0])
)
pivoted_data["gross sales growth"] = (
    pivoted_data["gross sales"]
    .rolling(window=2)
    .apply(lambda x: (x.iloc[1] - x.iloc[0]) / x.iloc[0])
)
pivoted_data["G2N Index"] = (
    (1 + pivoted_data["net sales growth"])
    / (1 + pivoted_data["gross sales growth"])
    * 100
)
g2n_calcs = pivoted_data[
    [
        "gross sales",
        "gross sales growth",
        "total discount",
        "net sales",
        "net sales growth",
        "G2N Index",
    ]
]
pivoted_data.drop(
    [
        "gross sales",
        "net sales",
        "net sales growth",
        "gross sales growth",
        "G2N Index",
        "total discount",
    ],
    axis=1,
    inplace=True,
)
styled_data = pivoted_data.style.background_gradient(cmap=cn, axis=1).hide(axis=1)
st.markdown("### G2N Index")
st.dataframe(g2n_calcs)
st.markdown("### Pivoted Annual Sum")
st.dataframe(styled_data)
st.markdown("### Correlation Matrix")
st.dataframe(df_corr_viz)
st.markdown("### Pivoted Yearly Percent of Total Discount")
percent_pivot = (
    pivoted_data.divide(
        pivoted_data.sum(axis=1),
        axis=0,
    )
    .style.background_gradient(cmap=cn, axis=1)
    .format("{:.4%}")
)
st.dataframe(percent_pivot)
