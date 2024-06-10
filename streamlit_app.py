import streamlit as st
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import plotly.express as px

(st.set_page_config(layout="wide"),)

data = pd.read_csv("discount_data.csv", header=0)
data["halves"] = 0
data.loc[
    data["Row Labels"].isin([2021003, 2021004, 2021005, 2021006, 2021007, 2021008]),
    "halves",
] = 1
data.loc[
    data["Row Labels"].isin([2021009, 2021010, 2021011, 2021012, 2022001, 2022002]),
    "halves",
] = 1
data.loc[
    data["Row Labels"].isin([2022003, 2022004, 2022005, 2022006, 2022007, 2022008]),
    "halves",
] = 3
data.loc[
    data["Row Labels"].isin([2022009, 2022010, 2022011, 2022012, 2023001, 2023002]),
    "halves",
] = 4
data.loc[
    data["Row Labels"].isin([2023003, 2023004, 2023005, 2023006, 2023007, 2023008]),
    "halves",
] = 5
data.loc[
    data["Row Labels"].isin([2023009, 2023010, 2023011, 2023012, 2024001, 2024002]),
    "halves",
] = 6
tab1, tab2, tab3, tab4 = st.tabs(
    ["Overall Comp", "Breakdown", "ROI Comp", "ROI Benchmark"]
)
cm = sns.color_palette("vlag", as_cmap=True)
cn = sns.color_palette("light:r", as_cmap=True)

with tab1:
    countries_tab1 = st.multiselect(
        "Choose countries", list(data["Country "].unique()), key=1
    )
    countries_tab1 = (
        list(data["Country "].unique()) if countries_tab1 == [] else countries_tab1
    )
    brands_tab1 = st.multiselect("Choose brands", list(data["Brand "].unique()), key=2)
    brands_tab1 = list(data["Brand "].unique()) if brands_tab1 == [] else brands_tab1
    categories_tab1 = st.multiselect(
        "Choose categories", list(data["Category"].unique()), key=3
    )
    categories_tab1 = (
        list(data["Category"].unique()) if categories_tab1 == [] else categories_tab1
    )
    channels_tab1 = st.multiselect(
        "Choose channels", list(data["Channel"].unique()), key=4
    )
    channels_tab1 = (
        list(data["Channel"].unique()) if channels_tab1 == [] else channels_tab1
    )
    years_tab1 = st.multiselect("Choose years", list(data["Year"].unique()), key=5)
    years_tab1 = [2021, 2022, 2023] if years_tab1 == [] else years_tab1

    breakdown_tab1 = st.selectbox(
        "Choose pivot dimension",
        ["High Level G2N Classification", "Detailed G2N Classification"],
        key=6,
    )
    # pivot_index_tab1 = st.selectbox(
    #     "Choose pivot index dimension", ["Year", "halves"], key=7
    # )

    filtered_data_tab1 = data[
        data["Country "].isin(countries_tab1)
        & data["Brand "].isin(brands_tab1)
        & data["Category"].isin(categories_tab1)
        & data["Channel"].isin(channels_tab1)
        & data["Year"].isin(years_tab1)
    ]
    overall_comp_country = pd.pivot_table(
        data=filtered_data_tab1,
        index=["Country "],
        columns=breakdown_tab1,
        values="Value (in Maple Dollars)",
        aggfunc="sum",
    )
    overall_comp_country["total G2N"] = overall_comp_country.drop(
        ["gross sales"], axis=1
    ).sum(axis=1)
    overall_comp_country["net sales"] = (
        overall_comp_country["gross sales"] - overall_comp_country["total G2N"]
    )
    overall_comp_country["G2N ratio"] = (
        overall_comp_country["total G2N"].divide(
            overall_comp_country["gross sales"], axis=0
        )
        * 100
    )
    st.dataframe(overall_comp_country)

    overall_comp_brand = pd.pivot_table(
        data=filtered_data_tab1,
        index=["Brand "],
        columns=breakdown_tab1,
        values="Value (in Maple Dollars)",
        aggfunc="sum",
    )
    overall_comp_brand["total G2N"] = overall_comp_brand.drop(
        ["gross sales"], axis=1
    ).sum(axis=1)
    overall_comp_brand["net sales"] = (
        overall_comp_brand["gross sales"] - overall_comp_brand["total G2N"]
    )
    overall_comp_brand["G2N ratio"] = (
        overall_comp_brand["total G2N"].divide(
            overall_comp_brand["gross sales"], axis=0
        )
        * 100
    )
    st.dataframe(overall_comp_brand)
with tab2:
    countries_tab2 = st.multiselect(
        "Choose countries", list(data["Country "].unique()), key=11
    )
    countries_tab2 = (
        list(data["Country "].unique()) if countries_tab2 == [] else countries_tab2
    )
    brands_tab2 = st.multiselect("Choose brands", list(data["Brand "].unique()), key=12)
    brands_tab2 = list(data["Brand "].unique()) if brands_tab2 == [] else brands_tab2
    categories_tab2 = st.multiselect(
        "Choose categories",
        list(
            data["Category"].unique(),
        ),
        key=13,
    )
    categories_tab2 = (
        list(data["Category"].unique()) if categories_tab2 == [] else categories_tab2
    )
    channels_tab2 = st.multiselect(
        "Choose channels", list(data["Channel"].unique()), key=14
    )
    channels_tab2 = (
        list(data["Channel"].unique()) if channels_tab2 == [] else channels_tab2
    )
    years_tab2 = st.multiselect("Choose years", list(data["Year"].unique()), key=15)
    years_tab2 = [2021, 2022, 2023] if years_tab2 == [] else years_tab2

    breakdown_tab2 = st.selectbox(
        "Choose pivot dimension",
        ["High Level G2N Classification", "Detailed G2N Classification"],
        key=16,
    )
    pivot_index_tab2 = st.selectbox(
        "Choose pivot index dimension", ["Year", "halves"], key=17
    )

    filtered_data_tab2 = data[
        data["Country "].isin(countries_tab2)
        & data["Brand "].isin(brands_tab2)
        & data["Category"].isin(categories_tab2)
        & data["Channel"].isin(channels_tab2)
        & data["Year"].isin(years_tab2)
    ]
    totals_reference_brands_tab2 = pd.pivot_table(
        data=data,
        index=pivot_index_tab2,
        columns=["Brand "],
        values="Value (in Maple Dollars)",
        aggfunc="sum",
    )
    # totals_reference_brands_tab2["Year"] = totals_reference_brands_tab2.index
    totals_reference_brands_tab2 = totals_reference_brands_tab2[
        totals_reference_brands_tab2.index.isin([2021, 2022, 2023])
    ].T
    totals_reference_brands_tab2 = (
        totals_reference_brands_tab2.style.background_gradient(cmap=cn, axis=1)
    )
    totals_reference_country = pd.pivot_table(
        data=data,
        index=pivot_index_tab2,
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
        index=pivot_index_tab2,
        columns=["Country ", "Brand "],
        values="Value (in Maple Dollars)",
        aggfunc="sum",
    )
    totals_reference_both = totals_reference_both[
        totals_reference_both.index.isin([2021, 2022, 2023])
    ].T
    totals_reference_both = totals_reference_both.style.background_gradient(
        cmap=cn, axis=1
    )
    col1, col2, col3 = st.columns(3)
    col1.header("Annual Gross Sales by Brand")
    col1.dataframe(totals_reference_brands_tab2)

    col2.header("Annual Gross Sales by Country")
    col2.dataframe(totals_reference_country)

    col3.header("Annual Gross Sales")
    col3.dataframe(totals_reference_both)

    pivoted_data = pd.pivot_table(
        data=filtered_data_tab2,
        index=pivot_index_tab2,
        columns=breakdown_tab2,
        values="Value (in Maple Dollars)",
        aggfunc="sum",
    )
    pivoted_data["net sales"] = pivoted_data["gross sales"] - pivoted_data.drop(
        "gross sales", axis=1
    ).sum(axis=1)
    pivoted_data = pivoted_data.reindex(
        sorted(pivoted_data.columns, reverse=False), axis=1
    )
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
with tab3:
    countries_tab3 = st.multiselect(
        "Choose countries", list(data["Country "].unique()), key=31
    )
    countries_tab3 = (
        list(data["Country "].unique()) if countries_tab3 == [] else countries_tab3
    )
    brands_tab3 = st.multiselect("Choose brands", list(data["Brand "].unique()), key=32)
    brands_tab3 = list(data["Brand "].unique()) if brands_tab3 == [] else brands_tab3
    categories_tab3 = st.multiselect(
        "Choose categories", list(data["Category"].unique()), key=33
    )
    categories_tab3 = (
        list(data["Category"].unique()) if categories_tab3 == [] else categories_tab3
    )
    channels_tab3 = st.multiselect(
        "Choose channels", list(data["Channel"].unique()), key=34
    )
    channels_tab3 = (
        list(data["Channel"].unique()) if channels_tab3 == [] else channels_tab3
    )
    years_tab3 = st.multiselect("Choose years", list(data["Year"].unique()), key=35)
    years_tab3 = [2021, 2022, 2023] if years_tab3 == [] else years_tab3

    breakdown_tab3 = st.selectbox(
        "Choose pivot dimension",
        ["High Level G2N Classification", "Detailed G2N Classification"],
        key=36,
    )
    breakdown_tab3 = (
        "High Level G2n Classification" if breakdown_tab3 is None else breakdown_tab3
    )
    pivot_index_tab3 = st.selectbox(
        "Choose pivot index dimension", ["Year", "halves"], key=37
    )
    g2n_list = (
        list(data["High Level G2N Classification"].unique())
        if breakdown_tab3 is None
        else list(data[breakdown_tab3].unique())
    )
    g2n_list = [x for x in g2n_list if x != "gross sales"]
    g2n_breakdown_tab3 = st.multiselect(
        "Choose G2N levels",
        g2n_list,
        key=38,
    )
    g2n_breakdown_tab3 = (
        list(data[breakdown_tab3].unique())
        if g2n_breakdown_tab3 == []
        else g2n_breakdown_tab3
    )
    g2n_breakdown_tab3 = g2n_breakdown_tab3 + ["gross sales"]

    filtered_data_tab3 = data[
        data["Country "].isin(countries_tab3)
        & data["Brand "].isin(brands_tab3)
        & data["Category"].isin(categories_tab3)
        & data["Channel"].isin(channels_tab3)
        & data["Year"].isin(years_tab3)
        & data[breakdown_tab3].isin(g2n_breakdown_tab3)
    ]
    roi_comp = pd.pivot_table(
        data=filtered_data_tab3,
        index=pivot_index_tab3,
        columns=breakdown_tab3,
        values="Value (in Maple Dollars)",
        aggfunc="sum",
    )
    roi_comp["total G2N"] = roi_comp.drop(["gross sales"], axis=1).sum(axis=1)
    roi_comp["net sales"] = roi_comp["gross sales"] - roi_comp["total G2N"]
    roi_comp["net sales growth"] = (
        roi_comp["net sales"].rolling(window=2).apply(lambda x: (x.iloc[1] - x.iloc[0]))
    )
    roi_comp["roi"] = roi_comp["net sales growth"] / roi_comp["total G2N"]
    st.dataframe(roi_comp)

with tab4:
    breakdown_tab4 = st.selectbox(
        "Choose pivot dimension",
        ["High Level G2N Classification", "Detailed G2N Classification"],
        key=41,
    )
    # countries_tab4 = st.multiselect(
    #     "Choose countries", list(data["Country "].unique()), key=31
    # )
    # countries_tab4 = (
    #     list(data["Country "].unique()) if countries_tab4 == [] else countries_tab4
    # )
    # brands_tab4 = st.multiselect("Choose brands", list(data["Brand "].unique()), key=32)
    # brands_tab4 = list(data["Brand "].unique()) if brands_tab4 == [] else brands_tab4
    # categories_tab4 = st.multiselect(
    #     "Choose categories", list(data["Category"].unique()), key=33
    # )
    # categories_tab4 = (
    #     list(data["Category"].unique()) if categories_tab4 == [] else categories_tab4
    # )
    # channels_tab4 = st.multiselect(
    #     "Choose channels", list(data["Channel"].unique()), key=34
    # )
    # channels_tab4 = (
    #     list(data["Channel"].unique()) if channels_tab4 == [] else channels_tab4
    # )
    #
    # data.groupby(["Country ", "Year ", "High Level G2N Classification"])["Value (in Maple Dollars)"].sum()
    data_filtered = data[data["Year"].isin([2021, 2022, 2023])]

    roi_pivot_main = pd.pivot_table(
        data=data,
        index=["Country "],
        columns=[breakdown_tab4],
        values="Value (in Maple Dollars)",
        aggfunc="sum",
    )
    roi_pivot_main["total G2N"] = roi_pivot_main.drop(["gross sales"], axis=1).sum(
        axis=1
    )
    roi_pivot_main["net sales"] = (
        roi_pivot_main["gross sales"] - roi_pivot_main["total G2N"]
    )
    roi_pivot_main = roi_pivot_main.sort_values("net sales").reset_index()
    st.dataframe(roi_pivot_main)
    mean = roi_pivot_main["net sales"].mean()
    std = roi_pivot_main["net sales"].std()
    df = px.data.tips()
    fig = px.histogram(roi_pivot_main, x="net sales")
    st.plotly_chart(fig)
    st.markdown(
        "Average Total Net Sales for all countries is "
        + str(mean)
        + " with standard deviation "
        + str(std)
    )

    st.markdown(
        "This includes "
        + str(
            roi_pivot_main[
                (roi_pivot_main["net sales"] <= (mean + std))
                & (roi_pivot_main["net sales"] >= (mean - std))
            ].shape[0]
        )
        + " countries"
    )
    st.dataframe(
        roi_pivot_main[
            (roi_pivot_main["net sales"] <= (mean + std))
            & (roi_pivot_main["net sales"] >= (mean - std))
        ]
    )
    median_index = roi_pivot_main.index[
        roi_pivot_main["net sales"] == roi_pivot_main["net sales"].median()
    ][0]
    st.markdown("### Median 7 performers")
    st.dataframe(roi_pivot_main.iloc[(median_index - 3) : (median_index + 4)])
    country_list = roi_pivot_main.iloc[(median_index - 3) : (median_index + 4)][
        "Country "
    ]
    roi_calc_main = {}
    for _, value in country_list.items():
        roi_calc_main[value] = pd.pivot_table(
            data=data[data["Country "].isin([value])],
            index=["Year"],
            columns=[breakdown_tab4],
            values="Value (in Maple Dollars)",
            aggfunc="sum",
        )
    for dataframe in roi_calc_main:
        roi_calc_main[dataframe]["net sales"] = roi_calc_main[dataframe][
            "gross sales"
        ] - roi_calc_main[dataframe].drop(["gross sales"], axis=1).sum(axis=1)
        roi_calc_main[dataframe]["net sales growth"] = (
            roi_calc_main[dataframe]["net sales"]
            .rolling(window=2)
            .apply(lambda x: (x.iloc[1] - x.iloc[0]))
        )
        for column in roi_calc_main[dataframe].columns:
            if column not in ["net sales", "gross sales", "net sales growth"]:
                roi_calc_main[dataframe][column + " ROI"] = (
                    roi_calc_main[dataframe]["net sales growth"]
                    / roi_calc_main[dataframe][column]
                )
        st.markdown("### ROI for country " + dataframe)
        st.dataframe(roi_calc_main[dataframe])
    roi_calc_average = {}
    roi_column_names = set()
    for dataframe in roi_calc_main:
        roi_calc_average[dataframe] = roi_calc_main[dataframe].mean(axis=0)
        roi_calc_average[dataframe] = (
            roi_calc_average[dataframe].to_frame().rename(columns={0: dataframe})
        )
        roi_column_names.update(roi_calc_average[dataframe].index)
        st.markdown("### Average ROI for country " + dataframe)
        st.dataframe(roi_calc_average[dataframe])
    roi_column_names = [x for x in roi_column_names if "ROI" in x]
    roi_calc_average = [roi_calc_average[dataframe] for dataframe in roi_calc_average]
    roi_calc_final = pd.concat(
        [df for df in roi_calc_average],
        axis=1,
        keys=country_list,
    )
    roi_calc_final.columns = roi_calc_final.columns.droplevel(1)
    st.markdown("### Average ROIs of Select Countries")
    st.dataframe(roi_calc_final)
    st.markdown("### Average ROIs")
    st.dataframe(roi_calc_final.mean(axis=1))
