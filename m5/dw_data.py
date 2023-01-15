#%%

import numpy
import os
import pandas
import plotnine

data_dir = os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    "data"
)

calendar_pd = pandas.read_csv(os.path.join(data_dir, "clean", "calendar.csv"))
sell_prices_pd = pandas.read_csv(os.path.join(data_dir, "clean", "sell_prices.csv"))

#%%

calendar_with_sell_prices_pd = (
    calendar_pd
    .merge(sell_prices_pd, on = "wm_yr_wk", how = "inner")
)

#%%

for name in (
    "sales_test_evaluation",
    "sales_train_evaluation",
    "sales_test_validation",
    "sales_train_validation",
):
    sales_inpput_filename = os.path.join(data_dir, "clean", f"{name}.csv")
    sales_output_filename = os.path.join(data_dir, "dw", f"{name}.csv")

    print(f"Processing {sales_inpput_filename} to {sales_output_filename}")

    sales_with_prices_pd = (
        pandas.read_csv(sales_inpput_filename)
        .merge(
            calendar_with_sell_prices_pd,
            on = ["store_id", "item_id", "date"],
            how = "inner"
        )
    )

    (
        sales_with_prices_pd
        [[
            "date",
            "year",
            "month",
            "wday",
            "item_id",
            "dept_id",
            "cat_id",
            "sales",
            "sell_price",
            "snap_CA",
            "event_name_1",
            "event_type_1",
            "event_name_2",
            "event_type_2",
        ]]
        .to_csv(sales_output_filename, index=False)
    )

#%%

pd = (
    sales_with_prices_pd[lambda df : df["item_id"] == "HOBBIES_1_010"]
    .sort_values("date")
)

plotnine.qplot(numpy.arange(1913), pd.sales.to_numpy())

# %%
