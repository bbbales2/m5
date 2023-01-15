#%%

import os 
import pandas
import pendulum

data_dir = os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    "data"
)

first_date = pendulum.parse("2011-01-29")
last_date = pendulum.parse("2016-06-19")

#%%


def sales_transform_long(raw_pd):
    clean_pd = pandas.melt(
        raw_pd,
        id_vars = ["item_id", "dept_id", "cat_id", "store_id", "state_id"],
        var_name = "day_raw",
        value_name = "sales"
    )

    day_offset = [
        int(day[2:]) - 1
        for day in clean_pd["day_raw"].to_list()
    ]

    day = [
        first_date.add(days = offset).to_date_string()
        for offset in day_offset
    ]

    return (
        clean_pd
        .assign(date = day)
        .drop("day_raw", axis = 1)
    )

#%%
for name in (
    "sales_train_evaluation",
    "sales_test_evaluation",
    "sales_train_validation",
    "sales_test_validation",
    ):
    input_file = os.path.join(data_dir, "raw", f"{name}.csv")
    output_file = os.path.join(data_dir, "clean", f"{name}.csv")
    pd = (
        pandas.read_csv(input_file)
        [lambda df: df["store_id"] == "CA_1"]
    )

    print(f"Processing {input_file} ... Saving to {output_file}")
    long_pd = sales_transform_long(pd)
    long_pd.to_csv(output_file, index=False)

#%%

calendar_input_file = os.path.join(data_dir, "raw", "calendar.csv")
calendar_output_file = os.path.join(data_dir, "clean", "calendar.csv")

pd = (
    pandas.read_csv(calendar_input_file)
    .fillna("")
    [[
        "date",
        "wm_yr_wk",
        "wday",
        "month",
        "year",
        "event_name_1",
        "event_type_1",
        "event_name_2",
        "event_type_2",
        "snap_CA"
    ]]
)

pd.to_csv(calendar_output_file, index = False)

#%%

sell_prices_input_file = os.path.join(data_dir, "raw", "sell_prices.csv")
sell_prices_output_file = os.path.join(data_dir, "clean", "sell_prices.csv")

pd = (
    pandas.read_csv(sell_prices_input_file)
    [lambda df: df["store_id"] == "CA_1"]
)

pd.to_csv(sell_prices_output_file, index = False)

#tmp = transform_long(sales_test_eval_pd)
#sales_test_eval_pd.to_feather("clean")
#print("hi")
# %%
