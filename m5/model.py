#%%

from importlib import reload
import jax
import multilevel_tools
import numpy
import os
import pandas
import plotnine
import timeit

data_dir = os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    "data"
)

data_file = os.path.join(data_dir, "dw", "sales_train_validation.csv")

raw_pd = pandas.read_csv(data_file).fillna("")

#%%

pd = raw_pd[lambda df: df["item_id"] == "HOBBIES_1_008"].sort_values("item_id")
#pd = raw_pd[lambda df: df["cat_id"] == "HOBBIES"]#.sort_values("date")

xs = [x for x in pd.itertuples(index = False)]

#%%

reload(multilevel_tools)

mapper = multilevel_tools.Mapper(
    xs,
    prepare_constants=(
        lambda x: x.sales,
        lambda x: x.sell_price,
    ),
    prepare_subscripts=(
        lambda x: x.year,
        lambda x: x.month,
        lambda x: x.wday
    )
)
# %%

def map_function(constants, parameters):
    sales, sell_price = constants
    year, month, wday = parameters

    return jax.scipy.stats.norm.logpdf(sales * sell_price, year + month + wday, 1.5)

year, month, wday = mapper.domains

pyear = jax.numpy.zeros(len(year))
pmonth = jax.numpy.zeros(len(year))
pwday = jax.numpy.zeros(len(year))

jitted_value_and_grad_map = jax.jit(jax.value_and_grad(lambda x : jax.numpy.sum(mapper(map_function, parameters = x))))
jitted_value_and_grad_map((pyear, pmonth, pwday))

#%%

timer_map = timeit.Timer(lambda: jitted_value_and_grad_map((pyear, pmonth, pwday)))
number, total_time = timer_map.autorange()
print(f"{number / total_time:0.4f} gradients per second map")

#%%
reload(multilevel_tools)

scanner = multilevel_tools.Scanner(
    xs,
    group_by=lambda x: x.item_id,
    sort_by=lambda x: x.date,
    prepare_constants=(
        lambda x: x.sales,
        lambda x: x.sell_price,
    ),
    prepare_subscripts=(
        lambda x: x.year,
        lambda x: x.month,
        lambda x: x.wday
    ),
    num_partitions=1
)


#%%

def scan_function(prev, constants, parameters):
    sales, sell_price = constants
    year, month, wday = parameters

    return prev + jax.scipy.stats.norm.logpdf(sales * sell_price, year + month + wday, 1.5)


year, month, wday = scanner.domains

pyear = jax.numpy.zeros(len(year))
pmonth = jax.numpy.zeros(len(year))
pwday = jax.numpy.zeros(len(year))


initial_values = jax.numpy.zeros(len(scanner.groups))
jitted_value_and_grad_scan = jax.jit(jax.value_and_grad(lambda x : jax.numpy.sum(scanner(scan_function, initial_values=x[0], parameters = x[1]))))
jitted_value_and_grad_scan((initial_values, (pyear, pmonth, pwday)))

#%%
print(jax.make_jaxpr(jitted_value_and_grad_scan)((initial_values, (pyear, pmonth, pwday))))

#%%
timer_scan = timeit.Timer(lambda: jitted_value_and_grad_scan((initial_values, (pyear, pmonth, pwday))))
number, total_time = timer_scan.autorange()
print(f"{number / total_time:0.4f} gradients per second scan")

# %%
