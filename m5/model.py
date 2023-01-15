#%%

from collections import namedtuple
from importlib import reload
import blackjax
import haiku
import jax
import multilevel_tools
import numpy
import os
import pandas
import plotnine
import timeit
from haiku_tools import negative_binomial_log

data_dir = os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    "data"
)

data_file = os.path.join(data_dir, "dw", "sales_train_validation.csv")

raw_pd = pandas.read_csv(data_file).fillna("")

#%%

pd = raw_pd[lambda df: df["item_id"] == "HOBBIES_1_008"]
#pd = raw_pd[lambda df: df["cat_id"] == "HOBBIES"]#.sort_values("date")

xs = [x for x in pd.itertuples(index = False)]

#%%

reload(multilevel_tools)

Parameters = namedtuple(
    "Parameters",
    (
        "beta",
        "phi",
        "item_year_month",
        "wday"
    )
)

# It might be better to do these mappers less object-oriented
#  By that I mean make the interface like:
#    mapper, domains = mulitlevel_tools.build_mapper(xs, prepare_constants=...)
mapper = multilevel_tools.Mapper(
    xs,
    prepare_constants=(
        lambda x: x.sales,
        lambda x: x.sell_price,
    ),
    prepare_domains=Parameters(
        # These first two are scalar parameters
        #   There's probably a way to do better here
        #   Actually maybe pass optional *args and **kwargs through the mapper?
        lambda x: 0,
        lambda x: 0,
        lambda x: (x.item_id, x.year, x.month),
        lambda x: x.wday
    )
)

domains : Parameters = mapper.domains

#%%
def model1():
    def log_likelihood_function(constants, p : Parameters):
        sales, sell_price = constants
        mu = p.beta * sell_price + p.item_year_month + p.wday
        return negative_binomial_log(sales, mu, p.phi)

    # The parameters are awkward here for a few reasons and could be improved
    #  1. Having to declare the parameter names in strings and then as variables
    #  2. Not having easy constraints
    #  3. The transformed parameters aren't exposed to the outside automatically
    beta = haiku.get_parameter("beta", shape = [len(domains.beta)], init=jax.numpy.zeros)
    log_phi = haiku.get_parameter("log_phi", shape = [len(domains.phi)], init=jax.numpy.zeros)
    item_year_month_z = haiku.get_parameter("item_year_month_z", shape = [len(domains.item_year_month)], init=jax.numpy.zeros)
    wday = haiku.get_parameter("wday", shape = [len(domains.wday)], init=jax.numpy.zeros)
    log_item_year_month_scale = haiku.get_parameter("log_item_year_month_scale", shape=[], init=jax.numpy.zeros)

    item_year_month_scale = jax.numpy.exp(log_item_year_month_scale)
    item_year_month = item_year_month_z * item_year_month_scale
    phi = jax.numpy.exp(log_phi)

    log_sum_priors = (
        jax.numpy.sum(jax.scipy.stats.norm.logpdf(beta, 0.0, 1.0))
        + jax.numpy.sum(jax.scipy.stats.norm.logpdf(log_phi, 0.0, 1.0))
        + jax.numpy.sum(jax.scipy.stats.norm.logpdf(item_year_month_z, 0.0, 1.0))
        + jax.numpy.sum(jax.scipy.stats.norm.logpdf(wday, 0.0, 1.0))
        + jax.numpy.sum(jax.scipy.stats.norm.logpdf(log_item_year_month_scale, 0.0, 1.0))
    )

    log_likelihood = mapper(
        map_function = log_likelihood_function,
        parameters = Parameters(
            beta=beta,
            phi=phi,
            item_year_month=item_year_month,
            wday=wday
        )
    )

    return log_sum_priors + jax.numpy.sum(log_likelihood)

transformed = haiku.without_apply_rng(haiku.transform(model1))
rng_key = jax.random.PRNGKey(42)
params = transformed.init(rng=rng_key)
mu_out = transformed.apply(params)

#%%

rng_key = jax.random.PRNGKey(0)
rng_key, init_key, warmup_key = jax.random.split(rng_key, 3)
warmup = blackjax.window_adaptation(blackjax.nuts, transformed.apply)

#%%
initial_states, _, tuned_params = warmup.run(warmup_key, params, 1000)
position = initial_states.position

#%%

from plotnine import *

(
    ggplot(pd) +
    geom_point(aes("date", "sales"))
)

#%%

iym_pd = pandas.DataFrame([
        (item, year, month, float(value))
        for (item, year, month), value in
        zip(domains.item_year_month, position["~"]["item_year_month_z"])
    ],
    columns = ["item", "year", "month", "value"]
)

#%%
(
    ggplot(
        iym_pd
        .assign(date = lambda df: df.year.astype(str) + "-" + df.month.astype(str).str.zfill(1))
    )
    + geom_point(aes("date", "value"))
)
