#%%

from collections import namedtuple
from importlib import reload
from bunny.sampler import warmup_and_sample
import equinox
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
        lambda x: (x.item_id, x.year, x.month),
        lambda x: x.wday
    )
)

domains = mapper.domains

#%%

class Model1(equinox.Module):
    beta: jax.numpy.ndarray
    log_phi: jax.numpy.ndarray
    item_year_month_z: jax.numpy.ndarray
    wday: jax.numpy.ndarray
    log_item_year_month_scale: jax.numpy.ndarray

    def __init__(self, key):
        keys = jax.random.split(key, 5)
        self.beta = jax.random.normal(keys[0], ())
        self.log_phi = jax.random.normal(keys[1], ())
        self.item_year_month_z = jax.random.normal(keys[2], (len(domains.item_year_month),))
        self.wday = jax.random.normal(keys[3], (len(domains.wday),))
        self.log_item_year_month_scale = jax.random.normal(keys[4], ())

    def __call__(self):
        item_year_month_scale = jax.numpy.exp(self.log_item_year_month_scale)
        item_year_month = self.item_year_month_z * item_year_month_scale
        phi = jax.numpy.exp(self.log_phi)

        log_sum_priors = (
            jax.numpy.sum(jax.scipy.stats.norm.logpdf(self.beta, 0.0, 1.0))
            + jax.numpy.sum(jax.scipy.stats.norm.logpdf(self.log_phi, 0.0, 1.0))
            + jax.numpy.sum(jax.scipy.stats.norm.logpdf(self.item_year_month_z, 0.0, 1.0))
            + jax.numpy.sum(jax.scipy.stats.norm.logpdf(self.wday, 0.0, 1.0))
            + jax.numpy.sum(jax.scipy.stats.norm.logpdf(self.log_item_year_month_scale, 0.0, 1.0))
        )

        def log_likelihood_function(constants, parameters : Parameters):
            sales, sell_price = constants
            item_year_month, wday = parameters
            mu = self.beta * sell_price + item_year_month + wday
            return negative_binomial_log(sales, mu, phi)

        log_likelihood = mapper(
            map_function = log_likelihood_function,
            parameters = Parameters(
                item_year_month=item_year_month,
                wday=self.wday
            )
        )

        return log_sum_priors + jax.numpy.sum(log_likelihood)
#%%

rng_key = jax.random.PRNGKey(42)
model = Model1(rng_key)

value_and_grad = jax.jit(jax.value_and_grad(lambda x: -x()))

#%%

flat_model, model_treedef = jax.tree_util.tree_flatten(model)
@jax.jit
def flat_value_and_grad(x: numpy.ndarray):
    value, unflattened_grad = value_and_grad(jax.tree_util.tree_unflatten(model_treedef, x))
    return value, jax.tree_util.tree_flatten(unflattened_grad)

#%%

value_and_grad(model)

#%%

flat_value_and_grad(flat_model)

#%%

rng = numpy.random.default_rng(0)

draws = warmup_and_sample(value_and_grad, rng, initial_draw)

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
