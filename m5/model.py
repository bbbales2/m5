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

#pd = raw_pd[lambda df: df["item_id"] == "HOBBIES_1_008"]
pd = raw_pd[lambda df: df["cat_id"] == "HOBBIES"]#.sort_values("date")

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

ItemYearMonth = namedtuple(
    "ItemYearMonth",
    (
        "item_id",
        "year",
        "month"
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
        lambda x: ItemYearMonth(x.item_id, x.year, x.month),
        lambda x: x.wday
    )
)

domains : Parameters = mapper.domains

scanner = multilevel_tools.Scanner(
    domains.item_year_month,
    sort_by=lambda x: (x.year, x.month),
    group_by=lambda x: x.item_id,
    prepare_domains=(
        lambda x: (x.item_id, x.year, x.month),
        lambda x: x.item_id
    )
)

#%%

class Model1(equinox.Module):
    beta: jax.numpy.ndarray
    log_phi: jax.numpy.ndarray
    item_year_month_z: jax.numpy.ndarray
    wday: jax.numpy.ndarray
    log_item_year_month_scale: jax.numpy.ndarray
    item_year_month_init: jax.numpy.ndarray

    def __init__(self, key):
        keys = jax.random.split(key, 6)
        self.beta = jax.random.normal(keys[0], ())
        self.log_phi = jax.random.normal(keys[1], ())
        self.item_year_month_z = jax.random.normal(keys[2], (len(scanner.domains[0]),))
        self.wday = jax.random.normal(keys[3], (len(domains.wday),))
        self.log_item_year_month_scale = jax.random.normal(keys[4], (len(scanner.domains[1]),))
        self.item_year_month_init = jax.random.normal(keys[5], (len(scanner.groups),))

    def item_year_month(self):
        def rw(previous, _, parameters):
            innovation, log_scale = parameters
            scale = jax.numpy.exp(log_scale)
            return previous + innovation * scale

        return scanner(
            rw,
            initial_values=self.item_year_month_init,
            parameters=(
                self.item_year_month_z,
                self.log_item_year_month_scale
            )
        )

    def __call__(self):
        item_year_month = self.item_year_month()
        phi = jax.numpy.exp(self.log_phi)

        log_sum_priors = (
            jax.numpy.sum(jax.scipy.stats.norm.logpdf(self.beta, 0.0, 1.0))
            + jax.numpy.sum(jax.scipy.stats.norm.logpdf(self.log_phi, 0.0, 1.0))
            + jax.numpy.sum(jax.scipy.stats.norm.logpdf(self.item_year_month_z, 0.0, 1.0))
            + jax.numpy.sum(jax.scipy.stats.norm.logpdf(self.wday, 0.0, 1.0))
            + jax.numpy.sum(jax.scipy.stats.norm.logpdf(self.log_item_year_month_scale, 0.0, 1.0))
            + jax.numpy.sum(jax.scipy.stats.norm.logpdf(self.item_year_month_init, 0.0, 1.0))
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


@jax.jit
def flat_value_and_grad(x: numpy.ndarray):
    model = vector_to_pytree(x)
    value, unflattened_grad = jax.value_and_grad(lambda x: -x())(model)
    return value, multilevel_tools.pytree_to_vector(unflattened_grad)[0]

#%%

rng_key = jax.random.PRNGKey(42)
model = Model1(rng_key)
initial_draw, vector_to_pytree = multilevel_tools.pytree_to_vector(model)

#%%

rng = numpy.random.default_rng(0)
draws = warmup_and_sample(flat_value_and_grad, rng, initial_draw)

#%%
tree_draws = jax.tree_util.tree_map(
    lambda *xs: jax.numpy.array(xs),
    *[vector_to_pytree(x) for x in draws.reshape(-1, draws.shape[-1])]
)

item_year_month = jax.vmap(lambda x: x.item_year_month())(tree_draws)

(ql, m, qh) = numpy.quantile(item_year_month, [0.25, 0.5, 0.75], axis = 0)


item, year, month = zip(*domains.item_year_month)

df = pandas.DataFrame({
    "ql" : ql,
    "m" : m,
    "qh" : qh,
    "item" : item,
    "date" : [f"{year}-{month:02d}" for item, year, month in domains.item_year_month]
})[lambda df: df.item == "HOBBIES_1_010"]

#%%

(
    plotnine.ggplot(df)
    + plotnine.geom_point(plotnine.aes("date", "m", color = "item"))
    + plotnine.theme(
        axis_text_x=plotnine.element_text(rotation=90, hjust=1),
        legend_position='none'
    )
)


# %%

(
    plotnine.ggplot(pd[pd.item_id == "HOBBIES_1_010"])
    + plotnine.geom_point(plotnine.aes("date", "sales"))
)
# %%
