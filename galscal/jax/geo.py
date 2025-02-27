import jax
import jax.numpy as jnp

from collections import defaultdict
from functools import partial
from typing import NamedTuple, TypedDict


class GeoFeature(TypedDict):
    l: jax.Array
    p: jax.Array
    n: jax.Array
    value: jax.Array


class GeoFeatureOrders(NamedTuple):
    o0: GeoFeature
    o1: GeoFeature
    o2_S: GeoFeature
    o2_A: GeoFeature


v_outer = jax.jit(jax.vmap(jnp.multiply.outer))


@partial(jax.jit, static_argnames=["l"])
def v_outer_power(x: jax.Array, l: int) -> jax.Array:
    y = jnp.ones((x.shape[0],), dtype=x.dtype)
    for _ in range(l):
        y = v_outer(y, x)
    return y


@partial(jax.jit, static_argnames=["l", "p"])
def summand(X: jax.Array, V: jax.Array, l: int, p: int):
    return v_outer(v_outer_power(X, l), v_outer_power(V, p))


@jax.jit
def apply_window(w: jax.Array, x: jax.Array):
    return jnp.einsum("i,i...->...", w, x)


v_apply_window = jax.vmap(apply_window, in_axes=(0, None), out_axes=0)


def _concat_dict(d) -> GeoFeature:
    return {
        "l": jnp.concatenate(d["l"]),
        "p": jnp.concatenate(d["p"]),
        "n": jnp.concatenate(d["n"]),
        "value": jnp.concatenate(d["value"]),
    }


@jax.jit
def compute_geometric_features(M, X, V, r_edges) -> GeoFeatureOrders:
    radii = jnp.linalg.norm(X, axis=-1)
    bins = jnp.digitize(radii, jnp.array(r_edges)) - 1
    n_w = len(r_edges) - 1
    windows = jnp.stack(
        [M * (bins == i).astype(jnp.float32) for i in range(n_w)], axis=0
    )

    # order 0
    o0 = defaultdict(list)
    for l, p in [(0, 0)]:
        s = summand(X, V, l, p)
        o0["l"].append(jnp.ones(n_w) * l)
        o0["p"].append(jnp.ones(n_w) * p)
        o0["n"].append(jnp.arange(n_w))
        o0["value"].append(v_apply_window(windows, s))
    o0 = _concat_dict(o0)

    # order 1
    o1 = defaultdict(list)
    for l, p in [(1, 0), (0, 1)]:
        s = summand(X, V, l, p)
        o1["l"].append(jnp.ones(n_w) * l)
        o1["p"].append(jnp.ones(n_w) * p)
        o1["n"].append(jnp.arange(n_w))
        o1["value"].append(v_apply_window(windows, s))
    o1 = _concat_dict(o1)

    # order 2
    o2_S = defaultdict(list)
    o2_A = defaultdict(list)
    for l, p in [(2, 0), (0, 2)]:
        s = summand(X, V, l, p)
        o2_S["l"].append(jnp.ones(n_w) * l)
        o2_S["p"].append(jnp.ones(n_w) * p)
        o2_S["n"].append(jnp.arange(n_w))
        o2_S["value"].append(v_apply_window(windows, s))

    for l, p in [(1, 1)]:
        s = summand(X, V, l, p)

        s_S = 0.5 * (s + jnp.matrix_transpose(s))
        o2_S["l"].append(jnp.ones(n_w) * l)
        o2_S["p"].append(jnp.ones(n_w) * p)
        o2_S["n"].append(jnp.arange(n_w))
        o2_S["value"].append(v_apply_window(windows, s_S))

        s_A = 0.5 * (s - jnp.matrix_transpose(s))
        o2_A["l"].append(jnp.ones(n_w) * l)
        o2_A["p"].append(jnp.ones(n_w) * p)
        o2_A["n"].append(jnp.arange(n_w))
        o2_A["value"].append(v_apply_window(windows, s_A))

    o2_S = _concat_dict(o2_S)
    o2_A = _concat_dict(o2_A)

    return GeoFeatureOrders(o0, o1, o2_S, o2_A)


def geo_name(l: int, p: int, n: int, sym: bool | None = None) -> str:
    components = [str(l), str(p), str(n)]
    if sym is not None:
        components.append(str(sym))
    return "(" + ",".join(components) + ")"
