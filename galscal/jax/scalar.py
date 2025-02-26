import jax
import jax.numpy as jnp

from typing import List, TypedDict

from .geo import GeoFeature


class ScalarFeature(TypedDict):
    o_m: jax.Array
    o_x: jax.Array
    o_v: jax.Array
    value: jax.Array


def vector_contract(v1: jax.Array, v2: jax.Array):
    return jnp.einsum("j,j->", v1, v2)


def tensor_contract(t1: jax.Array, t2: jax.Array):
    return jnp.einsum("jk,jk->", t1, t2)


def tensor_eigenvalues(t: jax.Array):
    return jnp.linalg.eigvalsh(t)[::-1]


v_vector_contract = jax.jit(jax.vmap(vector_contract))
v_tensor_contract = jax.jit(jax.vmap(tensor_contract))
v_tensor_eigenvalues = jax.jit(jax.vmap(tensor_eigenvalues))


def _concat_feats(ds: List[ScalarFeature]) -> ScalarFeature:
    return {
        "o_m": jnp.concatenate([d["o_m"] for d in ds]),
        "o_x": jnp.concatenate([d["o_x"] for d in ds]),
        "o_v": jnp.concatenate([d["o_v"] for d in ds]),
        "value": jnp.concatenate([d["value"] for d in ds]),
    }


@jax.jit
def compute_scalar_features(
    o0: GeoFeature,
    o1: GeoFeature,
    o2_S: GeoFeature,
    o2_A: GeoFeature,
):
    n0 = len(o0["value"])
    o0_cs: ScalarFeature = {
        "o_m": jnp.ones(n0),
        "o_x": jnp.zeros(n0),
        "o_v": jnp.zeros(n0),
        "value": o0["value"],
    }

    n1 = len(o1["value"])
    idx1, idx2 = jnp.triu_indices(n1)
    o1_cs: ScalarFeature = {
        "o_m": 2 * jnp.ones(n1),
        "o_x": o1["l"][idx1] + o1["l"][idx2],
        "o_v": o1["p"][idx1] + o1["p"][idx2],
        "value": v_vector_contract(o1["value"][idx1], o1["value"][idx2]),
    }

    n2_S = len(o2_S["value"])
    idx1, idx2 = jnp.triu_indices(n2_S)
    o2_cs_S: ScalarFeature = {
        "o_m": 2 * jnp.ones(n2_S),
        "o_x": o2_S["l"][idx1] + o2_S["l"][idx2],
        "o_v": o2_S["p"][idx1] + o2_S["p"][idx2],
        "value": v_tensor_contract(o2_S["value"][idx1], o2_S["value"][idx2]),
    }

    n2_A = len(o2_A["value"])
    idx1, idx2 = jnp.triu_indices(n2_A)
    o2_cs_A: ScalarFeature = {
        "o_m": 2 * jnp.ones(n2_A),
        "o_x": o2_A["l"][idx1] + o2_A["l"][idx2],
        "o_v": o2_A["p"][idx1] + o2_A["p"][idx2],
        "value": v_tensor_contract(o2_A["value"][idx1], o2_A["value"][idx2]),
    }

    o2_evals: ScalarFeature = {
        "o_m": jnp.ones(n2_S),
        "o_x": jnp.repeat(o2_S["l"], 3),
        "o_v": jnp.repeat(o2_S["p"], 3),
        "value": v_tensor_eigenvalues(o2_S["value"]).flatten(),
    }

    mass_order_1 = _concat_feats([o0_cs, o2_evals])
    mass_order_2 = _concat_feats([o1_cs, o2_cs_S, o2_cs_A])

    n1 = len(mass_order_1["value"])
    idx1, idx2 = jnp.triu_indices(n1)
    s_pairs: ScalarFeature = {
        "o_m": 2 * jnp.ones(n1),
        "o_x": mass_order_1["o_x"][idx1] + mass_order_1["o_x"][idx2],
        "o_v": mass_order_1["o_v"][idx1] + mass_order_1["o_v"][idx2],
        "value": mass_order_1["value"][idx1] * mass_order_1["value"][idx2],
    }

    s_features = _concat_feats([mass_order_1, mass_order_2, s_pairs])
    return s_features
