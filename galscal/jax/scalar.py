import jax
import jax.numpy as jnp

from typing import List, TypedDict

from .geo import GeoFeature


class ScalarFeature(TypedDict):
    o_m: jax.Array  # mass order
    o_x: jax.Array  # x order
    o_v: jax.Array  # v order
    ls: jax.Array  # `l` of constituent geom features
    ps: jax.Array  # `p` of ""
    ns: jax.Array  # `n` of ""
    Hs: jax.Array  # hermitivity of ""
    s_nums: jax.Array
    # ^^ scalar number of constituent SCALAR features
    # (0=order-0 geo feature, 1,2,3=eigenvalue of order-2 geo feature)
    value: jax.Array  # feature value


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
        "ls": jnp.concatenate([d["ls"] for d in ds]),
        "ps": jnp.concatenate([d["ps"] for d in ds]),
        "ns": jnp.concatenate([d["ns"] for d in ds]),
        "Hs": jnp.concatenate([d["Hs"] for d in ds]),
        "s_nums": jnp.concatenate([d["s_nums"] for d in ds]),
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
        "ls": jnp.stack([o0["l"], -1 * jnp.ones(n0, dtype=int)], axis=-1),
        "ps": jnp.stack([o0["p"], -1 * jnp.ones(n0, dtype=int)], axis=-1),
        "ns": jnp.stack([o0["n"], -1 * jnp.ones(n0, dtype=int)], axis=-1),
        "Hs": jnp.zeros((n0, 2), dtype=int),
        "s_nums": jnp.stack(
            [jnp.zeros(n0, dtype=int), -1 * jnp.ones(n0, dtype=int)], axis=-1
        ),
        "value": o0["value"],
    }

    n1 = len(o1["value"])
    idx1, idx2 = jnp.triu_indices(n1)
    o1_cs: ScalarFeature = {
        "o_m": 2 * jnp.ones(len(idx1)),
        "o_x": o1["l"][idx1] + o1["l"][idx2],
        "o_v": o1["p"][idx1] + o1["p"][idx2],
        "ls": jnp.stack([o1["l"][idx1], o1["l"][idx2]], axis=-1),
        "ps": jnp.stack([o1["p"][idx1], o1["p"][idx2]], axis=-1),
        "ns": jnp.stack([o1["n"][idx1], o1["n"][idx2]], axis=-1),
        "Hs": jnp.zeros((len(idx1), 2), dtype=int),
        "s_nums": -1 * jnp.ones((len(idx1), 2), dtype=int),
        "value": v_vector_contract(o1["value"][idx1], o1["value"][idx2]),
    }

    n2_S = len(o2_S["value"])
    idx1, idx2 = jnp.triu_indices(n2_S)
    o2_cs_S: ScalarFeature = {
        "o_m": 2 * jnp.ones(len(idx1)),
        "o_x": o2_S["l"][idx1] + o2_S["l"][idx2],
        "o_v": o2_S["p"][idx1] + o2_S["p"][idx2],
        "ls": jnp.stack([o2_S["l"][idx1], o2_S["l"][idx2]], axis=-1),
        "ps": jnp.stack([o2_S["p"][idx1], o2_S["p"][idx2]], axis=-1),
        "ns": jnp.stack([o2_S["n"][idx1], o2_S["n"][idx2]], axis=-1),
        "Hs": jnp.ones((len(idx1), 2), dtype=int),
        "s_nums": -1 * jnp.ones((len(idx1), 2), dtype=int),
        "value": v_tensor_contract(o2_S["value"][idx1], o2_S["value"][idx2]),
    }

    n2_A = len(o2_A["value"])
    idx1, idx2 = jnp.triu_indices(n2_A)
    o2_cs_A: ScalarFeature = {
        "o_m": 2 * jnp.ones(len(idx1)),
        "o_x": o2_A["l"][idx1] + o2_A["l"][idx2],
        "o_v": o2_A["p"][idx1] + o2_A["p"][idx2],
        "ls": jnp.stack([o2_A["l"][idx1], o2_A["l"][idx2]], axis=-1),
        "ps": jnp.stack([o2_A["p"][idx1], o2_A["p"][idx2]], axis=-1),
        "ns": jnp.stack([o2_A["n"][idx1], o2_A["n"][idx2]], axis=-1),
        "Hs": -1 * jnp.ones((len(idx1), 2), dtype=int),
        "s_nums": -1 * jnp.ones((len(idx1), 2), dtype=int),
        "value": v_tensor_contract(o2_A["value"][idx1], o2_A["value"][idx2]),
    }

    o2_evals: ScalarFeature = {
        "o_m": jnp.ones(n2_S * 3),
        "o_x": jnp.repeat(o2_S["l"], 3),
        "o_v": jnp.repeat(o2_S["p"], 3),
        "ls": jnp.stack(
            [jnp.repeat(o2_S["l"], 3), -1 * jnp.ones(n2_S * 3, dtype=int)], axis=-1
        ),
        "ps": jnp.stack(
            [jnp.repeat(o2_S["p"], 3), -1 * jnp.ones(n2_S * 3, dtype=int)], axis=-1
        ),
        "ns": jnp.stack(
            [jnp.repeat(o2_S["n"], 3), -1 * jnp.ones(n2_S * 3, dtype=int)], axis=-1
        ),
        "Hs": jnp.stack(
            [jnp.ones(n2_S * 3, dtype=int), jnp.zeros(n2_S * 3, dtype=int)], axis=-1
        ),
        "s_nums": jnp.stack(
            [jnp.tile(jnp.arange(3) + 1, n2_S), -1 * jnp.ones(n2_S * 3, dtype=int)],
            axis=-1,
        ),
        "value": v_tensor_eigenvalues(o2_S["value"]).flatten(),
    }

    mass_order_1 = _concat_feats([o0_cs, o2_evals])
    mass_order_2 = _concat_feats([o1_cs, o2_cs_S, o2_cs_A])

    n1 = len(mass_order_1["value"])
    idx1, idx2 = jnp.triu_indices(n1)
    s_pairs: ScalarFeature = {
        "o_m": 2 * jnp.ones(len(idx1)),
        "o_x": mass_order_1["o_x"][idx1] + mass_order_1["o_x"][idx2],
        "o_v": mass_order_1["o_v"][idx1] + mass_order_1["o_v"][idx2],
        "ls": jnp.stack(
            [mass_order_1["ls"][idx1, 0], mass_order_1["ls"][idx2, 0]], axis=-1
        ),
        "ps": jnp.stack(
            [mass_order_1["ps"][idx1, 0], mass_order_1["ps"][idx2, 0]], axis=-1
        ),
        "ns": jnp.stack(
            [mass_order_1["ns"][idx1, 0], mass_order_1["ns"][idx2, 0]], axis=-1
        ),
        "Hs": jnp.stack(
            [mass_order_1["Hs"][idx1, 0], mass_order_1["Hs"][idx2, 0]], axis=-1
        ),
        "s_nums": jnp.stack(
            [mass_order_1["s_nums"][idx1, 0], mass_order_1["s_nums"][idx2, 0]], axis=-1
        ),
        "value": mass_order_1["value"][idx1] * mass_order_1["value"][idx2],
    }

    s_features = _concat_feats([mass_order_1, mass_order_2, s_pairs])
    return s_features
