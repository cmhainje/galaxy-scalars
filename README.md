# galaxy-scalars

an extension of [kate storey-fisher](https://github.com/kstoreyf)'s [work on invariant scalar halo statistics](https://github.com/kstoreyf/equivariant-cosmology).

## authors

- **Connor Hainje**, NYU
- **Kate Storey-Fisher**, Stanford
- **David W Hogg**, NYU+

## installation

```bash
pip install git+https://github.com/cmhainje/galaxy-scalars
```

or for local dev

```bash
git clone --recurse-submodules https://github.com/cmhainje/galaxy-scalars.git
cd galaxy-scalars
poetry install
```

(`--recurse-submodules` is necessary to properly download and install the dependency [scsample](https://github.com/aust427/scsample), formerly known as illustris_sam)

(these steps require that you have Poetry installed)
