import h5py
import illustris_python as il

from os.path import join
from glob import glob
from dataclasses import dataclass


@dataclass
class SnapshotReader:
    path: str
    snap_num: int

    def __post_init__(self):
        self.snap_files = sorted(
            glob(
                join(
                    self.path,
                    f"snapdir_{self.snap_num:03d}",
                    f"snap_{self.snap_num:03d}.*.hdf5",
                )
            ),
            key=lambda f: int(f.split(".")[-2]),
        )

        with h5py.File(self.snap_files[0], "r") as f:
            header = dict(f["Header"].attrs.items())
            self.m_dm = header["MassTable"][1]
            self.box_size = header["BoxSize"]

    def get_halo_positions(self, halo_num, shift=False):
        x = il.snapshot.loadHalo(
            self.path, self.snap_num, halo_num, "dm", "Coordinates"
        )
        if shift:
            raise NotImplementedError("shift not yet implemented")
        return x

    def get_halo_velocities(self, halo_num, shift=False):
        v = il.snapshot.loadHalo(self.path, self.snap_num, halo_num, "dm", "Velocities")
        if shift:
            raise NotImplementedError("shift not yet implemented")
        return v
