import os
import shutil
import tempfile
import unittest
import numpy as np
import xarray as xr
from mdt.tasks.data import save_data


class TestSaveData(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory for test stores
        self.test_dir = tempfile.mkdtemp()

        # Create a mock/dummy dataset
        self.ds = xr.Dataset(
            {
                "temperature": (["time", "y", "x"], np.random.rand(3, 4, 5)),
            },
            coords={
                "time": [1, 2, 3],
                "y": [10, 20, 30, 40],
                "x": [100, 110, 120, 130, 140],
            },
        )

    def tearDown(self):
        # Clean up temporary directory
        shutil.rmtree(self.test_dir)

    def test_save_zarr_success(self):
        url = os.path.join(self.test_dir, "test_store.zarr")

        # Save to Zarr
        save_data(name="test_zarr_task", data=self.ds, backend="zarr", url=url)

        # Verify Zarr store exists and matches original data
        self.assertTrue(os.path.exists(url))
        ds_loaded = xr.open_zarr(url)
        xr.testing.assert_equal(self.ds, ds_loaded)

    def test_save_icechunk_success(self):
        try:
            import icechunk
        except ImportError:
            self.skipTest("icechunk not installed")

        url = os.path.join(self.test_dir, "test_store_icechunk")

        # Save to Icechunk
        save_data(name="test_icechunk_task", data=self.ds, backend="icechunk", url=url)

        # Verify repository was committed to
        self.assertTrue(os.path.exists(url))

        storage = icechunk.local_filesystem_storage(url)
        repo = icechunk.Repository.open(storage)
        session = repo.readonly_session(branch="main")
        ds_loaded = xr.open_zarr(session.store, consolidated=False)

        xr.testing.assert_equal(self.ds["temperature"], ds_loaded["temperature"])

    def test_save_invalid_backend(self):
        url = os.path.join(self.test_dir, "test_store.zarr")
        with self.assertRaises(ValueError):
            save_data(name="test_invalid", data=self.ds, backend="invalid_backend", url=url)

    def test_save_invalid_type(self):
        url = os.path.join(self.test_dir, "test_store.zarr")
        with self.assertRaises(TypeError):
            save_data(name="test_invalid_type", data="not_a_dataset", backend="zarr", url=url)


if __name__ == "__main__":
    unittest.main()
