import numpy as np
import molpy as mp
import pandas as pd
import pytest
from molpy.db.h5df import H5DFAdapter
from nesteddict import ArrayDict
from tempfile import NamedTemporaryFile


class TestReadH5DF:

    @pytest.fixture(scope="class")
    def h5df_instance(self, test_data_path):
        path = test_data_path/"h5df/litfsi.h5"
        with H5DFAdapter(path) as db:
            yield db


    def test_author(self, h5df_instance):
        assert h5df_instance.author == "Roy Kid"


    def test_get_frame(self, h5df_instance):
        frame = h5df_instance.get_frame(0)
        assert isinstance(frame, mp.Frame)
        assert "position" in frame["atoms"]


class TestModifyH5DF:

    @pytest.fixture(scope="class")
    def h5df_instance(self, test_data_path):
        path = test_data_path/"h5df/litfsi.h5"
        # copy to a temporary file

        tmp_file = NamedTemporaryFile(delete=False)
        with open(path, "rb") as fsrc:
            with open(tmp_file.name, "wb") as fdst:
                fdst.write(fsrc.read())

        with H5DFAdapter(tmp_file.name, "r+") as db:
            yield db

    def test_set_frame(self, h5df_instance):
        print(h5df_instance._file["particles/all"].keys())
        position = np.random.rand(16, 3)

        frame = mp.Frame({
            "atoms": {
                "position": position,
            },
        })
        h5df_instance.set_frame(0, frame, group_name="all")

        assert h5df_instance.get_frame(0)["atoms"]["position"].shape == (16, 3)

class TestWriteH5DF:

    @pytest.fixture(scope="class")
    def h5df_instance(self, test_data_path):

        tmp_file = NamedTemporaryFile(delete=False)
        with H5DFAdapter(tmp_file.name, "w") as db:
            yield db

    def test_add_frame(self, h5df_instance):
        position = np.random.rand(16, 3)
        frame = mp.Frame({
            "atoms": {
                "position": position,
            },
        })
        h5df_instance.add_frame(0, frame, group_name="all", molecule_name="test")

        new_frame = h5df_instance.get_frame(0, group_name="all", molecule_name="test")
        print(new_frame['atoms'])
        assert h5df_instance.get_frame(0, group_name="all", molecule_name="test")["atoms"]["position"].shape == (16, 3)