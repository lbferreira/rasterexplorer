from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import rasterio
from affine import Affine
from numpy.typing import NDArray
from numpy.testing import assert_array_equal

from rasterexplorer.raster_data import RasterData


DATA_PATH = Path(__file__).parent / "data"


@pytest.fixture
def raster_data() -> RasterData:
    input_raster = DATA_PATH / "dataset1_ndvi.tif"
    band = 1
    with rasterio.open(input_raster) as dataset:
        raster_data = RasterData.from_rasterio_dataset(dataset, band=band)
    return raster_data


@dataclass
class RefDataset:
    data_no_mask: NDArray[Any]
    data_masked: np.ma.core.MaskedArray
    nodata: Any
    crs: Any
    transform: Affine
    bounds: rasterio.coords.BoundingBox


@pytest.fixture
def ref_dataset() -> RefDataset:
    input_raster = DATA_PATH / "dataset1_ndvi.tif"
    band = 1
    raster = {}
    with rasterio.open(input_raster) as dataset:
        raster["data_no_mask"] = dataset.read(band, masked=False)
        raster["data_masked"] = dataset.read(band, masked=True)
        raster["nodata"] = dataset.nodata
        raster["crs"] = dataset.crs
        raster["transform"] = dataset.transform
        raster["bounds"] = dataset.bounds

    return RefDataset(**raster)


@pytest.mark.parametrize("masked", [True, False])
def test_raster_data_creation(ref_dataset: RefDataset, masked: bool) -> None:
    data = ref_dataset.data_masked if masked else ref_dataset.data_no_mask
    raster_data = RasterData(
        data=data,
        nodata=ref_dataset.nodata,
        crs=ref_dataset.crs,
        transform=ref_dataset.transform,
    )
    # Testing properties
    assert_array_equal(raster_data.data, ref_dataset.data_masked)
    assert raster_data.nodata == ref_dataset.nodata
    assert raster_data.crs == ref_dataset.crs
    assert raster_data.transform == ref_dataset.transform
    assert raster_data.bounds == ref_dataset.bounds


def test_from_rasterio_dataset() -> None:
    input_raster = DATA_PATH / "dataset1_ndvi.tif"
    band = 1
    with rasterio.open(input_raster) as dataset:
        data_masked = dataset.read(band, masked=True)
        nodata = dataset.nodata
        crs = dataset.crs
        transform = dataset.transform
        bounds = dataset.bounds
        raster_data = RasterData.from_rasterio_dataset(dataset, band=band)

    # Testing properties
    assert_array_equal(raster_data.data, data_masked)
    assert raster_data.nodata == nodata
    assert raster_data.crs == crs
    assert raster_data.transform == transform
    assert raster_data.bounds == bounds


def test_to_crs(raster_data: RasterData) -> None:
    target_crs = 'EPSG:4326'
    ref_data_path = DATA_PATH / "dataset1_ndvi_wgs84.tif"
    band = 1
    with rasterio.open(ref_data_path) as dataset:
        expected_raster_data = RasterData.from_rasterio_dataset(dataset, band=band)
    reprojected_raster = raster_data.to_crs(target_crs)
    assert_array_equal(reprojected_raster.data, expected_raster_data.data)
    assert reprojected_raster.nodata == expected_raster_data.nodata
    assert reprojected_raster.crs == expected_raster_data.crs
    assert reprojected_raster.transform == expected_raster_data.transform
    assert reprojected_raster.bounds == expected_raster_data.bounds


def test__validade_data_shape(raster_data: RasterData) -> None:
    expected_one_dimension = np.ones((5))
    expected_two_dimensions = np.ones((5, 5))
    with pytest.raises(ValueError):
        raster_data._validade_data_shape(expected_one_dimension)
    assert raster_data._validade_data_shape(expected_two_dimensions) is None


def test__prepare_input_data(raster_data: RasterData, ref_dataset: RefDataset) -> None:
    prepared_from_masked = raster_data._prepare_input_data(
        ref_dataset.data_masked, nodata=ref_dataset.nodata
    )
    prepared_from_no_mask = raster_data._prepare_input_data(
        ref_dataset.data_no_mask, nodata=ref_dataset.nodata
    )
    assert_array_equal(prepared_from_masked, ref_dataset.data_masked)
    assert_array_equal(prepared_from_no_mask, ref_dataset.data_masked)
    with pytest.raises(TypeError):
        raster_data._prepare_input_data(None, None)


def test__compute_bounds(raster_data: RasterData, ref_dataset: RefDataset) -> None:
    computed_bounds = raster_data._compute_bounds()
    assert computed_bounds == ref_dataset.bounds


def test__to_masked_array(raster_data: RasterData, ref_dataset: RefDataset) -> None:
    nodata = ref_dataset.nodata
    values = ref_dataset.data_no_mask
    expected_masked_array = ref_dataset.data_masked
    masked_array = raster_data._to_masked_array(values=values, nodata=nodata)
    assert_array_equal(masked_array, expected_masked_array)
