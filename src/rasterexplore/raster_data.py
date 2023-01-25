from __future__ import annotations

from typing import Any

import numpy as np
import rasterio
from affine import Affine
from numpy.typing import NDArray
from pyproj import CRS
from rasterio.warp import (
    Resampling,
    array_bounds,
    calculate_default_transform,
    reproject,
)


class RasterData:
    def __init__(
        self, data: np.ma.core.MaskedArray, nodata: Any, crs: Any, transform: Affine
    ) -> None:
        """Stores raster data, including metadata.

        Args:
            data (np.ma.core.MaskedArray): 2D array with raster data. It must be a numpy masked array.
            nodata (NDArray[Any]): the value used to represent no data values in the input data.
            crs (Any): a CRS (Coordinate Reference System) of the raster. It can be anything accepted to create a pyproj.CRS object.
            transform (Affine): the raster transform.
        """
        # Validade data
        self._validade_data_shape(data=data)
        # Set instance attributes
        self._data = data
        self._nodata = nodata
        self._crs = CRS(crs)
        self._transform = transform
        self._bounds = self._compute_bounds()

    @property
    def data(self) -> np.ma.core.MaskedArray[Any]:
        """Raster data as a numpy masked array."""
        return self._data

    @property
    def nodata(self) -> Any:
        """Value used to represent no data values."""
        return self._nodata

    @property
    def crs(self) -> CRS:
        """An CRS object representing the raster coordinate reference system."""
        return self._crs

    @property
    def transform(self) -> Affine:
        """Raster transform."""
        return self._transform

    @property
    def bounds(self) -> rasterio.coords.BoundingBox:
        "Raster bounds."
        return self._bounds

    @staticmethod
    def to_masked_array(values: NDArray[Any], nodata: Any) -> np.ma.core.MaskedArray:
        """Convets a conventional numpy array to a masked array.

        Args:
            values (NDArray[Any]): input array.
            nodata (Any): the value used to represent no data values in the input array.

        Returns:
            np.ma.core.MaskedArray: masked array.
        """
        if np.isnan(nodata):
            mask = np.isnan(values)
        else:
            mask = values == nodata
        return np.ma.masked_array(values, mask)

    @staticmethod
    def from_rasterio_dataset(dataset: rasterio.DatasetReader, band: int) -> RasterData:
        """Creates a RasterData object based on a rasterio's DatasetReader object.

        Args:
            dataset (rasterio.DatasetReader): a DatasetReader object.
            band (int): a band of the dataset.

        Returns:
            RasterData: a RasterData object.
        """
        return RasterData(
            data=dataset.read(band, masked=True),
            nodata=dataset.nodata,
            crs=dataset.crs,
            transform=dataset.transform,
        )

    def to_crs(self, target_crs: Any) -> RasterData:
        """Reprojects data to a target CRS (Coordinate Reference System).

        Args:
            target_crs (Any): It can be anything accepted to create a pyproj.CRS object.

        Returns:
            RasterData: a new RasterData object after reprojection.
        """
        src_height, src_width = self.data.shape
        dst_transform, dst_width, dst_height = calculate_default_transform(
            self.crs, target_crs, src_width, src_height, *self.bounds
        )

        dst_reprojected = np.zeros((dst_height, dst_width), dtype=self.data.dtype)
        reproject(
            source=self.data,
            destination=dst_reprojected,
            src_transform=self.transform,
            src_crs=self.crs,
            src_nodata=self.nodata,
            dst_transform=dst_transform,
            dst_crs=target_crs,
            dst_nodata=self.nodata,
            resampling=Resampling.nearest,
        )

        reprojected_raster = RasterData(
            data=self.to_masked_array(values=dst_reprojected, nodata=self.nodata),
            nodata=self.nodata,
            crs=target_crs,
            transform=dst_transform,
        )

        return reprojected_raster

    def _validade_data_shape(self, data: NDArray[Any]) -> None:
        """Checks if the input data have 2 dimensions."""
        if len(data.shape) != 2:
            raise ValueError(
                "To create a RasterData object, the parameter data must be a 2D array."
            )

    def _compute_bounds(self) -> rasterio.coords.BoundingBox:
        """Computes data bounds as a BoundingBox object."""
        src_height, src_width = self.data.shape
        bounds = array_bounds(
            height=src_height, width=src_width, transform=self.transform
        )
        bounds = rasterio.coords.BoundingBox(*bounds)
        return bounds
