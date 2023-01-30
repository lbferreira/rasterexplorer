from typing import Any, List, Optional, Union

import branca as bc
import folium
import matplotlib
import matplotlib.cm as cm
import matplotlib.colors as mpl_colors
import numpy as np
import rasterio
from numpy.typing import NDArray

from .raster_data import RasterData


def explore(
    input_raster: Union[rasterio.DatasetReader, str, RasterData],
    label: str,
    band: int = 1,
    folium_map: Optional[folium.Map] = None,
    cmap: str = "RdYlGn",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    cbar_caption: str = "Scale",
    tiles: str = "OpenStreetMap",
    attr: Optional[str] = None,
    add_layer_control: bool = True,
) -> folium.Map:
    """Add raster data to an interactive map created using folium.

    Args:
        input_raster (Union[rasterio.DatasetReader, str, RasterData]): input raster.
        label (str): raster label.
        band (int, optional): band to consider in the raster data. Band numbers start from 1.
        This parameter is ignored if input_raster is a RasterData object. Defaults to 1.
        folium_map (Optional[folium.Map], optional): a folium.Map object.
        If None is passed a new map is created. Defaults to None.
        cmap (str, optional): a string representing a matplotlib colormap. Defaults to 'RdYlGn'.
        vmin (Optional[float], optional): the minimum value to consider when plotting raster.
        If None, vmin is obtained from data. Defaults to None.
        vmax (Optional[float], optional): the maximum value to consider when plotting raster.
        If None, vmin is obtained from data. Defaults to None.
        cbar_caption (str, optional): color bar caption. Defaults to 'Scale'.
        tiles (str, optional): the tiles provider used to create the interactive map.
        Please check folium documentation for more details. Defaults to 'OpenStreetMap'.
        attr (Optional[str], optional): the attribution for a custom tiles server. Defaults to None.
        add_layer_control (bool, optional): whether to add a LayerControl object to the map.
        When using the same folium.Map object for multiple calls of this function,
        setting `add_layer_control` to True only on the last call can be useful as LayerControl
        works properly only when it is added as the last map layer. Defaults to True.

    Returns:
        folium.Map: the map object.
    """
    raster_data = _format_input_raster(input_raster, band)
    if raster_data.crs != "epsg:4326":
        raster_data = raster_data.to_crs("epsg:4326")
    folium_map = _add_to_map(
        raster_data=raster_data,
        label=label,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        cbar_caption=cbar_caption,
        folium_map=folium_map,
        tiles=tiles,
        attr=attr,
        add_layer_control=add_layer_control,
    )
    return folium_map


def _format_input_raster(input_raster: Any, band: int) -> RasterData:
    """Converts input data to a RasterData object.

    Args:
        input_raster (Any): input raster.
        band (int): band number.

    Raises:
        TypeError: raised if a not support data type is passed.

    Returns:
        RasterData: object with the input data.
    """
    if isinstance(input_raster, rasterio.DatasetReader):
        return RasterData.from_rasterio_dataset(dataset=input_raster, band=band)
    elif isinstance(input_raster, str):
        with rasterio.open(input_raster) as dataset:
            raster_data = RasterData.from_rasterio_dataset(dataset=dataset, band=band)
        return raster_data
    elif isinstance(input_raster, RasterData):
        return input_raster
    else:
        raise TypeError("The input raster does not have a valid type.")


def _add_to_map(
    raster_data: RasterData,
    label: str,
    cmap: str = "RdYlGn",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    cbar_caption: str = "",
    folium_map: Optional[folium.Map] = None,
    tiles: str = "OpenStreetMap",
    attr: Optional[str] = None,
    add_layer_control: bool = True,
    **kwargs
) -> folium.Map:
    """Add raster data to a folium map.

    Args:
        raster_data (RasterData): raster data.
        label (str): raster label.
        cmap (str, optional): a string representing a matplotlib colormap. Defaults to 'RdYlGn'.
        vmin (Optional[float], optional): the minimum value to consider when plotting raster.
        If None, vmin is obtained from data. Defaults to None.
        vmax (Optional[float], optional): the maximum value to consider when plotting raster.
        If None, vmin is obtained from data. Defaults to None.
        cbar_caption (str, optional): color bar caption. Defaults to ''.
        folium_map (Optional[folium.Map], optional): a folium.Map object.
        If None is passed a new map is created. Defaults to None.
        tiles (str, optional): the tiles provider used to create the interactive map.
        Please check folium documentation for more details. Defaults to 'OpenStreetMap'.
        attr (Optional[str], optional): the attribution for a custom tiles server. Defaults to None.
        add_layer_control (bool, optional): whether to add a LayerControl object to the map.
        When using the same folium.Map object for multiple calls of this function,
        setting `add_layer_control` to True only on the last call can be useful as LayerControl
        works properly only when it is added as the last map layer. Defaults to True.

    Returns:
        folium.Map: a folium map object.
    """
    vmin = vmin if vmin is not None else raster_data.data.min()
    vmax = vmax if vmax is not None else raster_data.data.max()
    if folium_map is None:
        # Create the map
        bounds = raster_data.bounds
        mean_lat = (bounds.bottom + bounds.top) / 2
        mean_long = (bounds.left + bounds.right) / 2
        folium_map = folium.Map(
            location=[mean_lat, mean_long],
            tiles=tiles,
            attr=attr,
            zoom_start=kwargs.pop("zoom_start", 16),
            **kwargs
        )
        # Add a colorbar
        cbar = _create_cbar(cmap=cmap, vmin=vmin, vmax=vmax, caption=cbar_caption)
        folium_map.add_child(cbar)

    # Adding layer to folium
    colored_array = _apply_cmap(
        dataset_array=raster_data.data, cmap=cmap, vmin=vmin, vmax=vmax
    )
    bounds_folium_format = _format_bounds_to_folium(bounds=raster_data.bounds)
    folium.raster_layers.ImageOverlay(
        colored_array,
        bounds_folium_format,
        opacity=1,
        name=label,
        control=True,
    ).add_to(folium_map)
    if add_layer_control:
        folium.LayerControl().add_to(folium_map)
    return folium_map


def _apply_cmap(
    dataset_array: np.ma.core.MaskedArray,
    cmap: str,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
) -> NDArray[np.uint8]:
    """Apply a cmap on an array.

    Args:
        dataset_array (np.ma.core.MaskedArray): raster data.
        cmap (str): colormap.
        vmin (Optional[float], optional): minimum value to consider when applying the colormap.
        Values below vmin are clipped. Defaults to None.
        vmax (Optional[float], optional): maximum value to consider when applying the colormap.
        Values above vmax are clipped. Defaults to None.

    Returns:
        NDArray[np.uint8]: an RGBA array.
    """
    dataset_array = dataset_array.copy()
    if vmin is not None:
        dataset_array[dataset_array < vmin] = vmin
    if vmax is not None:
        dataset_array[dataset_array > vmax] = vmax

    vmin = dataset_array.min() if vmin is None else vmin
    vmax = dataset_array.max() if vmax is None else vmax

    normalized_array = (dataset_array - vmin) / (vmax - vmin)
    scalar_map = matplotlib.cm.ScalarMappable(cmap=cmap)
    rgba_array = scalar_map.to_rgba(normalized_array, norm=False, bytes=True)
    return rgba_array


def _format_bounds_to_folium(bounds: rasterio.coords.BoundingBox) -> List[List[float]]:
    """Formats raster bounds as needed for use with folium.

    Args:
        bounds (rasterio.coords.BoundingBox): raster bounds.

    Returns:
        List[List[float]]: bounds in folium format.
    """
    return [[bounds.bottom, bounds.left], [bounds.top, bounds.right]]


def _create_cbar(
    cmap: str, vmin: float, vmax: float, caption: str
) -> bc.colormap.StepColormap:
    """Create a colobar to be used in a folium map.

    Args:
        cmap (str): colormap.
        vmin (float): minimum value.
        vmax (float): maximum value.
        caption (str): caption.

    Returns:
        bc.colormap.StepColormap: colorbar.
    """
    nb_steps = 100
    cb_colors = cm.get_cmap(cmap, nb_steps)(np.arange(nb_steps))
    cb_colors = np.apply_along_axis(
        mpl_colors.to_hex,
        1,
        cb_colors,
    )
    colorbar = bc.colormap.StepColormap(
        cb_colors,
        vmin=vmin,
        vmax=vmax,
        caption=caption,
    )
    return colorbar
