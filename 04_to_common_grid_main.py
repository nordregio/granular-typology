"""
Final data processing pipeline:
- Aligns and reprojects multiple raster files to a common grid
- Spatially joins and aggregates values per polygon
- Merges accessibility and population change data from separate files
- Exports final dataset as GeoPackage and Parquet
"""

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.transform import xy
from shapely.geometry import Point
from pathlib import Path
from typing import List, Dict, Optional, Union


class RasterProcessor:
    def __init__(self, tolerance: float = 1e-6):
        self.tolerance = tolerance

    def check_alignment(self, file_paths: List[Union[str, Path]]) -> bool:
        specs = []
        for path in file_paths:
            with rasterio.open(path) as src:
                specs.append(
                    {"crs": src.crs, "transform": src.transform, "shape": src.shape}
                )

        reference = specs[0]
        for spec in specs[1:]:
            if (
                spec["crs"] != reference["crs"]
                or not np.allclose(
                    spec["transform"][:6],
                    reference["transform"][:6],
                    rtol=self.tolerance,
                )
                or spec["shape"] != reference["shape"]
            ):
                return False
        return True

    def reproject_to_reference(
        self, src_path: Union[str, Path], reference_profile: Dict
    ) -> np.ndarray:
        with rasterio.open(src_path) as src:
            dst_data = np.empty(
                (reference_profile["height"], reference_profile["width"]),
                dtype=src.dtypes[0],
            )
            reproject(
                source=rasterio.band(src, 1),
                destination=dst_data,
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=reference_profile["transform"],
                dst_crs=reference_profile["crs"],
                resampling=Resampling.bilinear,
            )
        return dst_data

    def rasters_to_dataframe(
        self,
        file_paths: List[Union[str, Path]],
        column_names: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        if column_names is None:
            column_names = [f"band_{i}" for i in range(len(file_paths))]

        with rasterio.open(file_paths[0]) as ref:
            reference_profile = ref.profile.copy()
            height, width = ref.shape
            transform = ref.transform

        data_arrays = []
        if self.check_alignment(file_paths):
            for path in file_paths:
                with rasterio.open(path) as src:
                    data_arrays.append(src.read(1))
        else:
            with rasterio.open(file_paths[0]) as src:
                data_arrays.append(src.read(1))
            for path in file_paths[1:]:
                data_arrays.append(self.reproject_to_reference(path, reference_profile))

        cols, rows = np.meshgrid(np.arange(width), np.arange(height))
        xs, ys = xy(transform, rows, cols)

        df_data = {"x": np.array(xs).flatten(), "y": np.array(ys).flatten()}
        for data, col_name in zip(data_arrays, column_names):
            df_data[col_name] = data.flatten()

        df = pd.DataFrame(df_data)

        for i, path in enumerate(file_paths):
            with rasterio.open(path) as src:
                if src.nodata is not None:
                    df.loc[df[column_names[i]] == src.nodata, column_names[i]] = np.nan

        return df.dropna(subset=column_names, how="all").reset_index(drop=True)


class PolygonAggregator:
    DEFAULT_FIELDS = [
        "GRD_ID",
        "T",
        "M",
        "F",
        "Y_LT15",
        "Y_1564",
        "Y_GE65",
        "EMP",
        "NAT",
        "EU_OTH",
        "LAND_SURFACE",
    ]

    def aggregate_raster_to_polygons(
        self,
        df_raster: pd.DataFrame,
        gpkg_path: Union[str, Path],
        selected_fields: Optional[List[str]] = None,
        agg_funcs: Optional[Dict[str, str]] = None,
    ) -> gpd.GeoDataFrame:
        gdf_polys = gpd.read_file(gpkg_path)

        if selected_fields:
            fields = ["geometry"] + selected_fields
        else:
            fields = ["geometry"] + [
                col for col in self.DEFAULT_FIELDS if col in gdf_polys.columns
            ]
        gdf_polys = gdf_polys[fields]

        geometry = [Point(xy) for xy in zip(df_raster["x"], df_raster["y"])]
        gdf_points = gpd.GeoDataFrame(df_raster, geometry=geometry, crs=gdf_polys.crs)

        joined = gpd.sjoin(gdf_points, gdf_polys, how="inner", predicate="within")

        if agg_funcs is None:
            raster_cols = [
                c for c in df_raster.columns if c not in ["x", "y", "geometry"]
            ]
            agg_funcs = {col: "mean" for col in raster_cols}
            if "population" in agg_funcs:
                agg_funcs["population"] = "sum"

        aggregated = joined.groupby("GRD_ID").agg(agg_funcs).reset_index()
        return gdf_polys.merge(aggregated, on="GRD_ID", how="left")


class DatasetMerger:
    def __init__(self, base_gdf: gpd.GeoDataFrame):
        self.gdf = base_gdf.copy()

    def merge_csv(
        self, csv_path: Union[str, Path], on: str = "GRD_ID", how: str = "left"
    ) -> "DatasetMerger":
        self.gdf = self.gdf.merge(pd.read_csv(csv_path), on=on, how=how)
        return self

    def merge_parquet(
        self, parquet_path: Union[str, Path], on: str = "GRD_ID", how: str = "left"
    ) -> "DatasetMerger":
        self.gdf = self.gdf.merge(pd.read_parquet(parquet_path), on=on, how=how)
        return self

    def add_computed_column(
        self, column_name: str, source_column: Optional[str] = None, value=None
    ) -> "DatasetMerger":
        self.gdf[column_name] = self.gdf[source_column] if source_column else value
        return self

    def select_columns(self, columns: List[str]) -> "DatasetMerger":
        self.gdf = self.gdf[columns]
        return self

    def get_result(self) -> gpd.GeoDataFrame:
        return self.gdf

    def save(
        self,
        output_path: Union[str, Path],
        driver: str = "GPKG",
        also_save_parquet: bool = False,
    ) -> None:
        self.gdf.to_file(output_path, driver=driver)
        print(f"Saved output to {output_path}")
        if also_save_parquet:
            parquet_path = Path(output_path).with_suffix(".parquet")
            self.gdf.to_parquet(parquet_path)
            print(f"Saved Parquet to {parquet_path}")


def main():
    config = {
        "raster_files": [
            "data/Eurostat_Census-GRID_2021_V2-0/ESTAT_OBS-VALUE-POPULATED_2021_V2.tiff",
            "data/GHS_BUILT_H_AGBH_E2018_GLOBE_R2023A_54009_100_V1_0/processed/GHS_BUILT_H_AGBH_aggregated_masked.tif",
            "data/GHS_BUILT_H_ANBH_E2018_GLOBE_R2023A_54009_100_V1_0/processed/GHS_BUILT_H_ANBH_aggregated_masked.tif",
            "data/Hemeroby_Index/EU_hemeroby_index_2018_V0.tif",
        ],
        "column_names": [
            "population",
            "GHS_BUILT_H_AGBH",
            "GHS_BUILT_H_ANBH",
            "hemeroby_index",
        ],
        "gpkg_path": "data/Eurostat_Census-GRID_2021_V2-0/ESTAT_Census_2021_V2.gpkg",
        "selected_fields": [
            "GRD_ID",
            "T",
            "M",
            "F",
            "Y_LT15",
            "Y_1564",
            "Y_GE65",
            "EMP",
            "NAT",
            "EU_OTH",
            "LAND_SURFACE",
        ],
        "accessibility_csv": "data/GRANULAR-WP3-Accessibility-indicators/data/outputs/1km_indicators/accessibility_UE.csv",
        "population_change_parquet": "data/GEOSTAT-grid-change/DF_POP_GRID_MOD.parquet",
        "output_path": "data/input_data.gpkg",
    }

    print("Processing rasters")
    df_raster = RasterProcessor().rasters_to_dataframe(
        config["raster_files"], config["column_names"]
    )

    print("Aggregating to grid")
    gdf_base = PolygonAggregator().aggregate_raster_to_polygons(
        df_raster, config["gpkg_path"], config["selected_fields"]
    )

    print("Merging additional datasets")
    final_columns = [
        "geometry",
        "GRD_ID",
        "POP_21",
        "ChgR_11_21",
        "YrChgR_11_21",
        "GHS_BUILT_H_AGBH",
        "hemeroby_index",
        "TIME_TOWN_5K_1",
        "TIME_TOWN_10K_1",
        "TIME_TOWN_50K_1",
        "N_TOWN_5K_30MN",
        "N_TOWN_10K_30MN",
        "N_TOWN_50K_30MN",
    ]

    gdf_final = (
        DatasetMerger(gdf_base)
        .merge_csv(config["accessibility_csv"])
        .merge_parquet(config["population_change_parquet"], how="outer")
        .add_computed_column("POP_21", source_column="T")
        .select_columns(final_columns)
        .get_result()
    )

    print("Saving results...")
    DatasetMerger(gdf_final).save(
        config["output_path"], driver="GPKG", also_save_parquet=True
    )

    print("\nFinal dataset info:")
    print(gdf_final.info())


main()

# if __name__ == "__main__":
#     main()
