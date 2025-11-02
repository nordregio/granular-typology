import rasterio
import numpy as np
from rasterio.warp import reproject, Resampling
import os
import time


def aggregate_to_reference_grid(
    reference_raster_path, input_raster_path, output_path, aggregation_method="mean"
):
    """
    Aggregate input raster to match the grid of a reference raster.
    Uses rasterio's optimized reproject with C++ resampling.
    Only outputs cells where the reference raster has valid data.
    """

    print("=" * 60)
    print("RASTER AGGREGATION TO REFERENCE GRID")
    print("=" * 60)

    # Reading reference raster metadata and data
    print("Reading reference raster metadata and data...")
    with rasterio.open(reference_raster_path) as ref_src:
        ref_profile = ref_src.profile.copy()
        ref_transform = ref_src.transform
        ref_crs = ref_src.crs
        ref_width = ref_src.width
        ref_height = ref_src.height
        ref_nodata = ref_src.nodata

        # Read reference data to create mask
        ref_data = ref_src.read(1)

        print(f"   Grid: {ref_width:,} x {ref_height:,} pixels")
        print(f"   Resolution: {ref_transform.a:.0f}m x {-ref_transform.e:.0f}m")
        print(f"   CRS: {ref_crs}")
        print(f"   NoData: {ref_nodata}")

    # Create reference mask (True where reference data exists)
    if ref_nodata is not None:
        ref_mask = ref_data != ref_nodata
    else:
        # If no nodata value defined, assume 0 or negative values are nodata
        ref_mask = ref_data > 0

    ref_valid_count = np.sum(ref_mask)
    print(
        f"   Valid pixels in reference: {ref_valid_count:,} / {ref_data.size:,} ({100*ref_valid_count/ref_data.size:.1f}%)"
    )

    # Reading input raster metadata
    print("\nReading input raster metadata...")
    with rasterio.open(input_raster_path) as input_src:
        input_transform = input_src.transform
        input_crs = input_src.crs
        input_width = input_src.width
        input_height = input_src.height
        input_nodata = input_src.nodata

        print(f"   Grid: {input_width:,} x {input_height:,} pixels")
        print(f"   Resolution: {input_transform.a:.1f}m x {-input_transform.e:.1f}m")
        print(f"   NoData: {input_nodata}")

    # Calculate aggregation ratio
    ratio_x = abs(ref_transform.a / input_transform.a)
    ratio_y = abs(ref_transform.e / input_transform.e)
    total_ratio = ratio_x * ratio_y

    print(f"\nAggregation: {total_ratio:.0f} input pixels → 1 output pixel")

    # Validate same CRS
    if ref_crs != input_crs:
        raise ValueError(f"CRS mismatch: Reference={ref_crs}, Input={input_crs}")

    # Set resampling method
    resampling_map = {
        "sum": Resampling.sum,
        "mean": Resampling.average,
        "max": Resampling.max,
        "min": Resampling.min,
    }

    resampling_method = resampling_map[aggregation_method]
    print(f"Using {aggregation_method.upper()} aggregation")

    # Configure output
    output_profile = ref_profile.copy()
    output_profile.update(
        {
            "dtype": "float32",
            "nodata": -9999,
            "compress": "lzw",
            "tiled": True,
            "blockxsize": 512,
            "blockysize": 512,
        }
    )

    # Allocate output array
    print(f"\nAllocating output array ({ref_height:,} x {ref_width:,})...")
    output_array = np.full(
        (ref_height, ref_width), output_profile["nodata"], dtype=np.float32
    )

    # Start aggregation
    print("\nStarting aggregation...")
    start_time = time.time()

    # Single optimized reproject operation
    with rasterio.open(input_raster_path) as input_src:
        reproject(
            source=rasterio.band(input_src, 1),
            destination=output_array,
            src_transform=input_transform,
            src_crs=input_crs,
            dst_transform=ref_transform,
            dst_crs=ref_crs,
            resampling=resampling_method,
            src_nodata=input_nodata,
            dst_nodata=output_profile["nodata"],
            num_threads=4,
        )

    elapsed_time = time.time() - start_time
    print(f"Aggregation completed in {elapsed_time:.1f} seconds")

    # Apply reference mask - set to nodata where reference doesn't have valid data
    print("\nApplying reference mask...")
    mask_start = time.time()
    output_array[~ref_mask] = output_profile["nodata"]
    mask_time = time.time() - mask_start
    print(f"Masking completed in {mask_time:.1f} seconds")

    # Statistics
    valid_mask = output_array != output_profile["nodata"]
    valid_count = np.sum(valid_mask)

    print(f"\nResults:")
    print(
        f"   Valid pixels: {valid_count:,} / {output_array.size:,} ({100*valid_count/output_array.size:.1f}%)"
    )

    if valid_count > 0:
        valid_data = output_array[valid_mask]
        print(f"   Min: {np.min(valid_data):.2f}")
        print(f"   Max: {np.max(valid_data):.2f}")
        print(f"   Mean: {np.mean(valid_data):.2f}")
        if aggregation_method == "sum":
            print(f"   Total: {np.sum(valid_data):.0f}")
    else:
        print("   WARNING: No valid data in output")

    # Verify mask alignment
    print(f"\nMask verification:")
    print(f"   Reference valid cells: {ref_valid_count:,}")
    print(f"   Output valid cells: {valid_count:,}")
    if ref_valid_count == valid_count:
        print("   Perfect mask alignment")
    else:
        print(f"   Mask difference: {abs(ref_valid_count - valid_count):,} cells")

    # Write result
    print(f"\nWriting to: {os.path.basename(output_path)}")
    with rasterio.open(output_path, "w", **output_profile) as dst:
        dst.write(output_array, 1)

    # Verify alignment
    print("\nVerifying raster alignment...")
    verify_raster_alignment(reference_raster_path, output_path)

    print("Done")
    return output_path


def verify_raster_alignment(raster1_path, raster2_path, tolerance=1e-6):
    """Verify alignment between two rasters."""

    with rasterio.open(raster1_path) as src1, rasterio.open(raster2_path) as src2:

        # Check all alignment criteria
        checks = [
            ("Dimensions", src1.width == src2.width and src1.height == src2.height),
            ("CRS", src1.crs == src2.crs),
            (
                "Transform",
                np.allclose(
                    [src1.transform[i] for i in range(6)],
                    [src2.transform[i] for i in range(6)],
                    atol=tolerance,
                ),
            ),
            (
                "Bounds",
                np.allclose(
                    [
                        src1.bounds.left,
                        src1.bounds.bottom,
                        src1.bounds.right,
                        src1.bounds.top,
                    ],
                    [
                        src2.bounds.left,
                        src2.bounds.bottom,
                        src2.bounds.right,
                        src2.bounds.top,
                    ],
                    atol=tolerance,
                ),
            ),
        ]

        print("Alignment checks:")
        all_pass = True
        for check_name, passed in checks:
            status = "PASS" if passed else "FAIL"
            print(f"   {status} {check_name}")
            if not passed:
                all_pass = False

        if all_pass:
            print("   Perfect alignment")
        else:
            print("   Alignment issues")

        return all_pass


def main():
    # File paths
    # reference_raster = "data/Eurostat_Census-GRID_2021_V2-0/ESTAT_OBS-VALUE-POPULATED_2021_V2.tiff"
    reference_raster = (
        "data/Eurostat_Census-GRID_2021_V2-0/ESTAT_OBS-VALUE-T_2021_V2_w_UK.tiff"
    )

    # Average of the Net Building Height (ANBH)
    # input_raster = "data/GHS_BUILT_H_ANBH_E2018_GLOBE_R2023A_54009_100_V1_0/processed/GHS_BUILT_H_ANBH_clipped.tif"
    # output_path = "data/GHS_BUILT_H_ANBH_E2018_GLOBE_R2023A_54009_100_V1_0/processed/GHS_BUILT_H_ANBH_aggregated.tif"

    # Average of the Gross Building Height (AGBH)
    input_raster = "data/GHS_BUILT_H_AGBH_E2018_GLOBE_R2023A_54009_100_V1_0/processed/GHS_BUILT_H_AGBH_clipped.tif"
    output_path = "data/GHS_BUILT_H_AGBH_E2018_GLOBE_R2023A_54009_100_V1_0/processed/GHS_BUILT_H_AGBH_aggregated.tif"

    # Validate inputs
    if not os.path.exists(reference_raster):
        print(f"Reference raster not found: {reference_raster}")
        exit(1)

    if not os.path.exists(input_raster):
        print(f"Input raster not found: {input_raster}")
        exit(1)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    try:
        result_path = aggregate_to_reference_grid(
            reference_raster_path=reference_raster,
            input_raster_path=input_raster,
            output_path=output_path,
            aggregation_method="mean",
        )

        print(f"\nAggregated raster saved to:")
        print(f"    {result_path}")

    except Exception as e:
        print(f"\nERROR: {e}")
        raise


main()
