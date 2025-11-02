import pandas as pd
import numpy as np
from shapely import wkb
from scipy.spatial import cKDTree
from sklearn.metrics import mean_squared_error, mean_absolute_error
import itertools
import os

# Step 1: Loading data
# Load the dataset from a Parquet file
df = pd.read_parquet("data/input_data_v2.parquet")

# Keep only relevant columns
# Rationale: Each column kept serves a specific purpose in the workflow
df = df[
    [
        "GRD_ID",             # Grid cell identifier - needed for tracking
        "POP_21",                  # Population count - helps understand missingness patterns
        "N_TOWN_5K_30MN",     # Number of towns possible to reach in 30 minutes by car (5K pop)
        "N_TOWN_10K_30MN",    # Number of towns possible to reach in 30 minutes by car (10K pop)
        "N_TOWN_50K_30MN",    # Number of towns possible to reach in 30 minutes by car (50K pop)
        "geometry",           # Polygon defining each grid cell - needed for coordinates
    ]
]

# Print dataset information for quality control
print("Initial dataset:")
print(df.info())
print(f"Missing values per column:\n{df.isnull().sum()}")

# Step 2: Data Preparation - Count Data Analysis
def analyze_count_distributions(df):
    """
    Analyze the distribution of count variables to understand the data structure.
    Count data often has different characteristics than continuous data.
    """
    print("\n=== COUNT DATA DISTRIBUTION ANALYSIS ===")

    count_cols = ["N_TOWN_5K_30MN", "N_TOWN_10K_30MN", "N_TOWN_50K_30MN"]

    for col in count_cols:
        if col in df.columns:
            data = df[col].dropna()
            print(f"\n{col}:")
            print(f"   Range: {data.min()} to {data.max()}")
            print(f"   Mean: {data.mean():.2f}, Std: {data.std():.2f}")
            print(f"   Zeros: {np.sum(data == 0)} ({100*np.sum(data == 0)/len(data):.1f}%)")
            print(f"   Percentiles: 25%={data.quantile(0.25):.1f}, 50%={data.quantile(0.5):.1f}, 75%={data.quantile(0.75):.1f}")

            # Check for common count values
            value_counts = data.value_counts().head(10)
            print(f"   Most common values: {dict(value_counts)}")

analyze_count_distributions(df)

# Convert WKB geometry to shapely polygons, then extract centroids
df['geometry'] = df['geometry'].apply(lambda b: wkb.loads(b))
df['centroid'] = df['geometry'].apply(lambda g: g.centroid)

# Extract x, y coordinates from centroids for spatial calculations
df['x'] = df['centroid'].apply(lambda g: g.x)
df['y'] = df['centroid'].apply(lambda g: g.y)

print(f"Coordinate range: X ({df['x'].min():.0f}, {df['x'].max():.0f}), Y ({df['y'].min():.0f}, {df['y'].max():.0f})")

# Step 3: Define the IDW Interpolation Function for Count Data
def idw_interpolation_counts(xy_known, values, xy_query, k=20, p=1.5, chunk_size=200000):
    """
    IDW interpolation optimized for count data

    Parameters:
    -----------
    xy_known : array-like, shape (n_known, 2)
        Coordinates of points with known values
    values : array-like, shape (n_known,)
        Known count values at the xy_known coordinates
    xy_query : array-like, shape (n_query, 2)
        Coordinates where we want to interpolate values
    k : int, default=20
        Number of nearest neighbors to use in interpolation
        - Higher k for count data to get more stable estimates
        - Count data can be more variable spatially
    p : float, default=1.5
        Power parameter controlling distance decay
        - Lower p than travel time data (less aggressive decay)
        - Count data often has more gradual spatial transitions
    chunk_size : int, default=200000
        Process points in chunks to manage memory usage

    Returns:
    --------
    array-like : Interpolated count values (continuous, will need rounding)
    """

    # Build KD-tree for efficient nearest neighbor search
    tree = cKDTree(xy_known)
    n = xy_query.shape[0]
    result = np.empty(n)

    # Process points in chunks to handle large datasets efficiently
    for i in range(0, n, chunk_size):
        end = min(i + chunk_size, n)

        # Find k nearest neighbors and their distances
        dist, idx = tree.query(xy_query[i:end], k=k)

        # Handle exact matches (distance = 0)
        dist[dist == 0] = 1e-10

        # Compute IDW weights
        weights = 1 / (dist ** p)
        weights /= weights.sum(axis=1)[:, None]

        # Compute weighted average
        result[i:end] = (weights * values[idx]).sum(axis=1)

    return result

def validate_count_interpolation(original_values, interpolated_values, variable_name):
    """
    Validate interpolated count values with count-specific checks
    """
    print(f"\n--- Validation for {variable_name} ---")

    # Basic statistics
    print(f"Original range: [{original_values.min()}, {original_values.max()}]")
    print(f"Interpolated range: [{interpolated_values.min():.3f}, {interpolated_values.max():.3f}]")

    # Check for negative values (should not exist for counts)
    negative_count = np.sum(interpolated_values < 0)
    if negative_count > 0:
        print(f"WARNING: {negative_count} negative interpolated values detected!")
        print(f"Minimum interpolated value: {interpolated_values.min():.3f}")

    # Check for extreme values
    original_max = original_values.max()
    interpolated_max = interpolated_values.max()
    if interpolated_max > original_max * 1.5:
        print(f"WARNING: Interpolated maximum ({interpolated_max:.1f}) exceeds 150% of original maximum ({original_max})")

    # Distribution comparison
    print(f"Original mean: {original_values.mean():.2f}, Interpolated mean: {interpolated_values.mean():.2f}")
    print(f"Original zeros: {np.sum(original_values == 0)} ({100*np.sum(original_values == 0)/len(original_values):.1f}%)")

    # After rounding
    rounded_values = np.round(interpolated_values)
    rounded_values = np.maximum(rounded_values, 0)  # Ensure no negative counts
    print(f"After rounding - Range: [{rounded_values.min()}, {rounded_values.max()}]")
    print(f"After rounding - Mean: {rounded_values.mean():.2f}")

    return rounded_values

# Step 4: Cross-validation optimized for count data
def idw_cv_counts(df, target_col, test_fraction=0.05, k_range=None, p_range=None):
    """
    Cross-validation for count data with count-specific metrics
    """

    if k_range is None:
        # Test higher k values for count data (more neighbors for stability)
        k_range = [10, 15, 20, 25, 30, 35]
    if p_range is None:
        # Test lower p values for count data (less aggressive distance weighting)
        p_range = [1.0, 1.2, 1.5, 1.8, 2.0]

    # Filter to rows with known values
    known_data = df[~df[target_col].isna()].copy()
    print(f"Cross-validation for {target_col}: {len(known_data):,} known points")

    # Random split into train and test
    test_size = int(len(known_data) * test_fraction)
    test_idx = np.random.choice(len(known_data), test_size, replace=False)
    train_mask = ~known_data.index.isin(known_data.iloc[test_idx].index)

    train_data = known_data[train_mask]
    test_data = known_data.iloc[test_idx]

    print(f"Training points: {len(train_data):,}, Test points: {len(test_data):,}")

    # Prepare coordinate arrays
    train_coords = np.vstack([train_data['x'], train_data['y']]).T
    test_coords = np.vstack([test_data['x'], test_data['y']]).T
    train_values = train_data[target_col].values
    test_values = test_data[target_col].values

    # Test all combinations of k and p parameters
    results = []

    for k, p in itertools.product(k_range, p_range):
        try:
            # Predict test values using training data
            predicted_continuous = idw_interpolation_counts(
                train_coords, train_values, test_coords, k=k, p=p
            )

            # Round to integers and ensure non-negative (count constraint)
            predicted_counts = np.maximum(np.round(predicted_continuous), 0)

            # Calculate metrics
            rmse = np.sqrt(mean_squared_error(test_values, predicted_counts))
            mae = mean_absolute_error(test_values, predicted_counts)

            # Count-specific metrics
            exact_matches = np.sum(test_values == predicted_counts) / len(test_values)
            within_1 = np.sum(np.abs(test_values - predicted_counts) <= 1) / len(test_values)

            # Correlation
            correlation = np.corrcoef(test_values, predicted_counts)[0, 1] if len(np.unique(test_values)) > 1 else 0

            # Check for negative predictions before rounding
            negative_preds = np.sum(predicted_continuous < 0)

            results.append({
                'k': k,
                'p': p,
                'rmse': rmse,
                'mae': mae,
                'correlation': correlation,
                'exact_matches': exact_matches,
                'within_1': within_1,
                'negative_preds': negative_preds,
                'mean_residual': np.mean(test_values - predicted_counts)
            })

        except Exception as e:
            print(f"Error with k={k}, p={p}: {e}")
            continue

    # Convert results to DataFrame and sort by RMSE
    results_df = pd.DataFrame(results).sort_values('rmse')

    print(f"\nTop 5 parameter combinations for {target_col}:")
    print(results_df.head()[['k', 'p', 'rmse', 'mae', 'exact_matches', 'within_1', 'negative_preds']].round(4))

    return results_df

# Step 5: Run cross-validation for each count variable
count_cols = ["N_TOWN_5K_30MN", "N_TOWN_10K_30MN", "N_TOWN_50K_30MN"]
best_params = {}

print("\n=== PARAMETER TUNING FOR COUNT DATA ===")

for col in count_cols:
    if col in df.columns:
        print(f"\n--- Optimizing parameters for {col} ---")

        # Perform cross-validation
        results = idw_cv_counts(df, col)
        best = results.iloc[0]  # Best result (lowest RMSE)

        best_params[col] = {
            "k": int(best.k),
            "p": float(best.p),
            "rmse": best.rmse,
            "mae": best.mae,
            "exact_matches": best.exact_matches,
            "within_1": best.within_1
        }

        print(f"   Optimal parameters for {col}:")
        print(f"   k={best.k} neighbors, p={best.p} power")
        print(f"   RMSE={best.rmse:.3f}, MAE={best.mae:.3f}")
        print(f"   Exact matches: {best.exact_matches:.1%}, Within ±1: {best.within_1:.1%}")

print(f"\n=== FINAL PARAMETER SUMMARY ===")
for col, params in best_params.items():
    print(f"{col}: k={params['k']}, p={params['p']}, RMSE={params['rmse']:.3f}")

# Step 6: Apply IDW interpolation to fill missing values
print("\n=== APPLYING IDW INTERPOLATION FOR COUNT DATA ===")

for col in count_cols:
    if col in df.columns:
        print(f"\nProcessing {col}...")

        # Separate known and missing values
        known = df[~df[col].isna()].copy()
        unknown = df[df[col].isna()].copy()

        print(f"   Known values: {len(known):,}")
        print(f"   Missing values to fill: {len(unknown):,}")

        if not unknown.empty:
            # Prepare coordinate arrays
            coords_known = np.vstack([known['x'], known['y']]).T
            coords_unknown = np.vstack([unknown['x'], unknown['y']]).T
            values = known[col].values

            print(f"   Using {best_params[col]['k']} neighbors with power {best_params[col]['p']}")

            # Apply IDW interpolation
            interpolated_continuous = idw_interpolation_counts(
                coords_known, values, coords_unknown,
                k=best_params[col]["k"],
                p=best_params[col]["p"]
            )

            # Validate and round the interpolated values
            interpolated_counts = validate_count_interpolation(
                values, interpolated_continuous, col
            )

            # Assign rounded values back to the dataframe
            df.loc[unknown.index, col] = interpolated_counts
            print(f"   Successfully filled {len(unknown):,} missing values")
        else:
            print(f"   No missing values found for {col}")

# Step 7: Final Quality Checks and Spatial Validation
print("\n=== COMPREHENSIVE QUALITY CHECKS ===")

# Check for missing values
print("Missing values after interpolation:")
missing_summary = df[count_cols].isna().sum()
print(missing_summary)

if missing_summary.sum() == 0:
    print("✓ All missing values successfully filled")

# Count-specific validations
print("\n--- Count Data Validation ---")
for col in count_cols:
    if col in df.columns:
        values = df[col].values
        print(f"\n{col}:")
        print(f"   Range: [{values.min()}, {values.max()}]")
        print(f"   Mean: {values.mean():.2f}, Std: {values.std():.2f}")
        print(f"   Zeros: {np.sum(values == 0)} ({100*np.sum(values == 0)/len(values):.1f}%)")
        print(f"   Integer check: All values are integers: {np.all(values == np.round(values))}")
        print(f"   Non-negative check: All values ≥ 0: {np.all(values >= 0)}")

        # Check for extreme values
        q99 = np.percentile(values, 99)
        extreme_values = np.sum(values > q99)
        print(f"   Values above 99th percentile ({q99:.1f}): {extreme_values}")

# Spatial consistency check (basic)
print("\n--- Spatial Consistency Check ---")
for col in count_cols:
    if col in df.columns:
        # Calculate basic spatial statistics
        values = df[col].values
        coords = np.vstack([df['x'], df['y']]).T

        # Build KD-tree and check local similarity
        tree = cKDTree(coords)

        # For each point, check similarity with 8 nearest neighbors
        distances, indices = tree.query(coords, k=9)  # k=9 includes the point itself

        local_differences = []
        for i in range(len(values)):
            neighbors = indices[i][1:]  # Exclude the point itself
            neighbor_values = values[neighbors]
            local_diff = np.mean(np.abs(values[i] - neighbor_values))
            local_differences.append(local_diff)

        avg_local_diff = np.mean(local_differences)
        print(f"{col}: Average difference with 8 nearest neighbors: {avg_local_diff:.2f}")

# Step 8: Prepare for export
print("\n=== PREPARING DATA FOR EXPORT ===")

# Convert geometries back to WKB format
df['geometry_wkb'] = df['geometry'].apply(lambda g: g.wkb)
df['centroid_wkb'] = df['centroid'].apply(lambda g: g.wkb)
df["geometry"] = df['geometry_wkb']

# Clean up temporary columns
columns_to_drop = ['centroid', 'geometry_wkb', 'centroid_wkb', 'x', 'y']
df = df.drop(columns=[col for col in columns_to_drop if col in df.columns])

# Final dataset information
print(f"Final dataset shape: {df.shape}")
print(f"Final columns: {list(df.columns)}")

# Export results
output_file = "data/accessibility_smoothed/town_counts_imputed_v2.parquet"
df.to_parquet(
    output_file,
    engine='pyarrow',
    index=False,
    compression='snappy'
)

print(f"\n✓ Saved to {output_file}")
print(f"✓ File size: {os.path.getsize(output_file) / (1024**2):.1f} MB")

# print("\n=== INTERPOLATION SUMMARY ===")
# for col in count_cols:
#     if col in df.columns:
#         total_values = len(df)
#         non_null_values = df[col].notna().sum()
#         print(f"{col}: {total_values - non_null_values:,} values interpolated")
