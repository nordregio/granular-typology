import pandas as pd
import numpy as np
from shapely import wkb
from scipy.spatial import cKDTree
from sklearn.metrics import mean_squared_error
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
        "TIME_TOWN_5K_1",     # Travel time to nearest town (5K pop)
        "TIME_TOWN_10K_1",    # Travel time to nearest town (10K pop)
        "TIME_TOWN_50K_1",    # Travel time to nearest town (50K pop)
        "geometry",           # Polygon defining each grid cell - needed for coordinates
    ]
]

# Print dataset information for quality control
print("Initial dataset:")
print(df.info())
print(f"Missing values per column:\n{df.isnull().sum()}")

# Step 2: Data Preparation
def run_data_preparation_time_only(df):
    """
    Prepare time-to-town features and add log transforms.

    The log transformation is crucial because:
    1. Travel times are right-skewed (many short times, few very long ones)
    2. IDW works better with normally distributed data
    3. Prevents interpolated values from becoming negative
    4. Makes spatial averaging more meaningful (geometric vs arithmetic means)
    """
    print("\n=== DATA PREPARATION: TIME VARIABLES ONLY ===")

    df_clean = df.copy()

    # Log transform each time-to-town column
    # np.log1p(x) = log(1+x) prevents issues with zero values
    if "TIME_TOWN_5K_1" in df_clean:
        df_clean["log_time_5k"] = np.log1p(df_clean["TIME_TOWN_5K_1"])
    if "TIME_TOWN_10K_1" in df_clean:
        df_clean["log_time_10k"] = np.log1p(df_clean["TIME_TOWN_10K_1"])
    if "TIME_TOWN_50K_1" in df_clean:
        df_clean["log_time_50k"] = np.log1p(df_clean["TIME_TOWN_50K_1"])

    return df_clean

df = run_data_preparation_time_only(df)

# Convert WKB geometry to shapely polygons, then extract centroids
df['geometry'] = df['geometry'].apply(lambda b: wkb.loads(b))
df['centroid'] = df['geometry'].apply(lambda g: g.centroid)

# Extract x, y coordinates from centroids for spatial calculations
# These coordinates will be used for distance calculations in IDW
df['x'] = df['centroid'].apply(lambda g: g.x)
df['y'] = df['centroid'].apply(lambda g: g.y)

print(f"Coordinate range: X ({df['x'].min():.0f}, {df['x'].max():.0f}), Y ({df['y'].min():.0f}, {df['y'].max():.0f})")

###########################
# saving df with logged values before interpolation for the map

# df['geometry_wkb'] = df['geometry'].apply(lambda g: g.wkb)
# df['centroid_wkb'] = df['centroid'].apply(lambda g: g.wkb)
# df["geometry"] = df['geometry_wkb']
# columns_to_drop = ['centroid', 'geometry_wkb', 'centroid_wkb', 'x', 'y']
# df = df.drop(columns=[col for col in columns_to_drop if col in df.columns])
# output_file = "data/accessibility_log.parquet"
# df.to_parquet(
#     output_file,
#     engine='pyarrow',
#     index=False,
#     compression='snappy'
# )

##########################


# Step 3: Define the IDW Interpolation Function
def idw_interpolation(xy_known, values, xy_query, k=16, p=2, chunk_size=200000):
    """
    IDW interpolation

    Parameters:
    -----------
    xy_known : array-like, shape (n_known, 2)
        Coordinates of points with known values
    values : array-like, shape (n_known,)
        Known values at the xy_known coordinates
    xy_query : array-like, shape (n_query, 2)
        Coordinates where we want to interpolate values
    k : int, default=16
        Number of nearest neighbors to use in interpolation
        - Higher k = smoother results but more computation
        - Lower k = more local variation but potential instability
    p : float, default=2
        Power parameter controlling distance decay
        - p=1: linear decay, p=2: quadratic decay (standard)
        - Higher p = more localized influence
    chunk_size : int, default=200000
        Process points in chunks to manage memory usage
        - Prevents memory overflow with very large datasets
        - Balances memory usage vs. computational efficiency

    Returns:
    --------
    array-like : Interpolated values at query points
    """

    # Build KD-tree for efficient nearest neighbor search
    # KD-trees provide O(log n) search time vs O(n) for brute force
    tree = cKDTree(xy_known)
    n = xy_query.shape[0]
    result = np.empty(n)

    # Process points in chunks to handle large datasets efficiently
    # This prevents memory overflow while maintaining reasonable performance
    for i in range(0, n, chunk_size):
        end = min(i + chunk_size, n)

        # Find k nearest neighbors and their distances
        # query() returns (distances, indices) for k nearest points
        dist, idx = tree.query(xy_query[i:end], k=k)

        # Handle the case where query point exactly matches a known point
        # Setting distance to small value prevents division by zero
        # This preserves exact interpolation property of IDW
        dist[dist == 0] = 1e-10

        # Compute IDW weights using inverse distance relationship
        # weights = 1 / distance^p
        weights = 1 / (dist ** p)

        # Normalize weights so they sum to 1 for each query point
        # This ensures interpolated values stay within reasonable bounds
        weights /= weights.sum(axis=1)[:, None]

        # Compute weighted average: interpolated_value = Σ(weight_i × value_i)
        result[i:end] = (weights * values[idx]).sum(axis=1)

    return result

# Example usage with diagnostic output
def idw_with_diagnostics(xy_known, values, xy_query, k=16, p=2):
    """Wrapper function that provides diagnostic information about the interpolation"""

    print(f"IDW Parameters: k={k} neighbors, p={p} power")
    print(f"Known points: {len(xy_known):,}, Query points: {len(xy_query):,}")

    # Calculate some diagnostics
    mean_dist_to_kth_neighbor = []
    tree = cKDTree(xy_known)

    # Sample a few query points to estimate typical neighbor distances
    sample_idx = np.random.choice(len(xy_query), min(1000, len(xy_query)), replace=False)
    sample_dists, _ = tree.query(xy_query[sample_idx], k=k)

    print(f"Typical distance to {k}th nearest neighbor: {np.mean(sample_dists[:, -1]):.1f} units")
    print(f"Distance range: {np.min(sample_dists):.1f} to {np.max(sample_dists):.1f} units")

    return idw_interpolation(xy_known, values, xy_query, k, p)

# Step 4: Cross-validation to tune IDW parameters
def idw_cv(df, target_col, test_fraction=0.05, k_range=None, p_range=None):
    """
    Cross-validation to find optimal IDW parameters.

    Strategy: Hold out a random sample of known values, predict them using
    remaining known values, and measure prediction accuracy.

    Parameters:
    -----------
    df : DataFrame
        Dataset with coordinates and target variable
    target_col : str
        Name of column to interpolate
    test_fraction : float, default=0.05
        Fraction of known values to hold out for testing
        - 5% provides good estimate while keeping most data for training
        - Higher fractions reduce training data quality
        - Lower fractions reduce test reliability
    k_range : list, default=[4, 6, 8, 10, 12, 16, 20]
        Range of k values to test
    p_range : list, default=[1.0, 1.5, 2.0, 2.5, 3.0]
        Range of p values to test
    """

    if k_range is None:
        k_range = [4, 6, 8, 10, 12, 16, 20]
    if p_range is None:
        p_range = [1.0, 1.5, 2.0, 2.5, 3.0]

    # Filter to rows with known values for this variable
    known_data = df[~df[target_col].isna()].copy()
    print(f"Cross-validation for {target_col}: {len(known_data):,} known points")

    # Randomly split into train and test sets
    # Random sampling ensures test set represents full spatial range
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
            predicted = idw_interpolation(train_coords, train_values, test_coords, k=k, p=p)

            # Calculate root mean squared error
            rmse = np.sqrt(mean_squared_error(test_values, predicted))

            # Calculate additional metrics for comprehensive evaluation
            mae = np.mean(np.abs(test_values - predicted))  # Mean Absolute Error
            r2 = np.corrcoef(test_values, predicted)[0, 1]**2  # R-squared

            results.append({
                'k': k,
                'p': p,
                'rmse': rmse,
                'mae': mae,
                'r2': r2,
                'mean_residual': np.mean(test_values - predicted)  # Bias check
            })

        except Exception as e:
            print(f"Error with k={k}, p={p}: {e}")
            continue

    # Convert results to DataFrame and sort by RMSE
    results_df = pd.DataFrame(results).sort_values('rmse')

    print(f"\nTop 5 parameter combinations for {target_col}:")
    print(results_df.head()[['k', 'p', 'rmse', 'r2']].round(4))

    return results_df

# Run cross-validation for each log-transformed variable
log_cols = ["log_time_5k", "log_time_10k", "log_time_50k"]
best_params = {}

print("=== PARAMETER TUNING VIA CROSS-VALIDATION ===")

for col in log_cols:
    print(f"\n--- Optimizing parameters for {col} ---")

    # Perform cross-validation to find optimal k and p
    results = idw_cv(df, col)
    best = results.iloc[0]  # Best result (lowest RMSE)

    best_params[col] = {
        "k": int(best.k),
        "p": float(best.p),
        "rmse": best.rmse,
        "r2": best.r2
    }

    print(f"   Optimal parameters for {col}:")
    print(f"   k={best.k} neighbors, p={best.p} power")
    print(f"   RMSE={best.rmse:.4f}, R²={best.r2:.4f}")

print(f"\n=== FINAL PARAMETER SUMMARY ===")
for col, params in best_params.items():
    print(f"{col}: k={params['k']}, p={params['p']}, RMSE={params['rmse']:.4f}")

# Step 5: Applying IDW
print("=== APPLYING IDW INTERPOLATION TO FILL MISSING VALUES ===")

for col in log_cols:
    print(f"\nProcessing {col}...")

    # Separate known and missing values
    # This preserves all existing data (exact interpolation property)
    known = df[~df[col].isna()].copy()
    unknown = df[df[col].isna()].copy()

    print(f"   Known values: {len(known):,}")
    print(f"   Missing values to fill: {len(unknown):,}")

    if not unknown.empty:
        # Prepare coordinate arrays for IDW
        coords_known = np.vstack([known['x'], known['y']]).T
        coords_unknown = np.vstack([unknown['x'], unknown['y']]).T
        values = known[col].values

        print(f"   Using {best_params[col]['k']} neighbors with power {best_params[col]['p']}")

        # Apply IDW interpolation with optimized parameters
        interpolated = idw_interpolation(
            coords_known, values, coords_unknown,
            k=best_params[col]["k"],
            p=best_params[col]["p"]
        )

        # Quality checks on interpolated values
        print(f"   Interpolated value range: [{interpolated.min():.3f}, {interpolated.max():.3f}]")
        print(f"   Original value range: [{values.min():.3f}, {values.max():.3f}]")

        # Check for any anomalous values
        if interpolated.min() < 0:
            print(f"   Warning: {np.sum(interpolated < 0)} negative interpolated values detected")

        # Assign interpolated values back to the dataframe
        df.loc[unknown.index, col] = interpolated
        print(f"   Successfully filled {len(unknown):,} missing values")
    else:
        print(f"   No missing values found for {col}")

# Summary of interpolation results
print(f"\n=== INTERPOLATION SUMMARY ===")
total_filled = 0
for col in log_cols:
    filled = len(df[df.index.isin(df[df[col].notna()].index)]) - len(df[~df[col].isna()])
    print(f"{col}: {filled:,} values interpolated")
    total_filled += filled

print(f"Total values interpolated across all variables: {total_filled:,}")

# Step 6: Final Checks and Export
print("=== FINAL QUALITY CHECKS ===")

# Verify no missing values remain
print("Missing values after interpolation:")
missing_summary = df[log_cols].isna().sum()
print(missing_summary)

if missing_summary.sum() == 0:
    print("All missing values successfully filled")

# Check value ranges and distributions
print("\nValue distribution summary (log-transformed):")
for col in log_cols:
    values = df[col]
    print(f"{col}:")
    print(f"   Range: [{values.min():.3f}, {values.max():.3f}]")
    print(f"   Mean: {values.mean():.3f}, Std: {values.std():.3f}")
    print(f"   Percentiles: 25%={values.quantile(0.25):.3f}, 75%={values.quantile(0.75):.3f}")

# Back-transform to original scale for interpretation
print("\nBack-transformed values (original time scale):")
for col in log_cols:
    original_col = col.replace('log_time_', 'TIME_TOWN_').replace('5k', '5K_1').replace('10k', '10K_1').replace('50k', '50K_1')
    back_transformed = np.expm1(df[col])  # expm1 is inverse of log1p
    print(f"{original_col} (minutes):")
    print(f"   Range: [{back_transformed.min():.1f}, {back_transformed.max():.1f}]")
    print(f"   Median: {back_transformed.median():.1f}")

# Prepare data for export
print("\n=== PREPARING DATA FOR EXPORT ===")

# Convert geometries back to WKB format for efficient storage
# WKB is more compact than WKT and preserves precision
df['geometry_wkb'] = df['geometry'].apply(lambda g: g.wkb)
df['centroid_wkb'] = df['centroid'].apply(lambda g: g.wkb)

# Use WKB geometry as the main geometry column
df["geometry"] = df['geometry_wkb']

# Clean up temporary columns to reduce file size
columns_to_drop = ['centroid', 'geometry_wkb', 'centroid_wkb', 'x', 'y']
df = df.drop(columns=[col for col in columns_to_drop if col in df.columns])

# Final dataset information
print(f"Final dataset shape: {df.shape}")
print(f"Final columns: {list(df.columns)}")

output_file = "data/accessibility_smoothed/accessibility_imputed_v2.parquet"
df.to_parquet(
    output_file,
    engine='pyarrow',
    index=False,       # Don't save row indices
    compression='snappy'
)

print(f"Saved to {output_file}")
print(f"File size: {os.path.getsize(output_file) / (1024**2):.1f} MB")
