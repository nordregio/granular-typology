import os
import time
import warnings
from typing import List, Tuple, Dict, Any

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely import wkb
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
)

warnings.filterwarnings("ignore")


class RuralClassifier:
    """
    Classifier using MiniBatchKMeans with silhouette-based k-selection.
    """

    def __init__(
        self,
        exclude_degurba_values: List[int] = [30],
        k_range: Tuple[int, int] = (3, 10),
        batch_size: int = 10000,
        random_state: int = 42,
        verbose: bool = True,
    ):
        """
        Initialize the classifier.

        Parameters:
        -----------
        exclude_degurba_values : List[int]
            DEGURBA values to exclude from analysis
        k_range : Tuple[int, int]
            Range of cluster numbers to test (min_k, max_k)
        batch_size : int
            Batch size for MiniBatchKMeans
        random_state : int
            Random state for reproducibility
        verbose : bool
            Whether to print progress messages
        """
        self.exclude_degurba_values = exclude_degurba_values
        self.k_range = k_range
        self.batch_size = batch_size
        self.random_state = random_state
        self.verbose = verbose

        # Store results
        self.df_prepared = None
        self.clustering_features = None
        self.kmeans_model = None
        self.scaler = None
        self.n_clusters = None
        self.quality_metrics = None

    def _print(self, message: str):
        """Print message if verbose is True."""
        if self.verbose:
            print(message)

    def load_and_prepare_data(
        self,
        input_file: str,
        accessibility_traveltimes_file: str,
        accessibility_towncounts_file: str,
        degurba_file: str,
        crs: str = "EPSG:3035",
    ) -> pd.DataFrame:
        """Load and merge all data sources."""
        self._print("Loading data files...")

        # Load main data
        df = pd.read_parquet(input_file)
        df = df[
            [
                "GRD_ID",
                "POP_21",
                "ChgR_11_21",
                "GHS_BUILT_H_AGBH",
                "hemeroby_index",
                "geometry",
            ]
        ]

        # Load and merge accessibility data
        df_acc_traveltimes = pd.read_parquet(accessibility_traveltimes_file)
        df_acc_towncounts = pd.read_parquet(accessibility_towncounts_file)

        df = df.merge(
            df_acc_traveltimes[
                ["GRD_ID", "log_time_5k", "log_time_10k", "log_time_50k"]
            ],
            on="GRD_ID",
            how="left",
        )

        df = df.merge(
            df_acc_towncounts[
                ["GRD_ID", "N_TOWN_5K_30MN", "N_TOWN_10K_30MN", "N_TOWN_50K_30MN"]
            ],
            on="GRD_ID",
            how="left",
        )

        # Load and process DEGURBA data
        degurba = pd.read_parquet(degurba_file)

        # Convert WKB geometries
        degurba["geometry"] = degurba["geometry"].apply(
            lambda x: wkb.loads(x) if isinstance(x, (bytes, bytearray)) else x
        )
        df["geometry"] = df["geometry"].apply(
            lambda x: wkb.loads(x) if isinstance(x, (bytes, bytearray)) else x
        )

        # Create GeoDataFrames and filter
        degurba_gdf = gpd.GeoDataFrame(degurba, geometry="geometry", crs=crs)
        df_gdf = gpd.GeoDataFrame(df, geometry="geometry", crs=crs)

        self._print(f"Excluding DEGURBA values: {self.exclude_degurba_values}")
        degurba_filtered = degurba_gdf[
            ~degurba_gdf["value"].isin(self.exclude_degurba_values)
        ]

        joined = gpd.sjoin(df_gdf, degurba_filtered, how="inner", predicate="within")
        joined = joined.drop(columns="index_right")

        self._print(f"Data loaded: {len(joined):,} grid cells")
        return joined

    def prepare_features(
        self, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, np.ndarray, List[str]]:
        """Prepare features for clustering."""
        self._print("Preparing features...")

        df_clean = df.copy()

        # Handle missing values
        town_count_vars = ["N_TOWN_5K_30MN", "N_TOWN_10K_30MN", "N_TOWN_50K_30MN"]
        for var in town_count_vars:
            if var in df_clean.columns:
                df_clean[var].fillna(0, inplace=True)

        if "GHS_BUILT_H_AGBH" in df_clean.columns:
            df_clean["GHS_BUILT_H_AGBH"].fillna(0, inplace=True)

        if "hemeroby_index" in df_clean.columns:
            mode_value = (
                df_clean["hemeroby_index"].mode().iloc[0]
                if len(df_clean["hemeroby_index"].mode()) > 0
                else 3
            )
            df_clean["hemeroby_index"].fillna(mode_value, inplace=True)

        # Create derived features
        df_clean["log_building_density"] = np.log1p(df_clean["GHS_BUILT_H_AGBH"])
        df_clean["log_population_change"] = np.sign(df_clean["ChgR_11_21"]) * np.log1p(
            np.abs(df_clean["ChgR_11_21"])
        )

        # Create accessibility features
        accessibility_features = []

        if "log_time_5k" in df_clean.columns and "N_TOWN_5K_30MN" in df_clean.columns:
            df_clean["small_town_accessibility"] = (
                1 / (df_clean["log_time_5k"] + 1)
            ) + (df_clean["N_TOWN_5K_30MN"] / 10)
            accessibility_features.append("small_town_accessibility")

        if "log_time_10k" in df_clean.columns and "N_TOWN_10K_30MN" in df_clean.columns:
            df_clean["medium_town_accessibility"] = (
                1 / (df_clean["log_time_10k"] + 1)
            ) + (df_clean["N_TOWN_10K_30MN"] / 10)
            accessibility_features.append("medium_town_accessibility")

        if "log_time_50k" in df_clean.columns and "N_TOWN_50K_30MN" in df_clean.columns:
            df_clean["large_city_accessibility"] = (
                1 / (df_clean["log_time_50k"] + 1)
            ) + (df_clean["N_TOWN_50K_30MN"] / 10)
            accessibility_features.append("large_city_accessibility")

        # Define clustering features
        core_features = [
            "log_population_change",
            "log_building_density",
            "hemeroby_index",
        ]
        clustering_features = core_features + accessibility_features

        self._print(f"Clustering features: {clustering_features}")

        # Create clustering dataset
        X_clustering = df_clean[clustering_features].copy()

        # Handle any remaining missing values
        if X_clustering.isnull().sum().sum() > 0:
            X_clustering = X_clustering.fillna(method="ffill").fillna(method="bfill")

        self.df_prepared = df_clean
        self.clustering_features = clustering_features

        return df_clean, X_clustering.values, clustering_features

    def find_optimal_k_silhouette(
        self, X_scaled: np.ndarray, sample_size: int = 100000
    ) -> int:
        """
        Find optimal k using silhouette score.

        Selection logic:
        - If any silhouette score > 0.5, select the highest k with score > 0.5
        - Otherwise, select k with maximum silhouette score
        """
        self._print("Finding optimal k using silhouette score...")
        self._print(f"Testing k from {self.k_range[0]} to {self.k_range[1]}")

        # Sample for faster computation
        if len(X_scaled) > sample_size:
            self._print(f"Sampling {sample_size:,} points for k-selection")
            sample_idx = np.random.choice(len(X_scaled), sample_size, replace=False)
            X_sample = X_scaled[sample_idx]
        else:
            X_sample = X_scaled
            sample_idx = np.arange(len(X_scaled))

        scores = []

        for k in range(self.k_range[0], self.k_range[1] + 1):
            self._print(f"  Testing k={k}...")

            kmeans = MiniBatchKMeans(
                n_clusters=k,
                batch_size=self.batch_size,
                random_state=self.random_state,
                n_init=10,
                max_iter=100,
            )

            labels = kmeans.fit_predict(X_scaled)
            sil = silhouette_score(X_sample, labels[sample_idx])

            scores.append({"k": k, "silhouette": sil})
            self._print(f"    Silhouette: {sil:.3f}")

        scores_df = pd.DataFrame(scores)

        # Selection logic
        high_quality = scores_df[scores_df["silhouette"] > 0.5]

        if len(high_quality) > 0:
            optimal_k = high_quality["k"].max()
            optimal_sil = high_quality.loc[
                high_quality["k"] == optimal_k, "silhouette"
            ].values[0]
            self._print(f"Found {len(high_quality)} k values with silhouette > 0.5")
            self._print(
                f"Selecting highest k: {optimal_k} (silhouette: {optimal_sil:.3f})"
            )
        else:
            optimal_k = scores_df.loc[scores_df["silhouette"].idxmax(), "k"]
            optimal_sil = scores_df["silhouette"].max()
            self._print("No silhouette scores > 0.5")
            self._print(f"Optimal k: {optimal_k} (silhouette: {optimal_sil:.3f})")

        return optimal_k

    def run_clustering(self, X: np.ndarray) -> Dict[str, Any]:
        """Apply MiniBatchKMeans clustering."""
        self._print(f"Running clustering on {len(X):,} points...")

        # Scale features
        self.scaler = RobustScaler()
        X_scaled = self.scaler.fit_transform(X)

        # Find optimal k
        self.n_clusters = self.find_optimal_k_silhouette(X_scaled)

        # Fit final model
        self._print(f"Fitting final model with k={self.n_clusters}...")
        self.kmeans_model = MiniBatchKMeans(
            n_clusters=self.n_clusters,
            batch_size=self.batch_size,
            random_state=self.random_state,
            n_init=20,
            max_iter=200,
        )

        cluster_labels = self.kmeans_model.fit_predict(X_scaled)
        self.df_prepared["cluster"] = cluster_labels

        # Calculate quality metrics
        sample_size = min(100000, len(X_scaled))
        sample_idx = np.random.choice(len(X_scaled), sample_size, replace=False)

        self.quality_metrics = {
            "silhouette": silhouette_score(
                X_scaled[sample_idx], cluster_labels[sample_idx]
            ),
            "calinski_harabasz": calinski_harabasz_score(X_scaled, cluster_labels),
            "davies_bouldin": davies_bouldin_score(X_scaled, cluster_labels),
            "inertia": self.kmeans_model.inertia_,
        }

        # Print cluster sizes
        cluster_sizes = self.df_prepared["cluster"].value_counts().sort_index()
        self._print("Cluster sizes:")
        for cluster in sorted(cluster_sizes.index):
            size = cluster_sizes[cluster]
            pct = size / len(self.df_prepared) * 100
            self._print(f"  Cluster {cluster}: {size:>9,} cells ({pct:>5.1f}%)")

        self._print("Clustering done")
        self._print(f"Silhouette: {self.quality_metrics['silhouette']:.3f}")
        self._print(
            f"Calinski-Harabasz: {self.quality_metrics['calinski_harabasz']:.1f}"
        )
        self._print(f"Davies-Bouldin: {self.quality_metrics['davies_bouldin']:.3f}")

        return {
            "cluster_labels": cluster_labels,
            "quality_metrics": self.quality_metrics,
            "n_clusters": self.n_clusters,
        }

    def save_results(self, output_dir: str, filename: str) -> str:
        """Save classification results."""
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, filename)
        self.df_prepared.to_parquet(output_path, index=False)
        self._print(f"Results saved to: {output_path}")
        return output_path

    def classify(
        self,
        input_file: str,
        accessibility_traveltimes_file: str,
        accessibility_towncounts_file: str,
        degurba_file: str,
        output_dir: str,
        filename: str,
        crs: str = "EPSG:3035",
    ) -> Dict[str, Any]:
        """Complete classification pipeline."""

        start_time = time.time()

        # Load data
        df = self.load_and_prepare_data(
            input_file,
            accessibility_traveltimes_file,
            accessibility_towncounts_file,
            degurba_file,
            crs,
        )

        # Prepare features
        df_prepared, X_clustering, clustering_features = self.prepare_features(df)

        # Run clustering
        clustering_results = self.run_clustering(X_clustering)

        # Save results
        output_path = self.save_results(output_dir, filename)

        elapsed = time.time() - start_time

        self._print(f"Time: {elapsed/60:.1f} minutes")
        self._print(f"Clusters: {self.n_clusters}")
        self._print(f"Silhouette: {self.quality_metrics['silhouette']:.3f}")
        self._print("=" * 70)

        return {
            "df_classified": self.df_prepared,
            "clustering_results": clustering_results,
            "output_path": output_path,
            "summary": {
                "total_cells": len(self.df_prepared),
                "n_clusters": self.n_clusters,
                "clustering_features": clustering_features,
                "quality_metrics": self.quality_metrics,
            },
        }


def apply_model_to_uk(
    classifier: RuralClassifier,
    main_grid_ids: set,
    deg_config: dict,
    output_dir: str = "data/results/",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Apply trained classifier to UK data.

    Parameters:
    -----------
    classifier : RuralClassifier
        Trained classifier from main data
    main_grid_ids : set
        Set of main data GRD_IDs to identify UK cells
    deg_config : dict
        DEGURBA configuration dictionary
    output_dir : str
        Output directory for results

    Returns:
    --------
    Tuple[pd.DataFrame, pd.DataFrame]
        UK classified data and output path
    """
    print(f"\n" + "." * 70)
    print(f"Applying main model to UK data")

    # Load combined data (suppress verbose output)
    classifier.verbose = False
    df_combined = classifier.load_and_prepare_data(
        input_file="data/input_data_v2.parquet",
        accessibility_traveltimes_file="data/accessibility_smoothed/accessibility_imputed_v2.parquet",
        accessibility_towncounts_file="data/accessibility_smoothed/town_counts_imputed_v2.parquet",
        degurba_file="data/degurba/grid_degurba.parquet",
    )

    # Identify UK cells
    df_uk = df_combined[~df_combined["GRD_ID"].isin(main_grid_ids)].copy()

    if len(df_uk) == 0:
        print(f"WARNING: No UK cells found for DEGURBA class {deg_config['class']}")
        return None, None

    # Prepare UK features
    df_uk_prepared, X_uk, _ = classifier.prepare_features(df_uk)
    classifier.verbose = True

    # Scale and predict
    X_uk_scaled = classifier.scaler.transform(X_uk)
    uk_clusters = classifier.kmeans_model.predict(X_uk_scaled)
    df_uk_prepared["cluster"] = uk_clusters

    # Save UK results
    output_path_uk = os.path.join(output_dir, deg_config["filename_uk"])
    df_uk_prepared.to_parquet(output_path_uk, index=False)

    print(f"UK classification complete")
    print(f"  Output: {output_path_uk}")

    return df_uk_prepared, output_path_uk


def combine_main_uk_results(
    df_main: pd.DataFrame,
    df_uk: pd.DataFrame,
    deg_config: dict,
    output_dir: str = "data/results/",
) -> pd.DataFrame:
    """
    Combine main and UK results into a single dataset.

    Parameters:
    -----------
    df_main : pd.DataFrame
        Main classified data
    df_uk : pd.DataFrame
        UK classified data
    deg_config : dict
        DEGURBA configuration dictionary
    output_dir : str
        Output directory for results

    Returns:
    --------
    pd.DataFrame
        Combined main+UK dataset
    """
    # Add region identifier
    df_main_copy = df_main.copy()
    df_main_copy["region"] = "main"

    df_uk_copy = df_uk.copy()
    df_uk_copy["region"] = "UK"

    # Combine datasets
    df_combined = pd.concat([df_main_copy, df_uk_copy], ignore_index=True)

    # Save combined results
    output_path_combined = os.path.join(output_dir, deg_config["filename_combined"])
    df_combined.to_parquet(output_path_combined, index=False)

    print(f"Combined main+UK dataset created")
    print(f"  Output: {output_path_combined}")
    print(
        f"  Total cells: {len(df_combined):,} (main: {len(df_main_copy):,}, UK: {len(df_uk_copy):,})"
    )

    # Print combined cluster distribution
    print(f"  Combined cluster distribution:")
    cluster_counts = df_combined["cluster"].value_counts().sort_index()

    for cluster, count in cluster_counts.items():
        pct = count / len(df_combined) * 100
        main_count = len(df_main_copy[df_main_copy["cluster"] == cluster])
        uk_count = len(df_uk_copy[df_uk_copy["cluster"] == cluster])

        print(
            f"    Cluster {cluster}: {count:>9,} cells ({pct:>5.1f}%) - main: {main_count:,}, UK: {uk_count:,}"
        )

    print("." * 70)

    return df_combined


def run_classification():
    """
    Run rural classification on main data and apply to UK data.

    Main workflow:
    1. Train classifier on main data for each DEGURBA class
    2. Apply trained model to UK data
    3. Combine main and UK results
    4. Save all outputs separately
    """

    # DEGURBA class configurations
    degurba_configs = [
        {
            "class": 11,
            "exclude": [30, 22, 21, 13, 12],
            "filename_main": "typology_kmeans_silhouette_11_main.parquet",
            "filename_uk": "typology_kmeans_silhouette_11_UK.parquet",
            "filename_combined": "typology_kmeans_silhouette_11.parquet",
        },
        {
            "class": 12,
            "exclude": [30, 22, 21, 13, 11],
            "filename_main": "typology_kmeans_silhouette_12_main.parquet",
            "filename_uk": "typology_kmeans_silhouette_12_UK.parquet",
            "filename_combined": "typology_kmeans_silhouette_12.parquet",
        },
        {
            "class": 13,
            "exclude": [30, 22, 21, 11, 12],
            "filename_main": "typology_kmeans_silhouette_13_main.parquet",
            "filename_uk": "typology_kmeans_silhouette_13_UK.parquet",
            "filename_combined": "typology_kmeans_silhouette_13.parquet",
        },
    ]

    all_results = {}

    for deg_config in degurba_configs:
        print("=" * 70)
        print(f"PROCESSING DEGURBA CLASS {deg_config['class']}")

        # ===== Train main classifier =====

        classifier = RuralClassifier(
            exclude_degurba_values=deg_config["exclude"],
            k_range=(2, 5),
            batch_size=10000,
            verbose=True,
        )

        results_main = classifier.classify(
            input_file="data/input_data.parquet",
            accessibility_traveltimes_file="data/accessibility_smoothed/accessibility_imputed.parquet",
            accessibility_towncounts_file="data/accessibility_smoothed/town_counts_imputed.parquet",
            degurba_file="data/degurba/grid_degurba.parquet",
            output_dir="data/results/",
            filename=deg_config["filename_main"],
        )

        # Get main data GRD_IDs for later UK identification
        main_grid_ids = set(results_main["df_classified"]["GRD_ID"])

        # ===== Apply model to UK data =====
        df_uk_prepared, output_path_uk = apply_model_to_uk(
            classifier=classifier,
            main_grid_ids=main_grid_ids,
            deg_config=deg_config,
            output_dir="data/results/",
        )

        if df_uk_prepared is None:
            continue

        # ===== Combine main and UK results =====
        df_combined_classified = combine_main_uk_results(
            df_main=results_main["df_classified"],
            df_uk=df_uk_prepared,
            deg_config=deg_config,
            output_dir="data/results/",
        )

        # Store results
        all_results[deg_config["class"]] = {
            "main": results_main,
            "uk": df_uk_prepared,
            "combined": df_combined_classified,
            "classifier": classifier,
        }

        # ===== Summary for this DEGURBA class =====
        print("-" * 70)
        print(f"SUMMARY FOR DEGURBA CLASS {deg_config['class']}:")
        print(f"Main cells:    {len(results_main['df_classified']):,}")
        print(f"UK cells:      {len(df_uk_prepared):,}")
        print(f"Combined:      {len(df_combined_classified):,}")
        print(f"Clusters:      {classifier.n_clusters}")
        print(
            f"Silhouette score: {results_main['summary']['quality_metrics']['silhouette']:.3f}"
        )
        print(f"\nOutput files:")
        print(f"  Main:     {results_main['output_path']}")
        print(f"  UK:       {output_path_uk}")
        print(f"  Combined: data/results/{deg_config['filename_combined']}")

    # ===== Final overall summary =====
    print("\n" + "=" * 70)
    print("OVERALL SUMMARY")
    print("=" * 70)

    for deg_class, results in all_results.items():
        print(f"\nDEGURBA Class {deg_class}:")
        print(f"  Main cells:    {len(results['main']['df_classified']):,}")
        print(f"  UK cells:      {len(results['uk']):,}")
        print(f"  Combined:      {len(results['combined']):,}")
        print(f"  Clusters:      {results['classifier'].n_clusters}")
        print(
            f"  Silhouette:    {results['main']['summary']['quality_metrics']['silhouette']:.3f}"
        )

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)
    print("\nOutput files for each DEGURBA class:")
    for deg_config in degurba_configs:
        print(f"\nDEGURBA {deg_config['class']}:")
        print(f"  - data/results/{deg_config['filename_combined']}")
        print(f"  - data/results/{deg_config['filename_main']}")
        print(f"  - data/results/{deg_config['filename_uk']}")

    return all_results


# if __name__ == "__main__":
#     results = run_classification()

results = run_classification()
