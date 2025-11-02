# =============================================================================
# R Script: Hemeroby Index
# Author: Tristan Berchoux
# Purpose: Generate a 1km grid of hemeroby values from CORINE Land Cover
#          aligned with DEGURBA classification
# =============================================================================

# ----------------------------
# INSTALL AND LOAD PACKAGES
# ----------------------------
# Uncomment to install if needed:
# install.packages(c("terra", "dplyr", "readr"))

library(terra)
library(dplyr)
library(ggplot2)

# ----------------------------
# SET FILE PATHS
# ----------------------------

# Path to the CORINE Land Cover raster (e.g., GeoTIFF)
corine_file <- "/Users/berchoux/Downloads/hemeroby/U2018_CLC2018_V2020_20u1.tif"  

# Path to your official DEGURBA grid raster (1km resolution)
degurba_file <- "/Users/berchoux/Downloads/hemeroby/Grid/ESTAT_OBS-VALUE-POPULATED_2021_V2.tiff" 

# Output file path for the hemeroby GeoTIFF aligned with DEGURBA
output_tif <- "/Users/berchoux/Downloads/hemeroby/EU_hemeroby_index_2018.tif"

# ----------------------------
# LOAD RASTERS
# ----------------------------
corine <- terra::rast(corine_file)
degurba_grid <- terra::rast(degurba_file)

print(corine)
print(degurba_grid)

# ----------------------------
# CONVERT CORINE TO NUMERICAL
# ----------------------------
#terra::activeCat(corine) <- "CODE_18"
corine_numeric <- as.numeric(corine)

reclass_matrix <- as.matrix(
  cbind(
    is = c(1:44, 48),
    becomes = c(111,112,121,122,123,124,131,132,133,141,142,
                211,212,213,221,222,223,231,241,242,243,244,
                311,312,313,321,322,323,324,331,332,333,334,335,
                411,412,421,422,423,
                511,512,521,522,523,
                999)
    ))

corine <- terra::classify(corine, reclass_matrix, others = NA)

# ----------------------------
# CORINE CLC CODES AND CORRESPONDING HEMEROBY INDEX VALUES
# ----------------------------
hemeroby_table <- data.frame(
  CLC_CODE = c(
    # Level 7: Artificial surfaces (Metahemerob)
    111, 112, 121, 122, 123, 124,
    
    # Level 6: Mixed artificial (Polyhemerob)
    131, 132, 133, 141, 142,
    
    # Level 5: Intensive agriculture (α-euhemerob)
    211, 212, 213, 334, 422,
    
    # Level 4: Moderate agriculture (β-euhemerob) 
    221, 222, 223, 231, 241, 242, 243, 244,
    
    # Level 3: Semi-natural (Mesohemerob)
    321, 324,
    
    # Level 2: Near-natural (Oligohemerob)
    311, 312, 313, 322, 323, 331, 333, 411, 412, 421,
    
    # Level 1: Natural (Ahemerob)
    332, 335, 423
  ),
  HEMEROBY = c(
    rep(7, 6), rep(6, 5), rep(5, 5), rep(4, 8), rep(3, 2), rep(2, 10), rep(1, 3)
  )
)

# Create reclassification matrix for terra::classify (from, to, becomes)
reclass_matrix <- as.matrix(
  cbind(
    is = hemeroby_table$CLC_CODE,
    becomes = hemeroby_table$HEMEROBY
  )
)

# ----------------------------
# RECLASSIFY CORINE TO HEMEROBY INDEX
# ----------------------------
hemeroby <- terra::classify(corine, reclass_matrix, others = NA)
plot(hemeroby)

# ----------------------------
# ALIGN HEMEROBY TO DEGURBA GRID
# ----------------------------

# Check if CRS and extent are compatible; reproject hemeroby if necessary
if (!terra::compareGeom(hemeroby, degurba_grid, stopOnError = FALSE)) {
  hemeroby_reproj <- terra::project(hemeroby, degurba_grid, method='near')
}

# Resample hemeroby raster to DEGURBA grid resolution and alignment
hemeroby_on_degurba <- terra::resample(hemeroby_reproj, degurba_grid, method = "mode")
names(hemeroby_on_degurba) <- "hemeroby"

# ----------------------------
# SUMMARY OUTPUT
# ----------------------------
print(summary(hemeroby_on_degurba))
freco <- terra::freq(hemeroby_on_degurba)
print(freco)

ggplot(freco,aes(x=freco[,2],y=freco[,3]))+
  geom_col(color='black',fill='cadetblue3')+
  xlab('Hemeroby Index')+
  ylab('Count')


# ----------------------------
# EXPORT RESULT AS GEOTIFF
# ----------------------------
terra::writeRaster(hemeroby_on_degurba, filename = output_tif, overwrite = TRUE)
