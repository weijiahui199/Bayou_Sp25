library(gstat)
library(purrr)
library(spgwr)
library(GWmodel)

#Load in data
rain <- read.csv("Data/Rainfall Complete/rainfall_complete.csv")
public <- read.csv("Data/BCW Public and Private Original Data/Public and Private csv/all_public_data.csv")
private <- read.csv("Data/BCW Public and Private Original Data/Public and Private csv/combined_private_data.csv")

rain_sf <- st_as_sf(rain, coords = c("Longitude", "Latitude"), crs = 4326) %>%
  st_transform(32615)  # UTM zone for Houston

# Extract time
rain_sf <- rain_sf %>%
  mutate(month = floor_date(as.Date(Date, format = "%d/%m/%Y"), "month"))

rain_monthly <- rain_sf %>%
  group_by(Source.Address, month) %>%
  summarise(rainfall = sum(Rainfall), geometry = st_geometry(geometry), .groups = "drop")

# Create a bounding box polygon that captures Houston
bbox_loop <- st_bbox(c(
  xmin = 254000,  # West
  xmax = 290000,  # East
  ymin = 3277500, # South
  ymax = 3312000  # North
), crs = 32615)

bbox_loop_sf <- st_as_sfc(bbox_loop)

# Create grid
grid <- st_make_grid(
  bbox_loop_sf,
  cellsize = 3000,
  what = "polygons"
) %>%
  st_sf(grid_id = 1:length(.), geometry = .)

# Join sensor data to grid cells
rain_with_grid <- st_join(rain_monthly, grid, join = st_within)

# 1. Drop geometry and aggregate rainfall by grid_id and month
grid_rain_month <- rain_with_grid %>%
  st_drop_geometry() %>%
  group_by(grid_id, month) %>%
  summarise(rainfall = mean(rainfall, na.rm = TRUE), .groups = "drop")

# 2. Reattach grid geometry
grid_rain_month <- grid_rain_month %>%
  left_join(grid %>% select(grid_id, geometry), by = "grid_id") %>%
  st_as_sf()

# Loop over months to fill missing grid cells
all_months <- unique(grid_rain_month$month)[1:36]


grid_interp <- map_dfr(all_months, function(m) {
  # Get this month's points
  points <- grid_rain_month %>%
    filter(month == m)

  # Get centroids as POINT geometries
  points <- points %>%
    mutate(geometry = st_centroid(geometry)) %>%
    st_as_sf()

  # Drop any rows with empty or missing geometry
  points <- points[!st_is_empty(points), ]
  points <- points[!is.na(st_coordinates(points)[, 1]), ]

  # Get empty grids
  empty_grids <- grid %>%
    filter(!grid_id %in% points$grid_id)

  # If no interpolation needed, return points as-is
  if (nrow(empty_grids) == 0 || nrow(points) == 0) {
    return(points)
  }

  # Perform IDW interpolation
  idw_result <- gstat::idw(
    formula = rainfall ~ 1,
    locations = as(points, "Spatial"),
    newdata = as(empty_grids, "Spatial"),
    idp = 2.0
  )

  # Combine observed + interpolated
  interpolated_sf <- st_as_sf(idw_result) %>%
    select(rainfall = var1.pred, geometry) %>%
    mutate(grid_id = empty_grids$grid_id, month = m)

  bind_rows(points, interpolated_sf)
})

# CLean up overflow data for modeling
private <- private %>%
  transmute(date = ymd(`Date.of.WW.Release`), longitude = long, latitude = lat, source = "private")

public <- public %>%
  transmute(
    date = ymd(Start_Date),
    latitude = lat,
    longitude = long,
    source = "public")
all_events <- bind_rows(private, public)

all_events <- all_events %>%
  filter(!is.na(latitude), !is.na(longitude))
write.csv(all_events, "all_events.csv", row.names = FALSE)

overflow_sf <- st_as_sf(all_events, coords = c("longitude", "latitude"), crs = 4326) %>%
  st_transform(32615)

# Convert to sf object and reproject to meters
events_sf <- st_as_sf(all_events, coords = c("longitude", "latitude"), crs = 4326) %>%
  st_transform(32615) %>%
  mutate(month = floor_date(date, "month"))

events_gridded <- st_join(events_sf, grid, join = st_within) %>%
  filter(!is.na(grid_id))  # keep only matched grid cells

event_counts <- events_gridded %>%
  st_drop_geometry() %>%
  group_by(grid_id, month) %>%
  summarise(overflows = n(), .groups = "drop") %>%
  left_join(grid, by = "grid_id") %>%
  st_as_sf()
library(tidyr)
gwr_data <- left_join(
  grid_interp,
  st_drop_geometry(event_counts),   # remove geometry
  by = c("grid_id", "month")
) %>%
  mutate(overflows = replace_na(overflows, 0))

coords <- st_coordinates(st_centroid(gwr_data))

#cleaned overflow data
st_write(event_counts, "overflow_revised.shp", delete_layer = TRUE)


# Run GWR
gwr_result_rain <- gwr(overflows ~ rainfall
, data = gwr_data, bandwidth = 4500, se.fit = TRUE, hatmatrix = TRUE, coords = coords)

gwr_df <- as.data.frame(gwr_result_rain$SDF)

gwr_sf <- st_as_sf(gwr_df, coords = c("X", "Y"))
st_crs(gwr_sf) <- 32615 

gwr_with_grid <- st_join(gwr_sf, grid, join = st_within)

gwr_summary <- gwr_with_grid %>%
  group_by(grid_id) %>%
  summarise(
    rain_coef = mean(rainfall, na.rm = TRUE), 
    rain_coef_se = mean(rainfall_se, na.rm = TRUE), 
    geometry = st_union(geometry),                 # preserve the polygon shape
    .groups = "drop"
  ) %>%
  st_as_sf()

gwr_data %>%
  group_by(grid_id) %>%
  summarise(over_range = max(overflows) - min(overflows)) %>%
  arrange(over_range)

gwr_summary2 <- gwr_summary %>%
  mutate(
     rain_t = rain_coef/rain_coef_se,
    rain_sig = abs(rain_t) > 1.645
  )
results_rain <- gwr_summary2[gwr_summary2$rain_sig == TRUE, ]

st_crs(results_rain) <- 32615 

gwr_summary2 <- gwr_summary2 %>%
  rename(se = rain_coef_se)

# Download GWR
st_write(results_rain, "gwr_grid.shp", delete_layer = TRUE)
# Download Grid Overlay
gwr_poly <- grid %>%
  left_join(st_drop_geometry(results_rain), by = "grid_id")
st_write(gwr_poly, "rain_revised.shp", delete_layer = TRUE)


##Now we want to split 70/30

set.seed(123)
n <- nrow(gwr_data)
train_idx <- sample(n, size = floor(0.7 * n))
test_idx  <- setdiff(seq_len(n), train_idx)

train_sf <- gwr_data[train_idx, ]
test_sf  <- gwr_data[test_idx, ]

train_coords <- coords[train_idx, ]
test_coords  <- coords[test_idx, ]

#Fit on training data
gwr_train <- gwr(
  overflows ~ rainfall,
  data = train_sf,
  coords = train_coords,
  bandwidth = 4500,
  se.fit = TRUE
)

#Predict on the test set
gwr_pred <- gwr.predict(
  formula = overflows ~ rainfall,
  data = train_sf,
  predictdata = test_sf,
  bw = 4500
)


# Extract predicted values and bind back to test_sf
test_preds <- st_as_sf(gwr_pred$SDF) %>%
  rename(pred_overflows = prediction)

#Evaluate performance
test_results <- test_sf %>%
  st_drop_geometry() %>%
  bind_cols(
    test_preds %>% 
      st_drop_geometry() %>% 
      select(pred_overflows)
  ) %>%
  mutate(
    resid   = overflows - pred_overflows,
    sq_err  = resid^2,
    abs_err = abs(resid)
  )
mse <- mean(test_results$sq_err)
mae  <- mean(test_results$abs_err)

cat("Test MSE =", round(mse,3), "\n",
    "Test  MAE =", round(mae,3), "\n")

# Get train performance

#Predict back on the train set
gwr_pred_train <- gwr.predict(
  formula     = overflows ~ rainfall,
  data        = train_sf,
  predictdata = train_sf,
  bw = 4500
)

#Extract the fitted values
train_preds <- st_as_sf(gwr_pred_train$SDF) %>%
  rename(pred_overflows = prediction)

train_results <- train_sf %>%
  st_drop_geometry() %>%
  bind_cols(
    train_preds %>% st_drop_geometry() %>% select(pred_overflows)
  ) %>%
  mutate(
    resid  = overflows - pred_overflows,
    sq_err = resid^2, 
    abs_err = abs(resid)
  )

#Compute train MSE
train_mse <- mean(train_results$sq_err)
train_mae <- mean(train_results$abs_err)
cat("Train  MAE =", round(train_mae, 3), "\n")
cat("Train  MSE =", round(train_mse, 3), "\n")

##Now we want to split 70/30

set.seed(123)
n <- nrow(gwr_data)
cutoff <- as.Date("2024-01-01")

train_sf <- filter(gwr_data, month <= cutoff)
test_sf  <- filter(gwr_data, month  > cutoff)

train_pts <- st_cast(train_sf, "POINT")
test_pts  <- st_cast(test_sf,  "POINT")
coords_train <- st_coordinates(train_pts)
coords_test  <- st_coordinates(test_pts)

df_train <- st_drop_geometry(train_pts)
df_test  <- st_drop_geometry(test_pts)

crs_utm15 <- CRS("+proj=utm +zone=15 +datum=WGS84 +units=m +no_defs")

train_sp <- SpatialPointsDataFrame(
  coords      = coords_train,
  data        = df_train,
  proj4string = crs_utm15
)
test_sp <- SpatialPointsDataFrame(
  coords      = coords_test,
  data        = df_test,
  proj4string = crs_utm15
)

# helper for metrics
calc_metrics <- function(obs, pred) {
  resid <- obs - pred
  list(
    MSE = mean(resid^2),
    MAE = mean(abs(resid))
  )
}

#Fit on training data
gwr_train <- gwr(
  overflows ~ rainfall,
  data = train_sp,
  coords = coordinates(train_sp),
  bandwidth = 4500,
  se.fit = TRUE
)

# Predict on train
pred_train <- gwr.predict(
  formula     = overflows ~ rainfall,
  data        = train_sp,
  predictdata = train_sp,
  bw          = 4500
)
# Predict on test
pred_med_test  <- gwr.predict(
  formula     = overflows ~ rainfall,
  data        = train_sp,
  predictdata = test_sp,
  bw          = 4500
)


train_pred <- pred_train$SDF@data$prediction
test_pred <- pred_test$SDF@data$prediction

# Compute metrics
train_met <- calc_metrics(train_sp@data$overflows, train_pred)
test_met  <- calc_metrics(test_sp@data$overflows,  test_pred)

# Print metrics
cat("Train MSE:", train_met$MSE, "\n")
cat("Train MAE:", train_met$MAE, "\n")
cat("Test MSE:", test_met$MSE, "\n")
cat("Test MAE:", test_met$MAE, "\n")

