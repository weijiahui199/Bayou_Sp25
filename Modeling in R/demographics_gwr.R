overflow <- read.csv("../Data/Event Count Data by Zipcode/3day_public_and_private_combined.csv")
rainfall <- read.csv("../Data/Rainfall Data by Zipcode/3day_rainfall_by_zipcode.csv")
demographics <- read.csv("../Data/Demographic Data by Zipcode/race and income by zipcode.csv")


library("tidyverse")

# Reshape from wide to long format
response_long <- pivot_longer(overflow, 
                               cols = -Zipcode, 
                               names_to = "DateRange", 
                               values_to = "Response")
input_long <- pivot_longer(rainfall, 
                            cols = -Zipcode, 
                            names_to = "DateRange", 
                            values_to = "Input")

# Merge datasets
data_long <- left_join(response_long, input_long, by = c("Zipcode", "DateRange"))
data_prep <- data_long %>%
    mutate(
      CleanDateRange = gsub("^X", "", DateRange),
      StartDate = as.Date(CleanDateRange, format = "%Y.%m.%d"),
    EndDate = as.Date(sub("^.{11}", "", CleanDateRange), format = "%Y.%m.%d"), 
      TimeIndex = as.numeric(StartDate - min(StartDate, na.rm = TRUE))  
  )


results_by_mo <- data_prep %>%
  mutate(StartDate = ymd(StartDate),
         Month = floor_date(StartDate, "month"))

month_group <- results_by_mo %>%
  group_by(Zipcode, Month) %>%
  summarise(Response = sum(Response, na.rm = TRUE), .groups = "drop", Input = sum(Input, na.rm = TRUE))


library(tidygeocoder)

# Get coordinates
zip_coords <- overflow %>%
  select(Zipcode) %>%
  distinct() %>%
  mutate(Zipcode = paste(Zipcode, "Houston, TX, USA")) %>%
  geocode(address = Zipcode, method = "osm") %>%
  mutate(
    long = jitter(long, factor = 0.0001),
    lat  = jitter(lat, factor = 0.0001)
  )

library(GWmodel)
library(sp)

library(GWmodel)
library(dplyr)
library(lubridate)

month_group <-month_group %>% mutate(Zipcode = as.character(Zipcode))
zip_coords <- zip_coords %>%
  mutate(Zipcode = substr(as.character(Zipcode), 1, 5))


data_geo <- left_join(month_group, zip_coords, by = "Zipcode")

library(sp)

# Remove rows with missing coordinates
data_geo <- data_geo %>% filter(!is.na(long), !is.na(lat))


data_geo$long <- as.numeric(data_geo$long)
data_geo$lat <- as.numeric(data_geo$lat)

# Create spatial object
coordinates(data_geo) <- ~ long + lat
proj4string(data_geo) <- CRS("+proj=longlat +datum=WGS84")


data_geo <- as.data.frame(data_geo)
demographics <- demographics %>%
  rename(Zipcode = `zip.code`) %>%
  mutate(Zipcode = as.character(Zipcode))

#Joining with demographic data
data_geo_dem <- left_join(data_geo, demographics, 
                          by = "Zipcode")

#Cleaning varibale types and removing NAs
data_geo_dem <- data_geo_dem %>%
  mutate(
    Median.earnings..dollars. = as.numeric(Median.earnings..dollars.)) %>%
  rename(median_earnings = Median.earnings..dollars.)
data_geo_dem_clean <- data_geo_dem %>%
  filter(
    !is.na(Response),
    !is.na(Input),
    !is.na(median_earnings),
    !is.na(lat),
    !is.na(long)
  )


library(spgwr)
#Fitting median earnings GWR
gwr_result_medearn <- spgwr::gwr(Response ~ median_earnings, data = data_geo_dem_clean, 
  bandwidth = .05, se.fit = TRUE, hatmatrix = TRUE, 
  coords = cbind(data_geo_dem_clean$long, data_geo_dem_clean$lat))

zip_lookup <- data_geo_dem_clean %>%
  select(Zipcode, lat, long) %>%
  distinct() %>%
  rename(coord.y = lat, coord.x = long)

data_medearn <- as.data.frame(gwr_result_medearn$SDF)
data_medearn <- data_medearn %>%
  left_join(zip_lookup, by = c("coord.x", "coord.y"))
data_medearn <- data_medearn %>%
  group_by(Zipcode) %>%
  summarise(across(everything(), mean, na.rm = TRUE), .groups = "drop")
results_medearn <- data_medearn %>%
  mutate(
     med_t = median_earnings/median_earnings_se,
    med_sig = abs(med_t) > 1.645
  )

results_medearn <- results_medearn[results_medearn$med_sig == TRUE, ]

write.csv(results_medearn, "gwr_medearn.csv", row.names = FALSE)

#Fitting "other" GWR
gwr_result_other <- spgwr::gwr(Response ~ Some.Other.Race...., data = data_geo_dem_clean, 
  bandwidth = .05, se.fit = TRUE, hatmatrix = TRUE, 
  coords = cbind(data_geo_dem_clean$long, data_geo_dem_clean$lat))

data_other <- as.data.frame(gwr_result_other$SDF)
data_other <- data_other %>%
  left_join(zip_lookup, by = c("coord.x", "coord.y"))
data_other <- data_other %>%
  group_by(Zipcode) %>%
  summarise(across(everything(), mean, na.rm = TRUE), .groups = "drop")
results_other <- data_other %>%
  mutate(
    o_t = Some.Other.Race..../Some.Other.Race...._se,
    o_sig = abs(o_t) > 1.645
  )
results_other <- results_other[results_other$o_sig == TRUE, ]

write.csv(results_other, "gwr_other.csv", row.names = FALSE)


#Now split for training/testing
set.seed(123)
cutoff <- as.Date("2023-10-01")

train_sf <- filter(data_geo_dem_clean, Month <= cutoff)
test_sf  <- filter(data_geo_dem_clean, Month  > cutoff)

train_sp <- SpatialPointsDataFrame(
  coords      = train_sf[, c("long","lat")],
  data        = train_sf,
  proj4string = CRS("+proj=longlat +datum=WGS84")
)
test_sp <- SpatialPointsDataFrame(
  coords      = test_sf[, c("long","lat")],
  data        = test_sf,
  proj4string = CRS("+proj=longlat +datum=WGS84")
)

# Helper for metrics
calc_metrics <- function(obs, pred) {
  resid  <- obs - pred
  mse    <- mean(resid^2)
  mae    <- mean(abs(resid))
  list(MSE = mse, MAE = mae)
}

gwr_train_med <- gwr(
  Response ~ median_earnings,
  data = train_sp,
  bandwidth = 4500,
  se.fit = TRUE,          
  hatmatrix = TRUE  
)

pred_med_train <- gwr.predict(
  formula     = Response ~ median_earnings,
  data        = train_sp,
  predictdata = train_sp,
  bw          = 4500)

pred_med_test  <- gwr.predict(
  formula     = Response ~ median_earnings,
  data        = train_sp,
  predictdata = test_sp,
  bw          = 4500
)
train_med_pred <- pred_med_train$SDF@data$prediction
test_med_pred <- pred_med_test$SDF@data$prediction

# Compute metrics
train_med_met <- calc_metrics(train_sp@data$Response, train_med_pred)
test_med_met  <- calc_metrics(test_sp@data$Response,  test_med_pred)

# Print metrics
cat("Train MSE:", train_med_met$MSE, "\n")
cat("Train MAE:", train_med_met$MAE, "\n")
cat("Test MSE:", test_med_met$MSE, "\n")
cat("Test MAE:", test_med_met$MAE, "\n")

pred_ot_train <- gwr.predict(
  formula     = Response ~ `Some.Other.Race....`,
  data        = train_sp,
  predictdata = train_sp,
  bw          = 4500
)
pred_ot_test  <- gwr.predict(
  formula     = Response ~ `Some.Other.Race....`,
  data        = train_sp,
  predictdata = test_sp,
  bw          = 4500
)

train_oth_pred <- pred_ot_train$SDF@data$prediction
test_oth_pred <- pred_ot_test$SDF@data$prediction

# Compute metrics
train_oth_met <- calc_metrics(train_sp@data$Response, train_oth_pred)
test_oth_met  <- calc_metrics(test_sp@data$Response,  test_oth_pred)

# Print metrics
cat("Train MSE:", train_oth_met$MSE, "\n")
cat("Train MAE:", train_oth_met$MAE, "\n")
cat("Test MSE:", test_oth_met$MSE, "\n")
cat("Test MAE:", test_oth_met$MAE, "\n")
