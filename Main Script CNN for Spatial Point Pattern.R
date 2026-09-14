library(readxl)
library(dplyr)
library(sf)
library(terra)
library(geosphere)
library(tidyr)
library(rnaturalearth)
library(ggplot2)

## 1. Buat grid 1x1 derajat dan hitung rata-rata magnitudo per grid
# Unduh peta dunia
world <- ne_countries(scale = "medium", returnclass = "sf")

# Baca data
data_gempa <- read_excel("Data/Gempa.xlsx")

# Plot dengan peta dunia sebagai latar
ggplot() +
  geom_sf(data = world, fill = "gray90", color = "white") +
  geom_point(data = data_gempa, aes(x = long, y = lat, color = Magnitudo), size = 2) +
  scale_color_viridis_c(option = "plasma") +
  coord_sf(xlim = range(data_gempa$long), ylim = range(data_gempa$lat), expand = FALSE) +
  labs(title = "Peta Sebaran Gempa", x = "Longitude", y = "Latitude", color = "Magnitudo") +
  theme_minimal()

data_gunung <- read_excel("Data/GunungApi.xlsx") %>% rename(long = 1, lat = 2)
# Plot dengan latar belakang peta
ggplot() +
  geom_sf(data = world, fill = "gray90", color = "white") +
  geom_point(data = data_gunung, aes(x = long, y = lat), color = "red", size = 2) +
  coord_sf(xlim = range(data_gunung$long), ylim = range(data_gunung$lat), expand = FALSE) +
  labs(title = "Peta Lokasi Gunung Api", x = "Longitude", y = "Latitude") +
  theme_minimal()


data_sesar <- read_excel("Data/Sesar.xlsx")      # x0, y0, x1, y1
# Ubah ke format sf: tiap baris jadi satu garis (LineString)
sesar_sf <- do.call(rbind, lapply(1:nrow(data_sesar), function(i) {
  coords <- matrix(c(data_sesar$x0[i], data_sesar$y0[i],
                     data_sesar$x1[i], data_sesar$y1[i]), 
                   ncol = 2, byrow = TRUE)
  st_sf(geometry = st_sfc(st_linestring(coords)), crs = 4326)
}))

# Plot dengan latar belakang peta
ggplot() +
  geom_sf(data = world, fill = "gray90", color = "white") +
  geom_sf(data = sesar_sf, color = "darkorange", size = 1) +
  coord_sf(xlim = range(c(data_sesar$x0, data_sesar$x1)),
           ylim = range(c(data_sesar$y0, data_sesar$y1)),
           expand = FALSE) +
  labs(title = "Peta Garis Sesar", x = "Longitude", y = "Latitude") +
  theme_minimal()

data_subduksi <- read_excel("Data/Subduksi.xlsx")# x0, y0, x1, y1

# Ubah data gempa ke sf
gempa_sf <- st_as_sf(data_gempa, coords = c("long", "lat"), crs = 4326)
# Ubah ke format sf: tiap baris jadi satu garis (LineString)
subduksi_sf <- do.call(rbind, lapply(1:nrow(data_subduksi), function(i) {
  coords <- matrix(c(data_subduksi$x0[i], data_subduksi$y0[i],
                     data_subduksi$x1[i], data_subduksi$y1[i]), 
                   ncol = 2, byrow = TRUE)
  st_sf(geometry = st_sfc(st_linestring(coords)), crs = 4326)
}))

# Plot dengan latar belakang peta
ggplot() +
  geom_sf(data = world, fill = "gray90", color = "white") +
  geom_sf(data = subduksi_sf, color = "darkorange", size = 1) +
  coord_sf(xlim = range(c(data_subduksi$x0, data_subduksi$x1)),
           ylim = range(c(data_subduksi$y0, data_subduksi$y1)),
           expand = FALSE) +
  labs(title = "Peta Garis Subduksi", x = "Longitude", y = "Latitude") +
  theme_minimal()

# Buat grid 1x1 derajat berdasarkan extent gempa
grid <- st_make_grid(gempa_sf, cellsize = 1, square = TRUE)
grid <- st_sf(geometry = grid)

# Hitung centroid grid (titik tengah)
grid$centroid <- st_centroid(grid$geometry)

# Hitung rata-rata magnitudo per grid
grid$avg_magnitude <- sapply(st_contains(grid, gempa_sf), function(id) {
  if (length(id) == 0) return(NA)
  mean(data_gempa$Magnitudo[id], na.rm = TRUE)
})

## 2. Hitung jarak terdekat ke gunung berapi
# Konversi data gunung ke sf
gunung_sf <- st_as_sf(data_gunung, coords = c("long", "lat"), crs = 4326)

# Dapatkan koordinat centroid grid
centroids <- st_coordinates(grid$centroid)

# Hitung jarak ke semua gunung, ambil yang minimum
grid$dist_gunung <- apply(centroids, 1, function(pt) {
  min(distHaversine(pt, st_coordinates(gunung_sf)))
})

## 3. Hitung jarak ke garis sesar
# Konversi sesar ke sf LINESTRING
sesar_sf <- do.call(rbind, lapply(1:nrow(data_sesar), function(i) {
  coords <- matrix(c(data_sesar$x0[i], data_sesar$y0[i],
                     data_sesar$x1[i], data_sesar$y1[i]), ncol = 2, byrow = TRUE)
  st_sf(geometry = st_sfc(st_linestring(coords)), crs = 4326)
}))

# Hitung jarak ke garis sesar
grid$dist_sesar <- st_distance(grid$centroid, sesar_sf) %>% 
  apply(1, min) %>% as.numeric()

## 4. Hitung jarak ke garis subduksi
subduksi_sf <- do.call(rbind, lapply(1:nrow(data_subduksi), function(i) {
  coords <- matrix(c(data_subduksi$x0[i], data_subduksi$y0[i],
                     data_subduksi$x1[i], data_subduksi$y1[i]), ncol = 2, byrow = TRUE)
  st_sf(geometry = st_sfc(st_linestring(coords)), crs = 4326)
}))

# Hitung jarak
grid$dist_subduksi <- st_distance(grid$centroid, subduksi_sf) %>%
  apply(1, min) %>% as.numeric()

## 5. Split Data Train-Test (80:20)
library(caret)  # untuk splitting

# Ambil data non-NA (yang punya magnitudo)
df_model <- grid %>% 
  st_drop_geometry() %>%
  dplyr::select(avg_magnitude, dist_gunung, dist_sesar, dist_subduksi) %>%
  drop_na()

set.seed(123)
train_idx <- createDataPartition(df_model$avg_magnitude, p = 0.8, list = FALSE)
train_data <- df_model[train_idx, ]
test_data  <- df_model[-train_idx, ]

## 6. Buat Model CNN di R
library(keras)

# Standarisasi
x_train <- scale(train_data[, -1])
y_train <- train_data$avg_magnitude

x_test <- scale(test_data[, -1])
y_test <- test_data$avg_magnitude

# Reshape ke array [samples, features, 1] agar bisa masuk CNN
x_train_array <- array_reshape(as.matrix(x_train), dim = c(nrow(x_train), 3, 1))
x_test_array  <- array_reshape(as.matrix(x_test), dim = c(nrow(x_test), 3, 1))

# Build model CNN sederhana
model <- keras_model_sequential() %>%
  layer_conv_1d(filters = 32, kernel_size = 2, activation = "relu", input_shape = c(3, 1)) %>%
  layer_flatten() %>%
  layer_dense(units = 16, activation = "relu") %>%
  layer_dense(units = 1)

# Compile
model %>% compile(
  loss = "mse",
  optimizer = "adam",
  metrics = c("mae")
)

# Train
history <- model %>% fit(
  x_train_array, y_train,
  epochs = 50,
  batch_size = 8,
  validation_split = 0.2,
  verbose = 1
)

## 7. Evaluasi dan Prediksi
# Evaluasi
model %>% evaluate(x_test_array, y_test)

# Prediksi
prediksi <- model %>% predict(x_test_array)

# Bandingkan dengan data asli
hasil <- data.frame(
  aktual = y_test,
  prediksi = as.vector(prediksi)
)

library(Metrics)  # bisa install dulu kalau belum ada: install.packages("Metrics")

# RMSE
rmse_val <- rmse(hasil$aktual, hasil$prediksi)
mape_val <- mape(hasil$aktual, hasil$prediksi)

cat(sprintf("RMSE: %.4f\nMAPE: %.4f\n", rmse_val, mape_val))

## 8. Prediksi untuk Semua Grid yang Memiliki Data
# Ambil semua baris yang memiliki jarak dan bukan NA
grid_model <- grid %>%
  st_drop_geometry() %>%
  dplyr::select(dist_gunung, dist_sesar, dist_subduksi, avg_magnitude) %>%
  drop_na()

# Standarisasi menggunakan parameter dari data training
x_all <- scale(grid_model[, 1:3], center = attr(x_train, "scaled:center"), scale = attr(x_train, "scaled:scale"))
x_all_array <- array_reshape(as.matrix(x_all), dim = c(nrow(x_all), 3, 1))

# Prediksi
pred_all <- model %>% predict(x_all_array)

## 9. Masukkan Prediksi ke dalam Grid
# Buat salinan grid yang hanya memiliki data lengkap
grid_valid <- grid[!is.na(grid$avg_magnitude) & !is.na(grid$dist_gunung), ]
grid_valid$predicted <- as.vector(pred_all)
grid_valid$residual <- grid_valid$avg_magnitude - grid_valid$predicted

## 10. Plot Data Asli dan Prediksi dalam Raster Grid
library(ggplot2)
library(patchwork)
library(ggspatial)
library(viridis)

# Peta data asli
p1 <- ggplot() +
  geom_sf(data = grid_valid, aes(fill = avg_magnitude), color = NA) +
  scale_fill_viridis(name = "Avg Magnitude", option = "magma") +
  labs(title = "Data Gempa Asli", x = "Longitude", y = "Latitude") +
  annotation_scale(location = "bl") +
  annotation_north_arrow(location = "tl", which_north = "true",
                         pad_x = unit(0.3, "cm"), pad_y = unit(0.5, "cm"),
                         style = north_arrow_fancy_orienteering) +
  coord_sf(expand = FALSE) +
  theme_minimal()

# Peta hasil prediksi
p2 <- ggplot() +
  geom_sf(data = grid_valid, aes(fill = predicted), color = NA) +
  scale_fill_viridis(name = "Predicted Magnitude", option = "plasma") +
  labs(title = "Hasil Prediksi CNN", x = "Longitude", y = "Latitude") +
  annotation_scale(location = "bl") +
  annotation_north_arrow(location = "tl", which_north = "true",
                         pad_x = unit(0.3, "cm"), pad_y = unit(0.5, "cm"),
                         style = north_arrow_fancy_orienteering) +
  coord_sf(expand = FALSE) +
  theme_minimal()

p3 <- ggplot() +
  geom_sf(data = grid_valid, aes(fill = residual), color = NA) +
  scale_fill_viridis(name = "Residual", option = "plasma") +
  labs(title = "Residual CNN", x = "Longitude", y = "Latitude") +
  annotation_scale(location = "bl") +
  annotation_north_arrow(location = "tl", which_north = "true",
                         pad_x = unit(0.3, "cm"), pad_y = unit(0.5, "cm"),
                         style = north_arrow_fancy_orienteering) +
  coord_sf(expand = FALSE) +
  theme_minimal()

# Gabungkan ketiga plot
p1 + p2 + p3
