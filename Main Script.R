# --- 0. LIBRARY ---
library(tidyverse)
library(sp)
library(raster)
library(readxl)
library(keras)
library(reshape2)
library(gridExtra)

# --- 1. IMPORT DATA ---
data_gempa <- read_excel("Data/Gempa.xlsx")
coordinates(data_gempa) <- ~long+lat

# --- 2. BUAT RASTER GRID (Resolusi 1 derajat) ---
grid_raster <- raster(extent(data_gempa), res = 1)
grid_raster[] <- NA

# --- 3. HITUNG RATA-RATA MAGNITUDO PER GRID ---
mean_magnitude <- rasterize(data_gempa, grid_raster, field = "Magnitudo", fun = mean, na.rm = TRUE)

# --- 4. KONVERSI RASTER KE MATRIKS ---
y_matrix <- as.matrix(mean_magnitude)
y_matrix[is.na(y_matrix)] <- 0  # Ganti NA dengan 0

# --- 5. SPLIT DATA (80:20) BERDASARKAN GRID SEL ---
grid_height <- dim(y_matrix)[1]
grid_width <- dim(y_matrix)[2]

# Flatten ke vektor
y_vector <- as.vector(y_matrix)

# Buat index train dan test
set.seed(42)
n_pixel <- length(y_vector)
train_idx <- sample(1:n_pixel, size = 0.8 * n_pixel)
test_idx <- setdiff(1:n_pixel, train_idx)

# Buat matriks train dan test (dengan masking)
y_train_vector <- y_vector
y_train_vector[test_idx] <- 0  # Kosongkan bagian test
y_train_matrix <- matrix(y_train_vector, nrow = grid_height, ncol = grid_width)

y_test_vector <- y_vector
y_test_vector[train_idx] <- 0  # Kosongkan bagian train
y_test_matrix <- matrix(y_test_vector, nrow = grid_height, ncol = grid_width)

# Siapkan array input CNN
input_array <- array(y_train_matrix, dim = c(1, grid_height, grid_width, 1))  # X input
target_array <- input_array  # Target sama: belajar dari pola parsial

# Simpan ground truth untuk perbandingan prediksi
ground_truth_array <- array(y_matrix, dim = c(1, grid_height, grid_width, 1))

# --- 6. BANGUN MODEL CNN ---
model <- keras_model_sequential() %>%
  layer_conv_2d(filters = 32, kernel_size = c(3,3), activation = 'relu',
                input_shape = c(grid_height, grid_width, 1), padding = "same") %>%
  layer_max_pooling_2d(pool_size = c(2,2), padding = "same") %>%
  layer_conv_2d(filters = 64, kernel_size = c(3,3), activation = 'relu', padding = "same") %>%
  layer_upsampling_2d(size = c(2,2)) %>%
  layer_conv_2d(filters = 1, kernel_size = c(3,3), activation = 'linear', padding = "same")

# --- 7. KOMPILE & TRAIN MODEL ---
model %>% compile(
  loss = 'mean_squared_error',
  optimizer = optimizer_adam(),
  metrics = c('mae')
)

history <- model %>% fit(
  x = input_array,
  y = target_array,
  epochs = 50,
  verbose = 2
)

# --- 8. PREDIKSI ---
predicted <- model %>% predict(input_array)
predicted_matrix <- predicted[1,,,1]

# --- 9. EVALUASI ---
error_matrix <- predicted_matrix - y_matrix

mae <- mean(abs(error_matrix[test_idx]))
rmse <- sqrt(mean(error_matrix[test_idx]^2))

cat("MAE (Test Set):", round(mae, 4), "\nRMSE (Test Set):", round(rmse, 4), "\n")

# --- 10. VISUALISASI ---
mat_to_df <- function(mat, varname = "value") {
  df <- melt(mat)
  names(df) <- c("y", "x", varname)
  return(df)
}

input_df <- mat_to_df(y_matrix, "Actual")
pred_df  <- mat_to_df(predicted_matrix, "Predicted")
err_df   <- mat_to_df(error_matrix, "Error")

plot_data <- input_df %>%
  left_join(pred_df, by = c("x", "y")) %>%
  left_join(err_df, by = c("x", "y"))

# Buat visualisasi
p1 <- ggplot(plot_data, aes(x = x, y = y, fill = Actual)) +
  geom_tile() +
  scale_fill_viridis_c() +
  ggtitle("Ground Truth (Rata-rata Magnitudo)") +
  coord_equal()

p2 <- ggplot(plot_data, aes(x = x, y = y, fill = Predicted)) +
  geom_tile() +
  scale_fill_viridis_c() +
  ggtitle("CNN Prediction") +
  coord_equal()

p3 <- ggplot(plot_data, aes(x = x, y = y, fill = Error)) +
  geom_tile() +
  scale_fill_gradient2(low = "blue", mid = "white", high = "red", midpoint = 0) +
  ggtitle("Residual Error (Predicted - Actual)") +
  coord_equal()

grid.arrange(p1, p2, p3, ncol = 3)
