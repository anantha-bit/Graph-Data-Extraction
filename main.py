# -----------------------------
# IMPORT LIBRARIES
# -----------------------------
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# -----------------------------
# STEP 1: LOAD IMAGE
# -----------------------------
image_path = "apple_graph.jpg"

img = cv2.imread(image_path)

if img is None:
    print("Error: Image not found")
    exit()

print("Image loaded successfully.")

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# -----------------------------
# STEP 2: EDGE DETECTION
# -----------------------------
edges = cv2.Canny(gray, 50, 150)
print("Edge detection completed.")

# -----------------------------
# STEP 3: EXTRACT POINTS
# -----------------------------
points = np.column_stack(np.where(edges > 0))
points = points[:, ::-1]  # convert (row,col) → (x,y)

print("Total raw points:", len(points))

# -----------------------------
# STEP 4: EXTRACT CURVE (MEDIAN METHOD)
# -----------------------------
filtered_points = []

for xi in np.unique(points[:, 0]):
    y_vals = points[points[:, 0] == xi][:, 1]
    filtered_points.append([xi, np.median(y_vals)])  # BEST METHOD

points = np.array(filtered_points)

# Convert to x, y
x = points[:, 0].astype(float)
y = points[:, 1].astype(float)

# -----------------------------
# STEP 5: INVERT Y-AXIS
# -----------------------------
y = max(y) - y

# -----------------------------
# STEP 6: REMOVE NOISE (CRITICAL FIX)
# -----------------------------
# Remove axis + grid + bottom noise
mask = (y > 150) & (y < 480)

x = x[mask]
y = y[mask]

# Reshape for regression
x = x.reshape(-1, 1)

print("Total clean points:", len(x))

# -----------------------------
# STEP 7: SAVE CSV
# -----------------------------
df = pd.DataFrame({
    "Time_Index": x.flatten(),
    "Price_Value": y
})

df.to_csv("generated_from_image.csv", index=False)

print("\nCSV generated successfully")
print(df.head())

# -----------------------------
# STEP 8: REGRESSION
# -----------------------------
model = LinearRegression()
model.fit(x, y)

predicted = model.predict(x)

print("\nRegression model trained")
print("Slope:", model.coef_[0])
print("Intercept:", model.intercept_)

# -----------------------------
# STEP 9: FUTURE PREDICTION
# -----------------------------
future_x = np.arange(x.max() + 1, x.max() + 101).reshape(-1, 1)
future_y = model.predict(future_x)

print("\nFuture prediction generated")

# -----------------------------
# GRAPH 1: CLEAN GRAPH
# -----------------------------
plt.figure(figsize=(10, 5))
plt.scatter(x, y, s=10, label="Extracted Curve")
plt.plot(x, predicted, color='red', label="Trend Line")
plt.title("Clean Extracted Graph with Trend")
plt.xlabel("Time Index")
plt.ylabel("Price")
plt.legend()
plt.grid()
plt.show()

# -----------------------------
# GRAPH 2: FUTURE GRAPH
# -----------------------------
plt.figure(figsize=(10, 5))
plt.plot(x, predicted, color='red', label="Current Trend")
plt.plot(future_x, future_y, linestyle='--', label="Future Prediction")
plt.title("Future Prediction Graph")
plt.xlabel("Time Index")
plt.ylabel("Price")
plt.legend()
plt.grid()
plt.show()

# -----------------------------
# PROJECT SUMMARY
# -----------------------------
print("\nProject Summary:")
print("1. Image processed successfully.")
print("2. Edge detection applied.")
print("3. Median filtering used to extract graph curve.")
print("4. Noise removed using range filtering.")
print("5. Clean dataset generated.")
print("6. Linear regression applied.")
print("7. Future prediction generated.")