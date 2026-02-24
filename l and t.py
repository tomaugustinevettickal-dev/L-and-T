import cv2
import numpy as np
import os

# -----------------------------
# IMAGE PATH
# -----------------------------
image_path = r"C:\Users\tomau\Desktop\image3.jpg"

if not os.path.exists(image_path):
    print(f"Image NOT found at: {image_path}")
    exit()

image = cv2.imread(image_path)

if image is None:
    print(f"Error: Unable to load image from {image_path}")
    exit()

# Resize (optional)
image = cv2.resize(image, (800, 600))

# Convert to HSV
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# -----------------------------
# HSV RANGES (Improved)
# -----------------------------

# Vegetation (Green)
veg_lower = np.array([35, 40, 40])
veg_upper = np.array([85, 255, 255])

# Water (Blue)
water_lower = np.array([90, 50, 50])
water_upper = np.array([140, 255, 255])

# Roads (Darker gray)
road_lower = np.array([0, 0, 60])
road_upper = np.array([180, 60, 180])

# Buildings (Bright / rooftops / concrete)
build_lower = np.array([0, 0, 180])
build_upper = np.array([180, 80, 255])

# -----------------------------
# CREATE MASKS
# -----------------------------
veg_mask = cv2.inRange(hsv, veg_lower, veg_upper)
water_mask = cv2.inRange(hsv, water_lower, water_upper)
road_mask = cv2.inRange(hsv, road_lower, road_upper)
build_mask = cv2.inRange(hsv, build_lower, build_upper)

# -----------------------------
# MORPHOLOGICAL CLEANING
# -----------------------------
kernel = np.ones((5, 5), np.uint8)

veg_mask = cv2.morphologyEx(veg_mask, cv2.MORPH_OPEN, kernel)
water_mask = cv2.morphologyEx(water_mask, cv2.MORPH_OPEN, kernel)
road_mask = cv2.morphologyEx(road_mask, cv2.MORPH_OPEN, kernel)
build_mask = cv2.morphologyEx(build_mask, cv2.MORPH_OPEN, kernel)

# -----------------------------
# CONTOUR FILTERING
# -----------------------------
def filter_small_regions(mask, min_area=500):
    filtered = np.zeros_like(mask)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        if cv2.contourArea(cnt) > min_area:
            cv2.drawContours(filtered, [cnt], -1, 255, -1)

    return filtered

veg_mask = filter_small_regions(veg_mask)
water_mask = filter_small_regions(water_mask)
road_mask = filter_small_regions(road_mask)
build_mask = filter_small_regions(build_mask)

# -----------------------------
# REMOVE OVERLAPPING CLASSES
# -----------------------------

# Roads should NOT include buildings
road_mask = cv2.bitwise_and(road_mask, cv2.bitwise_not(build_mask))

# Buildings should NOT include vegetation/water
build_mask = cv2.bitwise_and(build_mask, cv2.bitwise_not(veg_mask))
build_mask = cv2.bitwise_and(build_mask, cv2.bitwise_not(water_mask))

# -----------------------------
# PIXEL COUNTING
# -----------------------------
total_pixels = image.shape[0] * image.shape[1]

veg_pixels = cv2.countNonZero(veg_mask)
water_pixels = cv2.countNonZero(water_mask)
road_pixels = cv2.countNonZero(road_mask)
build_pixels = cv2.countNonZero(build_mask)

classified_pixels = veg_pixels + water_pixels + road_pixels + build_pixels
others_pixels = total_pixels - classified_pixels

# -----------------------------
# PERCENTAGE FUNCTION
# -----------------------------
def calc_percentage(count):
    return (count / total_pixels) * 100

# -----------------------------
# PRINT RESULTS
# -----------------------------
print("\n------ LAND CLASSIFICATION RESULTS ------")

print(f"Vegetation : {veg_pixels} pixels ({calc_percentage(veg_pixels):.2f}%)")
print(f"Water      : {water_pixels} pixels ({calc_percentage(water_pixels):.2f}%)")
print(f"Roads      : {road_pixels} pixels ({calc_percentage(road_pixels):.2f}%)")
print(f"Buildings  : {build_pixels} pixels ({calc_percentage(build_pixels):.2f}%)")
print(f"Others     : {others_pixels} pixels ({calc_percentage(others_pixels):.2f}%)")

# -----------------------------
# MARKDOWN TABLE OUTPUT (For Report)
# -----------------------------
print("\n| Land Class | Pixel Count | Percentage |")
print("|------------|-------------|------------|")
print(f"| Buildings  | {build_pixels} | {calc_percentage(build_pixels):.2f}% |")
print(f"| Roads      | {road_pixels} | {calc_percentage(road_pixels):.2f}% |")
print(f"| Vegetation | {veg_pixels} | {calc_percentage(veg_pixels):.2f}% |")
print(f"| Water      | {water_pixels} | {calc_percentage(water_pixels):.2f}% |")

# -----------------------------
# COLORED OVERLAY (Correct Priority)
# -----------------------------
overlay = image.copy()

overlay[veg_mask > 0] = (0, 255, 0)        # Green
overlay[water_mask > 0] = (255, 0, 0)      # Blue
overlay[road_mask > 0] = (128, 128, 128)   # Gray
overlay[build_mask > 0] = (0, 0, 255)      # Red (highest priority)

blended = cv2.addWeighted(image, 0.6, overlay, 0.4, 0)

# -----------------------------
# DISPLAY WINDOWS
# -----------------------------
cv2.imshow("Original Image", image)
cv2.imshow("Vegetation Mask", veg_mask)
cv2.imshow("Water Mask", water_mask)
cv2.imshow("Road Mask", road_mask)
cv2.imshow("Building Mask", build_mask)
cv2.imshow("Segmented Overlay", blended)

cv2.waitKey(0)
cv2.destroyAllWindows()
