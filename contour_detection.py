import cv2
from matplotlib import pyplot as plt

# Load image
img = cv2.imread('sample.jpg')

# Convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Apply thresholding to create binary image
_, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# Find contours
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# Find the contour with the largest area
max_area = 0
max_contour = None
for contour in contours:
    area = cv2.contourArea(contour)
    if area > max_area:
        max_area = area
        max_contour = contour

# Draw a red line at the bottom of the largest contour
x, y, w, h = cv2.boundingRect(max_contour)
cv2.line(img, (x, y + h), (x + w, y + h), (0, 0, 255), 3)

cv2.imwrite('result_image.jpg', img)

# Display image with red line
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()