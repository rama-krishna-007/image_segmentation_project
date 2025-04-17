import cv2
import numpy as np

# Load the image
image = cv2.imread('sample.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply thresholding
_, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)

# Find contours
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Draw contours
output = image.copy()
cv2.drawContours(output, contours, -1, (0, 255, 0), 2)

# Save and show result
cv2.imwrite('segmented_output.jpg', output)
cv2.imshow('Segmented Image', output)
cv2.waitKey(0)
cv2.destroyAllWindows()
