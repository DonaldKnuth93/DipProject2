import cv2
import matplotlib.pyplot as plt
import numpy as np

image_original = cv2.imread('./img/infertile egg.jpg')

# OpenCV python stores image in BGR mode by default, we need to convert the BGR into RGB and then operate on it.
imageRGB = cv2.cvtColor(image_original, cv2.COLOR_BGR2RGB)

# -----------------------------------------------------------------------------------------------------------------------------------------
# convert RGB image to Grayscale
image_gray = cv2.cvtColor(imageRGB, cv2.COLOR_RGB2GRAY)
# -----------------------------------------------------------------------------------------------------------------------------------------

# Reduce noise in image
img_median = cv2.medianBlur(image_gray, 5)  # Add median filter to image
img = cv2.GaussianBlur(img_median, (5, 5), 0)  # Add Gaussian filter to image

# -----------------------------------------------------------------------------------------------------------------------------------------
# Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
eggClahe = cv2.createCLAHE(clipLimit=5)
final_image = eggClahe.apply(img) + 5

# -----------------------------------------------------------------------------------------------------------------------------------------
# Apply Automatic OTSU's Binarization Thresholding
ret, threshEggOTSU = cv2.threshold(final_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
# Print Automatic OTSU's Binarization Thresholding Value
print("OTSU Threshold Value: ", ret)

# -----------------------------------------------------------------------------------------------------------------------------------------
# Crreate a subplot screen for the image outputs using plt library
(fig, (ax1, ax2, ax3, ax4, ax5, ax6, ax7)) = plt.subplots(1, 7, figsize=(30, 30))

# -----------------------------------------------------------------------------------------------------------------------------------------
# Apply Canny Edge detector on the binarized image
edgesCanny = cv2.Canny(threshEggOTSU, 100, 200)

# -----------------------------------------------------------------------------------------------------------------------------------------
# Show the detected edge image onto the screen before any other changes apply to it
ax5.title.set_text('Canny Edge Detection')
# show the image using color map (pink)
ax5.imshow(edgesCanny, cmap='pink')

# -----------------------------------------------------------------------------------------------------------------------------------------
# Apply ConvexHull geometrical ANN Algorithm
# copy the value of edgesCanny variable into imgHull
imgHull = edgesCanny

# find contours on imgHull image
contours, hirachy = cv2.findContours(imgHull, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# traverse through all contours on the image
hull = [cv2.convexHull(c) for c in contours]
# draw contours on the image
final = cv2.drawContours(imgHull, hull, -1, (51, 50, 255), 2)
# show the contoured ConvexHull image onto the subplot screen
ax6.title.set_text('Convex Hull')
ax6.imshow(final)

# -----------------------------------------------------------------------------------------------------------------------------------------
# labeling
num_labels, labels = cv2.connectedComponents(final)
# Map component labels to hue val, 0-179 is the hue range in OpenCV
label_hue = np.uint8(179 * labels / np.max(labels))
blank_ch = 255 * np.ones_like(label_hue)
labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])

# Converting cvt to BGR
labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)

# set bg label to black
labeled_img[label_hue == 0] = 0

# -----------------------------------------------------------------------------------------------------------------------------------------
# Plot outputs
ax1.title.set_text('Original Image')
ax1.imshow(imageRGB, cmap='gray')
ax2.title.set_text('Grayscale')
ax2.imshow(image_gray, cmap='gray')
ax3.title.set_text('CLAHE')
ax3.imshow(final_image, cmap='gray')
ax4.title.set_text('OTSU Threshold')
ax4.imshow(threshEggOTSU, cmap='gray')
ax7.title.set_text('CCL')
ax7.imshow(labeled_img)
# ax8.title.set_text('SIFT')
# ax8.imshow(kp)
plt.show()
