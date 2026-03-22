import cv2
import numpy as np

# ─────────────────────────────────────────
# METHOD 1: Histogram Equalization
# Makes dark images brighter and clearer
# ─────────────────────────────────────────
def histogram_equalization(image):
    # Convert to YCrCb color space
    # Y = brightness, Cr/Cb = color info
    ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    
    # Split into channels
    y, cr, cb = cv2.split(ycrcb)
    
    # Apply equalization only on brightness channel
    y_equalized = cv2.equalizeHist(y)
    
    # Merge back
    ycrcb_equalized = cv2.merge([y_equalized, cr, cb])
    
    # Convert back to BGR
    result = cv2.cvtColor(ycrcb_equalized, cv2.COLOR_YCrCb2BGR)
    return result

# ─────────────────────────────────────────
# METHOD 2: CLAHE (Advanced Equalization)
# Better than basic histogram equalization
# Prevents over-brightening
# ─────────────────────────────────────────
def clahe_enhancement(image):
    # Convert to LAB color space
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    
    # Split channels
    l, a, b = cv2.split(lab)
    
    # Apply CLAHE on L (lightness) channel
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l_clahe = clahe.apply(l)
    
    # Merge back
    lab_clahe = cv2.merge([l_clahe, a, b])
    
    # Convert back to BGR
    result = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)
    return result

# ─────────────────────────────────────────
# METHOD 3: Brightness & Contrast Auto Fix
# Automatically adjusts brightness
# ─────────────────────────────────────────
def auto_brightness_contrast(image, clip_percent=1):
    # Calculate histogram
    hist = cv2.calcHist([image], [0], None, [256], [0, 256])
    hist_size = len(hist)
    
    # Find min and max brightness levels
    accumulator = []
    accumulator.append(float(hist[0][0]))
    for index in range(1, hist_size):
        accumulator.append(accumulator[index-1] + float(hist[index][0]))
    
    maximum = accumulator[-1]
    clip_hist_percent = (maximum/100.0) * clip_percent / 2.0
    
    # Locate left cut
    minimum_gray = 0
    while accumulator[minimum_gray] < clip_hist_percent:
        minimum_gray += 1
    
    # Locate right cut
    maximum_gray = hist_size - 1
    while accumulator[maximum_gray] >= (maximum - clip_hist_percent):
        maximum_gray -= 1
    
    # Calculate alpha and beta values
    alpha = 255 / (maximum_gray - minimum_gray)
    beta = -minimum_gray * alpha
    
    result = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    return result

# ─────────────────────────────────────────
# METHOD 4: Gamma Correction
# Fixes overexposed (too bright) images
# ─────────────────────────────────────────
def gamma_correction(image, gamma=1.5):
    inv_gamma = 1.0 / gamma
    table = np.array([
        ((i / 255.0) ** inv_gamma) * 255
        for i in np.arange(0, 256)
    ]).astype("uint8")
    return cv2.LUT(image, table)

# ─────────────────────────────────────────
# MASTER FUNCTION: Apply all preprocessing
# This is what we call in our main system
# ─────────────────────────────────────────
def preprocess_image(image):
    # Step 1: Check brightness level
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    brightness = np.mean(gray)
    
    print(f"📊 Brightness level: {brightness:.2f}")
    
    if brightness < 50:
        # Very dark image - apply CLAHE
        print("🌙 Dark image detected - applying CLAHE enhancement")
        image = clahe_enhancement(image)
    elif brightness > 200:
        # Too bright - apply gamma correction
        print("☀️ Overexposed image detected - applying gamma correction")
        image = gamma_correction(image, gamma=2.0)
    else:
        # Normal lighting - apply auto brightness
        print("✅ Normal lighting - applying auto brightness adjustment")
        image = auto_brightness_contrast(image)
    
    return image