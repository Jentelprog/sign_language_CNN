import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import tensorflow as tf

# Initialize webcam and hand detector
cap = cv2.VideoCapture(1)  # Use 0 for the default webcam, change to 1 if needed
detector = HandDetector(maxHands=1)  # Detect only one hand at a time

# Load the trained model
model = tf.keras.models.load_model("model/handV2_model.h5")
input_size = (256, 256)  # Model's expected input size

# Parameters
offset = 20  # Extra space around the detected hand
imgSize = 256  # Size of the square image used for prediction
labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M',
          'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'X', 'Y']
recognized_text = ""  # Stores the recognized letters

# Open the output file in append mode
file = open("output.txt", "a")

while True:
    success, img = cap.read()  # Capture a frame from the webcam
    if not success:
        print("Failed to capture image")
        break

    # Detect hands in the frame
    hands, img = detector.findHands(img, draw=False)
    if hands:
        hand = hands[0]  # Get the first detected hand
        x, y, w, h = hand['bbox']  # Get bounding box (x, y, width, height)

        # Ensure the bounding box does not exceed image boundaries
        x1, y1 = max(0, x - offset), max(0, y - offset)
        x2, y2 = min(img.shape[1], x + w + offset), min(img.shape[0], y + h + offset)

        # Create a blank white image (256x256) to place the hand image
        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255

        try:
            # Crop the detected hand area from the frame
            imgCrop = img[y1:y2, x1:x2]

            # Ensure the cropped image is valid
            if imgCrop.size == 0:
                raise ValueError(f"Cropped image is empty: bbox={x1},{y1},{x2},{y2}")

            aspectRatio = h / w  # Get the aspect ratio of the hand

            # Resize and center the cropped image within the white background
            if aspectRatio > 1:  # Tall image (height > width)
                k = imgSize / h  # Scale factor
                wCal = math.ceil(w * k)  # Adjusted width
                imgResize = cv2.resize(imgCrop, (wCal, imgSize))  # Resize maintaining aspect ratio
                wGap = math.ceil((imgSize - wCal) / 2)  # Centering gap
                imgWhite[:, wGap:wCal + wGap] = imgResize  # Place on white background
            else:  # Wide image (width > height)
                k = imgSize / w  # Scale factor
                hCal = math.ceil(h * k)  # Adjusted height
                imgResize = cv2.resize(imgCrop, (imgSize, hCal))  # Resize maintaining aspect ratio
                hGap = math.ceil((imgSize - hCal) / 2)  # Centering gap
                imgWhite[hGap:hCal + hGap, :] = imgResize  # Place on white background

            # Resize the processed image to match model input size
            imgWhiteResized = cv2.resize(imgWhite, input_size)

            # Normalize the image (convert pixel values from [0,255] to [0,1])
            imgWhiteResized = imgWhiteResized.astype('float32') / 255.0
            imgWhiteResized = np.expand_dims(imgWhiteResized, axis=0)  # Add batch dimension

            # Predict the letter using the trained model
            predictions = model.predict(imgWhiteResized)
            index = np.argmax(predictions)  # Get the index of the highest probability class
            predicted_label = labels[index]  # Convert index to letter
            confidence = predictions[0][index]  # Get confidence score

            # Display the predicted label and confidence on the frame
            cv2.putText(img, f"{predicted_label} ({confidence:.2f})", (x, y - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

            # Show the cropped and processed images
            cv2.imshow("ImageCrop", imgCrop)
            cv2.imshow("ImageWhite", imgWhite)

            # Wait for key press and avoid multiple inputs
            key = cv2.waitKey(1) & 0xFF
            if key == ord('s'):  # If 's' is pressed, store the recognized letter
                recognized_text += predicted_label
                file.write(predicted_label + "\n")  # Write each letter on a new line
                file.flush()  # Ensure data is written immediately

        except Exception as e:
            print(f"Error processing hand: {e}")
            continue

    # Display the main image
    cv2.imshow("Image", img)

    # Exit when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release webcam and close all windows
cap.release()
cv2.destroyAllWindows()
file.close()