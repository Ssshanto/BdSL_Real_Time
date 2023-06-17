import cv2
import time
import mediapipe as mp
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import tensorflow as tf

# Load the pretrained model
model = tf.keras.models.load_model('model-47-0.99.hdf5')
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Grabbing the Holistic Model from Mediapipe and
# Initializing the Model
mp_holistic = mp.solutions.holistic
holistic_model = mp_holistic.Holistic(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
 
# Initializing the drawing utils for drawing the facial landmarks on image
mp_drawing = mp.solutions.drawing_utils

bangla_characters = [
"(অ/য়)",
"(আ)",
"(ই/ঈ)",
"(উ/ঊ)",
"(র/ঋ/ড়/ঢ়)",
"(এ )",
"(ঐ)",
"(ও)",
"(ঔ)",
"(ক)",
"(খ/ক্ষ)",
"(গ)",
"(ঘ)",
"(ঙ)",
"(চ)",
"(ছ)",
"(জ/য)",
"(ঝ)",
"(ঞ)",
"(ট)",
"(ঠ)",
"(ড)",
"(ঢ)",
"(ণ/ন)",
"(ত)",
"(থ)",
"(দ)",
"(ধ)",
"(প)",
"(ফ)",
"(ব/ভ)",
"(ম)",
"(ল)",
"(শ/ষ/স)",
"(হ)",
"(ং)",
"(ँ)"]

def text_to_image(text):
    # Set font and size
    font = ImageFont.truetype("Nikosh.ttf", 25)

    image_width = font.getsize(text)[0]
    image_height = font.getsize(text)[1]

    image = Image.new("RGB", (image_width, image_height), "white")

    # Draw the text on the image
    draw = ImageDraw.Draw(image)
    draw.text((0, 0), text, font=font, fill="black")
    
    return image
    
bangla_character_images = []
for char in bangla_characters:
    bangla_character_images.append(text_to_image(char))
    
# print(bangla_character_images)
# bangla_character_images[0].show()

from scipy.special import softmax
import warnings
warnings.filterwarnings('ignore')

# (0) in VideoCapture is used to connect to your computer's default camera
capture = cv2.VideoCapture(0)
 
# Initializing current time and precious time for calculating the FPS
previousTime = 0
currentTime = 0
 
while capture.isOpened():
    # capture frame by frame
    ret, frame = capture.read()

    # resizing the frame for better view
    frame = cv2.resize(frame, (640, 480))
    
    # Converting the from BGR to RGB
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Making predictions using holistic model
    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    results = holistic_model.process(image)
    image.flags.writeable = True

    # Converting back the RGB image to BGR
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    mp_drawing.draw_landmarks(
      image,
      results.right_hand_landmarks,
      mp_holistic.HAND_CONNECTIONS
    )

    # Drawing Left hand Land Marks
    mp_drawing.draw_landmarks(
      image,
      results.left_hand_landmarks,
      mp_holistic.HAND_CONNECTIONS
    )

    # Calculating the FPS
    currentTime = time.time()
    fps = 1 / (currentTime-previousTime)
    previousTime = currentTime
    
    positions = []

    if results.right_hand_landmarks:
        if results.right_hand_landmarks.landmark:
            for hand_landmarks in results.right_hand_landmarks.landmark:
                positions.append(hand_landmarks.x)
                positions.append(hand_landmarks.y)
                positions.append(hand_landmarks.z)

            positions = np.array(positions)
            positions = (positions + 1) / 2
            
            batch_positions = np.array([positions,])
            batch_predictions = model.predict(batch_positions)
            
            # Displaying predicted label on the image
            # cv2.putText(image, "Prediction: " + str(int(pred_label)), (10, 400), cv2.FONT_HERSHEY_COMPLEX, 0.75, (255,255,255), 2)
            
            all_predictions = np.array(batch_predictions[0])
            
            scores = softmax(all_predictions)
            # print(all_predictions, all_predictions.sum())
            best_predictions = np.argsort(all_predictions)[::-1][:5]
            
            for i in range(len(best_predictions)):
                prediction_image = bangla_character_images[int(best_predictions[i])]

                open_cv_image = np.array(prediction_image) 
                # Convert RGB to BGR 
                prediction_image = open_cv_image[:, :, ::-1].copy() 

                height, width, channels = prediction_image.shape
                roi = image[300 + 30 * i : 300 + 30 * i + height, 10 : 10 + width]
                result = cv2.addWeighted(roi, 1, prediction_image, 1, 0)
                
                image[300 + 30 * i : 300 + 30 * i + height, 10 : 10 + width] = result
                cv2.putText(image, str(scores[best_predictions[i]]), (60, 300 + 30 * i + 25), cv2.FONT_HERSHEY_COMPLEX, 0.6, (255, 255, 255), 2)
                
                # cv2.putText(image, str(int(fps))+" FPS", (10, 70), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0,255,0), 2)

    # Displaying FPS on the image
    cv2.putText(image, str(int(fps))+" FPS", (10, 70), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0,255,0), 2)
    
    
    # Display the resulting image
    cv2.imshow("Hand Landmarks and BdSL Prediction", image)

    # Enter key 'q' to break the loop
    
    if cv2.waitKey(5) & 0xFF == ord('g'):
        print("Landmarks: ")
        for val in results.right_hand_landmarks.landmark:
            print(val)
        # print(results.right_hand_landmarks.landmark)
    
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

# When all the process is done
# Release the capture and destroy all windows
capture.release()
cv2.destroyAllWindows()