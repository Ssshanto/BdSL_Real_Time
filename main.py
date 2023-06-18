import cv2
import time
import mediapipe as mp
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import tensorflow as tf
from scipy.special import softmax
import warnings

# Load the pretrained model
model = tf.keras.models.load_model('model-47-0.99.hdf5')
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Grabbing the Hands Model from Mediapipe and
# Initializing the Model
mp_hands = mp.solutions.hands
# Initializing the drawing utils for drawing the facial landmarks on image
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
 


TOP_PREDICTIONS_NUM = 5

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

font = ImageFont.truetype("Nikosh.ttf", 25)

def text_to_image(text):
    # Set font and size

    image_width = font.getsize(text)[0]
    image_height = font.getsize(text)[1]

    image = Image.new("RGB", (image_width, image_height), "white")

    # Draw the text on the image
    draw = ImageDraw.Draw(image)
    draw.text((0, 0), text, font=font, fill="black")
    
    return image

def generate_percentage_bar(percentage, bar_length=30):
    filled_length = int(bar_length * percentage / 100)
    empty_length = bar_length - filled_length
    
    filled_bar = '|' * filled_length
    empty_bar = ' ' * empty_length
    
    bar = filled_bar + empty_bar
    percentage_text = f'{percentage:.1f}%'
    
    # Combine the bar and percentage text
    bar_with_text = f'{percentage_text} {bar}'
    
    return bar_with_text
    
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

with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:
    while capture.isOpened():
        # capture frame by frame
        ret, frame = capture.read()

        # resizing the frame for better view
        image = cv2.resize(frame, (640, 480))
        
        # Converting the from BGR to RGB
        image.flags.writeable = False
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(image)
        image.flags.writeable = True

        # Converting back the RGB image to BGR
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # print(results.multi_handedness)
        handedness_list = results.multi_handedness
        hand_landmarks_list = results.multi_hand_landmarks

        positions = []

        if results.multi_hand_landmarks:
            for i in range(len(handedness_list)):
                # print(handedness_list[i].classification[0].label)
                if handedness_list[i].classification[0].label == 'Left': # label == 'Left' indicates Right hand as Mediapipe works in mirrored input
                    mp_drawing.draw_landmarks(
                        image,
                        hand_landmarks_list[i],
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style())
                    
                    for landmark in hand_landmarks_list[i].landmark:
                        positions.append(landmark.x)
                        positions.append(landmark.y)
                        positions.append(landmark.z)

        # Calculating the FPS
        currentTime = time.time()
        fps = 1 / (currentTime-previousTime)
        previousTime = currentTime

        if len(positions) == 63:
                positions = np.array(positions)
                positions = (positions + 1) / 2
                
                batch_positions = np.array([positions,])
                batch_predictions = model.predict(batch_positions)
                
                # Displaying predicted label on the image
                # cv2.putText(image, "Prediction: " + str(int(pred_label)), (10, 400), cv2.FONT_HERSHEY_COMPLEX, 0.75, (255,255,255), 2)
                
                all_predictions = np.array(batch_predictions[0])
                
                scores = softmax(all_predictions)
                # print(all_predictions, all_predictions.sum())
                best_predictions = np.argsort(all_predictions)[::-1][:TOP_PREDICTIONS_NUM]
                sorted_score = np.sort(scores)[::-1]
                total_score = np.sum(sorted_score[:TOP_PREDICTIONS_NUM])
                
                for i in range(len(best_predictions)):
                    prediction_image = bangla_character_images[int(best_predictions[i])]

                    open_cv_image = np.array(prediction_image)
                    # Convert RGB to BGR 
                    prediction_image = open_cv_image[:, :, ::-1].copy() 

                    height, width, channels = prediction_image.shape
                    x, y = 300 + 30 * i, 10
                    roi1 = image[x : x + height, y : y + width]
                    result1 = cv2.addWeighted(roi1, 1, prediction_image, 1, 0)
                    image[x : x + height, y : y + width] = result1

                    percentage = 100 * (scores[best_predictions[i]] / total_score)
                    # print(f"Percentage is: {percentage} {scores[best_predictions[i]]}, {total_score}")
                    cv2.putText(image, generate_percentage_bar(percentage), (90, 300 + 30 * i + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    
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