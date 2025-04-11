import cv2 as cv
import numpy as np
import itertools
import copy
import csv
from tensorflow import keras
from mediapipe import solutions

def select_mode(key, mode):
    number = -1
    if 48 <= key <= 57:  # Keys 0–9
        number = key - 48
    if key == ord('n'):
        mode = 0  # Prediction
    if key == ord('k'):
        mode = 1  # Data collection
    return number, mode

def calc_bounding_rect(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]
    landmark_array = np.array([[min(int(l.x * image_width), image_width - 1),
                                min(int(l.y * image_height), image_height - 1)] for l in landmarks.landmark])
    x, y, w, h = cv.boundingRect(landmark_array)
    return [x, y, x + w, y + h]

def calc_landmark_list(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]
    return [[min(int(l.x * image_width), image_width - 1),
             min(int(l.y * image_height), image_height - 1)] for l in landmarks.landmark]

def pre_process_landmark(landmark_list):
    temp_landmark_list = copy.deepcopy(landmark_list)
    base_x, base_y = temp_landmark_list[0]
    temp_landmark_list = [[x - base_x, y - base_y] for x, y in temp_landmark_list]
    temp_landmark_list = list(itertools.chain.from_iterable(temp_landmark_list))
    max_value = max(map(abs, temp_landmark_list))
    return [n / max_value for n in temp_landmark_list] if max_value != 0 else temp_landmark_list

def draw_landmarks(image, landmark_point):
    connections = [(2, 3), (3, 4), (5, 6), (6, 7), (7, 8), (9, 10), (10, 11), (11, 12),
                   (13, 14), (14, 15), (15, 16), (17, 18), (18, 19), (19, 20), (0, 1), (1, 2),
                   (2, 5), (5, 9), (9, 13), (13, 17), (17, 0)]
    for p1, p2 in connections:
        cv.line(image, tuple(landmark_point[p1]), tuple(landmark_point[p2]), (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[p1]), tuple(landmark_point[p2]), (255, 255, 255), 2)
    for i, (x, y) in enumerate(landmark_point):
        size = 8 if i in [4, 8, 12, 16, 20] else 5
        cv.circle(image, (x, y), size, (255, 255, 255), -1)
        cv.circle(image, (x, y), size, (0, 0, 0), 1)
    return image

def draw_bounding_rect(image, brect):
    cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[3]), (0, 0, 0), 1)
    return image

def draw_info_text(image, brect, handedness, hand_sign_text):
    cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[1] - 22), (0, 0, 0), -1)
    info_text = handedness.classification[0].label
    if hand_sign_text:
        info_text += f': {hand_sign_text}'
    cv.putText(image, info_text, (brect[0] + 5, brect[1] - 4),
               cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv.LINE_AA)
    return image

def load_labels(label_path):
    with open(label_path, "r") as f:
        return [row[0] for row in csv.reader(f)]

def main():
    model_path = input("Enter the path to the model (.keras file): ")
    label_path = input("Enter the path to the label CSV file: ")
    data_csv_path = "captured_signs.csv"

    model = keras.models.load_model(model_path)
    labels = load_labels(label_path)
    cap = cv.VideoCapture(0)
    mp_hands = solutions.hands
    hands = mp_hands.Hands()

    mode = 0  # Start in prediction mode
    selected_label = None

    print("Press 'k' to start capturing signs, 'n' to return to prediction, and 'e' to stop capturing.")
    print("Use number keys (0–9) to select label. Press 's' to save a sample.")

    while cap.isOpened():
        ret, image = cap.read()
        if not ret:
            break
        image = cv.flip(image, 1)
        results = hands.process(cv.cvtColor(image, cv.COLOR_BGR2RGB))

        key = cv.waitKey(1) & 0xFF
        number, new_mode = select_mode(key, mode)

        if new_mode != mode:
            mode = new_mode
            if mode == 0:
                print("Switched to Prediction Mode")
            elif mode == 1:
                print("Switched to Data Collection Mode")

        if number != -1 and mode == 1:
            selected_label = number
            print(f"Label selected: {selected_label}")

        if key == ord('e') and mode == 1:
            print("Stopped capturing for this sign.")
            selected_label = None
            mode = 0

        if results.multi_hand_landmarks:
            for landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                brect = calc_bounding_rect(image, landmarks)
                landmark_list = calc_landmark_list(image, landmarks)
                processed_landmarks = pre_process_landmark(landmark_list)

                if mode == 0:
                    prediction = model.predict(np.array([processed_landmarks]), verbose=0)
                    hand_sign_text = labels[np.argmax(prediction)]
                    image = draw_bounding_rect(image, brect)
                    image = draw_landmarks(image, landmark_list)
                    image = draw_info_text(image, brect, handedness, hand_sign_text)

                elif mode == 1 and selected_label is not None:
                    image = draw_bounding_rect(image, brect)
                    image = draw_landmarks(image, landmark_list)
                    image = draw_info_text(image, brect, handedness, f"Capturing: {selected_label}")

                    if key == ord('s'):
                        with open(data_csv_path, mode='a', newline='') as f:
                            writer = csv.writer(f)
                            writer.writerow([selected_label] + processed_landmarks)
                        print(f"Saved one sample for label {selected_label}")

        cv.imshow('Hand Gesture Recognition', image)

        if key == 27:  # ESC to exit
            break

    cap.release()
    cv.destroyAllWindows()

if __name__ == '__main__':
    main()
