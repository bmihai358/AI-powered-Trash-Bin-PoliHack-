from inference_sdk import InferenceHTTPClient
import cv2
import cvzone
import time
import socket
import threading


ARDUINO_IP = "192.168.43.224"
ARDUINO_PORT = 80


def send_signal_thread(command):
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(3.0)
            s.connect((ARDUINO_IP, ARDUINO_PORT))
            s.sendall(command)
            print(f" Signal Sent: {command}")
    except Exception as e:
        print(f" Wi-Fi Error: {e}")



cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
if not cap.isOpened():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

cap.set(3, 1280)
cap.set(4, 720)


API_KEY = "EieUNhS7oWih01ZtjYql"
MODEL_ID = "recyclable-materials-ljuil/2"

client = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key=API_KEY
)


recyclable_list = [
    'Bottle', 'Can', 'Plastic', 'Glass', 'Metal',
    'bottle', 'can', 'Other Plastics', 'Aluminium'
]

trash_list = [
    'Trash', 'Waste', 'Organic Waste', 'Food', 'Garbage',
    'trash', 'waste', 'organic', 'food', 'garbage'
]

ignore_list = ['paper', 'cardboard', 'Paper', 'Cardboard']


last_sent_time = 0
COOLDOWN = 1
last_inference_time = 0
INFERENCE_INTERVAL = 1.5


current_label = "Scanning..."
current_color = (255, 0, 255)

print("Starting Smart Bin (Cloud Wi-Fi Mode)...")

while True:
    success, img = cap.read()
    if not success:
        time.sleep(0.1)
        continue


    if time.time() - last_inference_time > INFERENCE_INTERVAL:
        try:

            result = client.infer(img, model_id=MODEL_ID)
            predictions = result['predictions']

            top_prediction = None

            all_preds = []
            if isinstance(predictions, list):
                all_preds = predictions
            elif isinstance(predictions, dict) and 'class' in predictions:
                all_preds = [predictions]

            for pred in all_preds:
                name = pred['class']
                if name in ignore_list:
                    continue
                top_prediction = pred
                break

            if top_prediction:
                currentClass = top_prediction['class']
                conf = top_prediction['confidence']

                print(f"☁ Cloud AI: {currentClass} ({int(conf * 100)}%)")

                if conf > 0.4:
                    if currentClass in recyclable_list:
                        current_label = f"RECYCLABLE ({currentClass})"
                        current_color = (0, 255, 0)  # Green

                        if (time.time() - last_sent_time > COOLDOWN):
                            threading.Thread(target=send_signal_thread, args=(b'0',)).start()
                            last_sent_time = time.time()

                    elif currentClass in trash_list:
                        current_label = f"TRASH ({currentClass})"
                        current_color = (0, 0, 255)  # Red

                        if (time.time() - last_sent_time > COOLDOWN):
                            threading.Thread(target=send_signal_thread, args=(b'1',)).start()
                            last_sent_time = time.time()
                    else:
                        current_label = currentClass
                        current_color = (255, 0, 255)  # Purple
                else:
                    current_label = "Scanning..."
                    current_color = (200, 200, 200)

            last_inference_time = time.time()

        except Exception as e:
            print(f"⚠️ Cloud Error: {e}")


    cvzone.putTextRect(img, current_label, (50, 50), scale=3, thickness=3, colorR=current_color)

    cv2.imshow("Smart Bin AI", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()