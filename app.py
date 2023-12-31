import replicate
import cv2
import time
import io
import requests
from smtplib import SMTP
from dotenv import load_dotenv
import os
import numpy as np
import math
from constants import Q_50
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage

load_dotenv()


def send_sms(message):
    CARRIERS = {
        "att": "@mms.att.net",
        "tmobile": "@tmomail.net",
        "verizon": "@vtext.com",
        "sprint": "@messaging.sprintpcs.com"
    }
    email = "aryopatel@gmail.com"
    password = os.getenv('GOOGLE_APP_PASSWORD')

    recipient = "2242160562" + CARRIERS["tmobile"]
    auth = (email, password)
 

 
    msg = MIMEMultipart()
    msg['From'] = email
    msg['To'] = recipient
    msg['Subject'] = 'There are dishes in the sink'

    body = message
    with open('dirty_dishes.jpg', 'rb') as image_file:
        image = MIMEImage(image_file.read())
    image.add_header('Content-Disposition', 'attachment; filename="dirty_dishes.jpeg"')
    msg.attach(image)
    msg.attach(MIMEText(body, 'plain'))
    
    server = SMTP("smtp.gmail.com", 587)
    server.starttls()
    server.login(auth[0], auth[1])
    server.send_message(msg)
    server.quit()

def play_video():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open camera")
        exit()

    start_time = time.time()
    while True:

        ret, frame = cap.read()

        if not ret:
            print("Error: Can't receive frame. Exiting ...")
            break

        cv2.imshow('Camera Feed', frame)

        curr_time = time.time()

        if curr_time - start_time > 300:
            # print("should send photo")
            start_time = curr_time
            cv2.imwrite("dirty_dishes.jpg", frame)
            _, buffer = cv2.imencode(".jpg", frame)
            io_buf = io.BytesIO(buffer)

            output = replicate.run(
                "cudanexus/detic:674d269c2b2867415fa4de13e85befde4f2debe7dc5823ec01ea95eb11eff72f",
                input={
                    "image": io_buf,
                    "vocabulary": "lvis",
                    "custom_vocabulary": "None"
                }
            )
            print(output)
            process_output(output)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def check_is_contained(bb_in, bb_out):
    """
    bounding boxes are returned as a list of 4 elements, where the first two elements are x/y of the upper left corner and the last two 
    are the x/y of the lower right corner
    """
    return bb_in[0] < bb_out[0] and bb_in[1] < bb_out[1] and bb_in[2] > bb_out[2] and bb_in[3] > bb_out[3]

dish_objects = {"cup", "spoon", "fork", "knife", "food", "plate", "bowl", "pan"}

def process_output(output):
    json_url = output["jsona"]

    json_data = requests.get(json_url).json()

    sink_bounding_box = None

    objects = []
    
    # isolate sink bounding box + boudning box for other objects
    for i, class_name in enumerate(json_data["class_names"]):
        if class_name == "sink":
            sink_bounding_box = json_data["pred_boxes"][i]
        
        else:
            objects.append((class_name, json_data["pred_boxes"][i]))
    
    if not sink_bounding_box:
        return
    
    filtered_objs = list(filter(lambda x: check_is_contained(sink_bounding_box, x[1]), objects))
    print(filtered_objs)
    if len(filtered_objs) > 0:
        send_sms("DISHES NEED TO BE CLEANED 🚨")


def DCT(content):
    image_array = np.asarray(bytearray(content), dtype=np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    cv2.imwrite("original_image.jpg", image)
    image = image - 128
    CHUNK_SIZE = 8
    T = np.full((CHUNK_SIZE, CHUNK_SIZE), None)
    for i in range(CHUNK_SIZE):
        for j in range(CHUNK_SIZE):
            if i == 0:
                T[i][j] = 1/(CHUNK_SIZE**0.5)
            else:
                T[i][j] = (2/CHUNK_SIZE)**0.5 * math.cos(((2 * j + 1) * i * math.pi)/(2 * CHUNK_SIZE))
    
    T_t = T.transpose()
    
    new_image = np.zeros_like(image)
    # image shape is 1080 x 1920 x 3
    # create 8x8 chunk
    i = 0
    j = 0
    while i < image.shape[0]:
        while j < image.shape[1]:
            chunk = image[i : i + CHUNK_SIZE,j : j + CHUNK_SIZE,:]
            result = np.zeros_like(chunk)
            # print(T)
            # print("-----")
            # print(T_t)
            # print("K shapes")
            for k in range(chunk.shape[2]):
                # print(chunk[:, :, k])
                result[:, :, k] = T.dot(chunk[:, :, k]).dot(T_t)
            # print("------ result info")
            # for k in range(result.shape[2]):
            #     print(result[:, :, k])
            # print(result.shape)
            # break
            N = np.zeros_like(result)
            for k in range(result.shape[2]):
                tmp  = (result[:,:, k]/Q_50).round()
                unscaled = T_t.dot(tmp).dot(T).astype(np.double)
                N[: , :, k] = (unscaled.round() + 128)
            
            new_image[i: i + CHUNK_SIZE, j : j + CHUNK_SIZE, :] = N
            # print(N)
            # break
            j += CHUNK_SIZE
        # break
        i += CHUNK_SIZE
    
    print(new_image[:, :, 0])
    cv2.imwrite("compressed_image.jpg", new_image)
    

if __name__ == "__main__":
    play_video()
    # output = {'image': 'https://replicate.delivery/pbxt/aE3yr9Wo3I4nK14wcekLCjDl8fSl1JboxfwAywEB6xV6hGPkA/out.png', 'jsona': 'https://replicate.delivery/pbxt/ddR3Dgvf2iS8Rq157pOOof5opEeS6zfSMJpUURfHPt71Ha8QC/output.json'}

    # body = requests.get(output["image"])
    # DCT(content=body.content)