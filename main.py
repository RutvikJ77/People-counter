"""People Counter."""
"""
 Copyright (c) 2018 Intel Corporation.
 Permission is hereby granted, free of charge, to any person obtaining
 a copy of this software and associated documentation files (the
 "Software"), to deal in the Software without restriction, including
 without limitation the rights to use, copy, modify, merge, publish,
 distribute, sublicense, and/or sell copies of the Software, and to
 permit person to whom the Software is furnished to do so, subject to
 the following conditions:
 The above copyright notice and this permission notice shall be
 included in all copies or substantial portions of the Software.
 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
 LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
 OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
 WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""


import os
import time
import logging as log
import sys
import cv2
import socket 
import json
import paho.mqtt.client as mqtt

from argparse import ArgumentParser
from inference import Network

#Mqtt variable intialization
HOSTNAME = socket.gethostname()
IPADDRESS = socket.gethostbyname(HOSTNAME)
MQTT_HOST = IPADDRESS
MQTT_PORT = 3001
MQTT_KEEPALIVE_INTERVAL = 60

def build_parser():
    parser = ArgumentParser()
    parser.add_argument("-m", "--model", required=True, type=str,
                        help="Path to an xml file with a trained model.")
    parser.add_argument("-i", "--input", required=True, type=str,
                        help="Path to image or video file")
    parser.add_argument("-l", "--cpu_extension", required=False, type=str,
                        default=None,
                        help="MKLDNN (CPU)-targeted custom layers."
                             "Absolute path to a shared library with the"
                             "kernels impl.")
    parser.add_argument("-d", "--device", type=str, default="CPU",
                        help="Specify the target device to infer on: "
                             "CPU, GPU, FPGA or MYRIAD is acceptable. Sample "
                             "will look for a suitable plugin for device "
                             "specified (CPU by default)")
    parser.add_argument("-pt", "--prob_threshold", type=float, default=0.5,
                        help="Probability threshold for detections filtering"
                        "(0.5 by default)")
    return parser


def draw_bound(frame,prob_thresh,init_w,init_h,result):
    """
    Draw the bounding box
    :param frame: frame from camera/video
    :param result: list contains the data to parse ssd
    :return: person frame and count
    """
    current_count = 0
    for obj in result[0][0]:
        ## Bounding box for detection
        
        if obj[2] > prob_thresh:
            start = (int(obj[3] * init_w),int(obj[4] * init_h))
            end = (int(obj[5] * init_w),int(obj[6] * init_h))
            frame = cv2.rectangle(frame, start,end,(255,0,0),2)
            current_count = current_count + 1
    return frame,current_count

def connect_mqtt():
    client = mqtt.Client()
    client.connect(MQTT_HOST,MQTT_PORT,MQTT_KEEPALIVE_INTERVAL)
    return client

def infer_on_stream(args,client):
    """Intialise the inference stream showcases the stats of the input image/video
    :args - arguments passed
    :client - connection to MQTT broker
    :return NONE"""  
    c_req_id = 0
    total_count = 0
    start_t = 0
    last_count= 0
    prob_thresh = args.prob_threshold
    single_image = False
    
    infer_network = Network()
    n,c,h,w = infer_network.load_model(args.model,args.device,1,1,c_req_id,args.cpu_extension)[1]

    #If the input is camera
    if args.input == 'CAM':
        input_stream = 0

    # Checks for input image
    elif args.input.endswith('.jpg') or args.input.endswith('.bmp') :
        single_image = True
        input_stream = args.input

    # Checks for video file
    else:
        input_stream = args.input
        assert os.path.isfile(args.input), "Given input file doesn't exist"

    cap = cv2.VideoCapture(input_stream)
    
    if input_stream:
        cap.open(args.input)
    if not cap.isOpened():
        log.error("ERROR! Unable to open the source video")
     
    #Get the initial height/width
    init_w = int(cap.get(3))
    init_h = int(cap.get(4))
       
    while cap.isOpened():
        flag,frame = cap.read()
        if not flag:
            key_press = cv2.waitKey(60)
        #Preprocessing the images/frame
        img = cv2.resize(frame,(h,w))
        
        img = img.transpose((2,0,1))
        img = img.reshape((n,c,h,w))
        
        inference_s = time.time()
        infer_network.exec_net(c_req_id, img)

        if infer_network.wait(c_req_id) == 0:
            
            est_time = time.time() - inference_s
            
            result = infer_network.get_output(c_req_id)

            frame, current_count = draw_bound(frame,prob_thresh,init_w,init_h,result)
            
            inference_t_mess = "Inference time: {:.2f}ms"\.format(est_time * 1000)
            
            cv2.putText(frame, inference_t_mess, (10, 10),cv2.FONT_HERSHEY_COMPLEX, 0.3, (255, 10, 10), 1)
            
            # When new person enters the video
            if current_count > last_count:
                start_t = time.time()
                total_count = total_count + current_count - last_count
                client.publish("person", json.dumps({"total": total_count}))

            # Person duration in the video is calculated
            if current_count < last_count:
                duration = int(time.time() - start_t)
                # Publish messages to the MQTT server
                client.publish("person/duration", json.dumps({"duration":duration}))
                
            client.publish("person", json.dumps({"count": current_count}))
            last_count = current_count
        
        sys.stdout.buffer.write(frame)  
        sys.stdout.flush()
        
        if single_image:
            cv2.imwrite('out.jpg',frame)
            
    cap.release()
    cv2.destroyAllWindows()
    client.disconnect()
    
def main():
    """
    Load the network and parse the output.
    return: None
    """
    
    # Grab command line args
    args = build_parser().parse_args()
    # Connect to the MQTT server
    client = connect_mqtt()
    # Perform inference on the input stream
    infer_on_stream(args, client)


if __name__ == '__main__':
    main()