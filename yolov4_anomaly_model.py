#!/usr/bin/env python3

import rospy
import rospkg
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2

import os
# comment out below line to enable tensorflow outputs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import time
import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

import core.utils as utils
from core.functions import *
from tensorflow.python.saved_model import tag_constants
# from PIL import Image
import numpy as np


class ObjectDetector:

    def __init__(self):
        
        self.image_sub = rospy.Subscriber("/front_cam/camera/image", Image ,self.camera_callback)
        self.bridge_object = CvBridge()

        self.saved_model_loaded = tf.saved_model.load('/home/deepu/catkin_ws/src/hector_quadrotor_sim/takeoff_land/src/yolov4-tiny-anomaly-768/saved_model', tags=[tag_constants.SERVING])
        self.infer = self.saved_model_loaded.signatures['serving_default']



    def camera_callback(self, data):

        try:
            cv_image = self.bridge_object.imgmsg_to_cv2(data, desired_encoding="bgr8")         
            
            frame = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)

            image_data = cv2.resize(frame, (768, 768))
            image_data = image_data / 255.
            image_data = image_data[np.newaxis, ...].astype(np.float32)

            batch_data = tf.constant(image_data)
            pred_bbox = self.infer(batch_data)
            for key, value in pred_bbox.items():
                boxes = value[:, :, 0:4]
                pred_conf = value[:, :, 4:]

            boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
                boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
                scores=tf.reshape(
                    pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
                max_output_size_per_class=50,
                max_total_size=50,
                iou_threshold=0.45,
                score_threshold=0.50
            )

            # format bounding boxes from normalized ymin, xmin, ymax, xmax ---> xmin, ymin, xmax, ymax
            original_h, original_w, _ = frame.shape
            bboxes = utils.format_boxes(boxes.numpy()[0], original_h, original_w)

            pred_bbox = [bboxes, scores.numpy()[0], classes.numpy()[0], valid_detections.numpy()[0]]

            # read in all class names from config
            class_names = {0: 'stealing', 1: 'pickpocketing', 2: 'snatching', 3: 'kidnapping', 4:'running'}

            # by default allow all classes in .names file
            allowed_classes = list(class_names.values())
            # count objects found
            counted_classes = count_objects(pred_bbox, by_class=True, allowed_classes=allowed_classes)
            # loop through dict and print
            for key, value in counted_classes.items():
                print("Number of {}s: {}".format(key, value))
            image = utils.draw_bbox(frame, pred_bbox, False, counted_classes, allowed_classes=allowed_classes,
                                    read_plate=False)

            result = np.asarray(image)
            # result = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


            cv2.imshow("Detection", result)
            cv2.waitKey(3)
        except CvBridgeError as e:
            print(e)


def main():

    object_detection = ObjectDetector()
    rospy.init_node("model_image")
    rospy.spin()





if __name__ == '__main__':
    main()
