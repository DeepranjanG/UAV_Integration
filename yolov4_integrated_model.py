#!/usr/bin/env python3

import rospy
import rospkg
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2
import time

class ObjectDetector:

    def __init__(self):
        
        self.image_sub = rospy.Subscriber("/front_cam/camera/image", Image ,self.camera_callback)
        self.bridge_object = CvBridge()        



    def camera_callback(self, data):

        try:
            cv_image = self.bridge_object.imgmsg_to_cv2(data, desired_encoding="bgr8")

            CONFIDENCE_THRESHOLD = 0.2
            NMS_THRESHOLD = 0.4

            class_names = ['human']

            net = cv2.dnn.readNet("/home/deepu/catkin_ws/src/hector_quadrotor_sim/takeoff_land/src/yolov4-tiny.weights", "/home/deepu/catkin_ws/src/hector_quadrotor_sim/takeoff_land/src/yolov4-tiny.cfg")
            net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)
            

            yolo = cv2.dnn_DetectionModel(net)
            yolo.setInputParams(size=(416, 416))

            classes, scores, boxes = yolo.detect(cv_image, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)
            print(classes, scores, boxes)

            for (classid, score, box) in zip(classes, scores, boxes):
                label = "%s : %f" % (class_names[classid[0]], score)
                cv2.rectangle(cv_image, box, 2)
                cv2.putText(cv_image, label, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            end_drawing = time.time()            

            cv2.imshow("Image window", cv_image)
            cv2.waitKey(3)
        except CvBridgeError as e:
            print(e)


def main():

    object_detection = ObjectDetector()
    rospy.init_node("model_image")
    rospy.spin()





if __name__ == '__main__':
    main()