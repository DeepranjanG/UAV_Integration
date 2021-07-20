#!/usr/bin/env python3

import rospy
import rospkg
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import time

#from PIL import Image

import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils

class ObjectDetector:

    def __init__(self):
        
        self.image_sub = rospy.Subscriber("/front_cam/camera/image", Image ,self.camera_callback)
        self.bridge_object = CvBridge()


        PATH_TO_MODEL_DIR = '/home/deepu/catkin_ws/src/hector_quadrotor_sim/takeoff_land/src/my_model'

        PATH_TO_LABELS = '/home/deepu/catkin_ws/src/hector_quadrotor_sim/takeoff_land/src/label_map.pbtxt'


        MIN_CONF_THRESH = float(0.60)

        PATH_TO_SAVED_MODEL = PATH_TO_MODEL_DIR + "/saved_model"

        print('Loading model...', end='')
        start_time = time.time()

        self.detect_fn = tf.saved_model.load(PATH_TO_SAVED_MODEL)

        end_time = time.time()

        elapsed_time = end_time - start_time
        print('Done! Took {} seconds'.format(elapsed_time))


        self.category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)

    """def load_image_into_numpy_array(self,path):
        return np.array(Image.open(path))"""


    def camera_callback(self, data):

        try:
            cv_image = self.bridge_object.imgmsg_to_cv2(data, desired_encoding="bgr8")
            #print(cv_image)
            #cv2.putText(cv_image, "HII", (15,100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255),2,cv2.LINE_AA)


            print(cv_image.shape)


            image_rgb = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)

            image_expanded = np.expand_dims(image_rgb, axis=0)

            input_tensor = tf.convert_to_tensor(cv_image)
            input_tensor = input_tensor[tf.newaxis, ...]


            detections = self.detect_fn(input_tensor)


            num_detections = int(detections.pop('num_detections'))
            detections = {key: value[0, :num_detections].numpy()
                            for key, value in detections.items()}
            detections['num_detections'] = num_detections


            detections['detection_classes'] = detections['detection_classes'].astype(np.int64)


            #print(detections)

            image_with_detections = cv_image.copy()

            viz_utils.visualize_boxes_and_labels_on_image_array(
                image_with_detections,
                detections['detection_boxes'],
                detections['detection_classes'],
                detections['detection_scores'],
                self.category_index,
                use_normalized_coordinates=True,
                max_boxes_to_draw=200,
                min_score_thresh=0.5,
                agnostic_mode=False)


            print("Done")

            cv2.imshow("Image window", image_with_detections)
            cv2.waitKey(3)
        except CvBridgeError as e:
            print(e)


def main():

    object_detection = ObjectDetector()
    rospy.init_node("model_image")
    rospy.spin()





if __name__ == '__main__':
    main()
