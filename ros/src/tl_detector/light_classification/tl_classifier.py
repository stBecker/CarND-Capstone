from styx_msgs.msg import TrafficLight
import hashlib
import urllib
import os
from collections import namedtuple

import cv2

import tensorflow as tf
import numpy as np

Box = namedtuple('Box', ['score', 'xmin', 'xmax', 'ymin', 'ymax'])
Size = namedtuple('Size', ['w', 'h'])

def decode_boxes(img_size, boxes, scores, threshold):
    """
    Decode the output of the network and convert it to a list of Box objects
    """
    scores = scores[scores > threshold]
    dec_boxes = []
    for i, box in enumerate(boxes[:len(scores)]):
        dec_box = Box(scores[i],
                      int(img_size.w*box[1]), int(img_size.w*box[3]),
                      int(img_size.h*box[0]), int(img_size.h*box[2]))
        dec_boxes.append(dec_box)
    return dec_boxes


class TLClassifier(object):

    UNKNOWN = 0
    GREEN = 1
    YELLOW = 2
    RED = 3

    def __init__(self, datadir):
        self.datadir = datadir
        self.classifier_graph = self.datadir+'/'+"traffic-lights-classifier.pb"
        self.detector_graph = self.datadir+'/'+"traffic-lights-detector-faster-r-cnn.pb"

        self.session = None
        self.det_input = None
        self.det_boxes = None
        self.det_scores = None
        self.class_input = None
        self.class_prediction = None
        self.class_keep_prob = None

        config = tf.ConfigProto(log_device_placement=False)
        config.gpu_options.allow_growth = True
        config.gpu_options.per_process_gpu_memory_fraction = 0.9
        self.session = tf.Session(config=config)
        sess = self.session

        detector_graph_def = tf.GraphDef()
        with open(self.detector_graph, 'rb') as f:
            serialized = f.read()
            detector_graph_def.ParseFromString(serialized)

        classifier_graph_def = tf.GraphDef()
        with open(self.classifier_graph, 'rb') as f:
            serialized = f.read()
            classifier_graph_def.ParseFromString(serialized)

        tf.import_graph_def(detector_graph_def, name='detector')
        self.det_input = sess.graph.get_tensor_by_name('detector/image_tensor:0')
        self.det_boxes = sess.graph.get_tensor_by_name('detector/detection_boxes:0')
        self.det_scores = sess.graph.get_tensor_by_name('detector/detection_scores:0')
        self.detection_classes = sess.graph.get_tensor_by_name('detector/detection_classes:0')

        # tf.import_graph_def(classifier_graph_def, name='classifier')
        # self.class_input = sess.graph.get_tensor_by_name('classifier/data/images:0')
        # self.class_prediction = sess.graph.get_tensor_by_name('classifier/predictions/prediction_class:0')
        # self.class_keep_prob = sess.graph.get_tensor_by_name('classifier/dropout_keep_probability:0')

    def get_classification(self, img):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        sess = self.session
        img_expanded = np.expand_dims(img, axis=0)
        boxes, scores, classes = sess.run([self.det_boxes, self.det_scores, self.detection_classes],
                                 feed_dict={self.det_input: img_expanded})

        if not classes[0]:
            return TrafficLight.UNKNOWN
        else:
            return classes[0][0]

        # img_size = Size(img.shape[1], img.shape[0])
        # detected_boxes = decode_boxes(img_size, boxes[0], scores[0], 0.9)
        #
        # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        #
        # boxes = []
        # for box in detected_boxes:
        #     img_light = img[box.ymin:box.ymax, box.xmin:box.xmax]
        #     img_light = cv2.resize(img_light, (32, 32))
        #     img_light_expanded = np.expand_dims(img_light, axis=0)
        #     light_class = sess.run(self.class_prediction,
        #                            feed_dict={
        #                                self.class_input: img_light_expanded,
        #                                self.class_keep_prob: 1.})
        #     boxes.append((box, light_class[0]))
        #
        # if not boxes:
        #     return self.UNKNOWN
        #
        # return boxes[0][1]
