import os
import cv2
import sys
import tensorflow as tf
import numpy as np

from styx_msgs.msg import TrafficLight


TRAFFIC_LIGHT_CLASS = 10
TRAFFIC_LIGHT_MIN_SCORE = 0.80


class TLClassifier(object):
    def __init__(self):
        script_dir = os.getcwd()
        research_dir = os.path.join(script_dir, "research")
        slim_dir = os.path.join(research_dir, "slim")

        if research_dir not in sys.path:
            sys.path.append(research_dir)

        if slim_dir not in sys.path:
            sys.path.append(slim_dir)

        # Load Frozen Model
        frozen_model_file = "./model/frozen_inference_graph.pb"
        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(frozen_model_file, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

        # Load labels
        from object_detection.utils import label_map_util
        current_dir = os.getcwd()
        labels_file = os.path.join(current_dir, "model/mscoco_label_map.pbtxt")
        self.category_index = label_map_util.create_category_index_from_labelmap(labels_file, use_display_name=True)

    def run_inference_for_single_image(self, image):
        from object_detection.utils import ops as utils_ops

        with self.detection_graph.as_default():
            with tf.Session() as sess:
                # Get handles to input and output tensors
                ops = tf.get_default_graph().get_operations()
                all_tensor_names = {output.name for op in ops for output in op.outputs}
                tensor_dict = {}
                for key in [
                    'num_detections', 'detection_boxes', 'detection_scores', 'detection_classes', 'detection_masks'
                ]:
                    tensor_name = key + ':0'
                    if tensor_name in all_tensor_names:
                        tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(tensor_name)

                if 'detection_masks' in tensor_dict:
                    # The following processing is only for single image
                    detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
                    detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
                    # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
                    real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
                    detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
                    detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
                    detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
                        detection_masks, detection_boxes, image.shape[0], image.shape[1])
                    detection_masks_reframed = tf.cast(
                        tf.greater(detection_masks_reframed, 0.5), tf.uint8)
                    # Follow the convention by adding back the batch dimension
                    tensor_dict['detection_masks'] = tf.expand_dims(
                        detection_masks_reframed, 0)

                image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

                # Run inference
                output_dict = sess.run(tensor_dict, feed_dict={image_tensor: np.expand_dims(image, 0)})

                # all outputs are float32 numpy arrays, so convert types as appropriate
                output_dict['num_detections'] = int(output_dict['num_detections'][0])
                output_dict['detection_classes'] = output_dict[
                    'detection_classes'][0].astype(np.uint8)
                output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
                output_dict['detection_scores'] = output_dict['detection_scores'][0]
                if 'detection_masks' in output_dict:
                    output_dict['detection_masks'] = output_dict['detection_masks'][0]
            return output_dict

    def get_traffic_light(self, cv2_image, output_dict):
        height, width, _ = cv2_image.shape

        traffic_light_score = 0
        traffic_light_index = None

        contrains_traffic_light = np.nonzero(output_dict["detection_classes"] == TRAFFIC_LIGHT_CLASS)
        traffic_light_indexes = contrains_traffic_light[0]

        for tf_light_index in traffic_light_indexes:
            current_score = output_dict['detection_scores'][tf_light_index]
            if current_score >= TRAFFIC_LIGHT_MIN_SCORE and current_score >= traffic_light_score:
                traffic_light_score = current_score
                traffic_light_index = tf_light_index

        # No traffic light has been found
        if traffic_light_index is None:
            return None

        # Traffic light found
        detection_box = output_dict["detection_boxes"][traffic_light_index]
        (ymin, xmin, ymax, xmax) = (int(detection_box[0] * height), int(detection_box[1] * width),
                                    int(detection_box[2] * height), int(detection_box[3] * width))
        return cv2_image[ymin:ymax, xmin:xmax]

    def get_classification(self, cv2_image):
        """Determines the color of the traffic light in the image

        Args:
            cv2_image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        def get_green_mask(img_hsv):
            lower_green = np.array([40, 10, 10])
            upper_green = np.array([90, 255, 255])
            mask = cv2.inRange(img_hsv, lower_green, upper_green)
            return mask

        def get_red_mask(img_hsv):
            # red lower mask (0-10)
            lower_red = np.array([20, 1, 150])
            upper_red = np.array([30, 120, 255])
            mask0 = cv2.inRange(img_hsv, lower_red, upper_red)

            # Red upper mask
            lower_red = np.array([170, 50, 50])
            upper_red = np.array([180, 255, 255])
            mask1 = cv2.inRange(img_hsv, lower_red, upper_red)

            # join my masks
            mask = mask0 + mask1
            return mask

        def get_traffic_light_color(cv2_image):
            # Convert BGR to HSV
            img_hsv = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2HSV)
            height, width, _ = img_hsv.shape

            green_mask = get_green_mask(img_hsv)
            red_mask = get_red_mask(img_hsv)

            dico = {
                TrafficLight.RED: np.count_nonzero(red_mask[0:int(height / 3), :]),
                TrafficLight.YELLOW: np.count_nonzero(red_mask[int(height / 3):int(height * 2 / 3), :]),
                TrafficLight.GREEN: np.count_nonzero(green_mask[int(height * 2 / 3):height, :])
            }

            v = list(dico.values())
            k = list(dico.keys())
            return k[v.index(max(v))]

        output_dict = self.run_inference_for_single_image(cv2_image)
        traffic_light_image = self.get_traffic_light(cv2_image, output_dict)

        # no traffic light found
        if traffic_light_image is None:
            return TrafficLight.UNKNOWN

        return get_traffic_light_color(traffic_light_image)
