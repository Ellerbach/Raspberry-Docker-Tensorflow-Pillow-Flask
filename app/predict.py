
from urllib.request import urlopen
from datetime import datetime

import tensorflow as tf

from PIL import Image
import numpy as np
import sys
import os  

filename = 'model.pb'
labels_filename = 'labels.txt'

network_input_size = 227

output_layer = 'loss:0'
input_node = 'Placeholder:0'

graph_def = tf.GraphDef()
labels = []

def initialize():
    print('Loading model...',end=''),
    with tf.gfile.FastGFile(filename, 'rb') as f:
        graph_def.ParseFromString(f.read())
        tf.import_graph_def(graph_def, name='')
    print('Success!')
    print('Loading labels...', end='')
    with open(labels_filename, 'rt') as lf:
        for l in lf:
            labels.append(l.strip())
    print(len(labels), 'found. Success!')

def log_msg(msg):
    print("{}: {}".format(datetime.now(),msg))

def crop_center(img,cropx,cropy):
    w,h = img.size
    startx = w//2-(cropx//2)
    starty = h//2-(cropy//2)
    log_msg("crop_center: " + str(w) + "x" + str(h) +" to " + str(cropx) + "x" + str(cropy))
    return img.crop((startx, starty, startx+cropx, starty+cropy))

def resize_down_to_1600_max_dim(image):
    w,h = image.size
    if (h < 1600 and w < 1600):
        return image

    new_size = (1600 * w // h, 1600) if (h > w) else (1600, 1600 * h // w)
    log_msg("resize: " + str(w) + "x" + str(h) + " to " + str(new_size[0]) + "x" + str(new_size[1]))
    return image.resize(new_size, Image.BILINEAR)

def resize_to_256_square(image):
    w,h = image.size
    log_msg("resize: " + str(w) + "x" + str(h) + " to 256x256")
    return image.resize((256, 256), Image.BILINEAR)

def predict_url(imageUrl):
    log_msg("Predicting from url: " +imageUrl)
    with urlopen(imageUrl) as testImage:
        image = Image.open(testImage)
        return predict_image(image)

def predict_image(image):
    tf.reset_default_graph()
    tf.import_graph_def(graph_def, name='')

    with tf.Session() as sess:
        prob_tensor = sess.graph.get_tensor_by_name(output_layer)
    
        log_msg('Predicting image')
        bgr_image = None
        try:
            if(image.mode != "RGB"):
                log_msg("Converting to RGB")
                image.convert("RGB")

            w,h = image.size
            log_msg("Image size: " + str(w) + "x" + str(h))

            # If the image has either w or h greater than 1600 we resize it down respecting
            # aspect ratio such that the largest dimention is 1600
            image = resize_down_to_1600_max_dim(image)

            # We next get the largest center square
            w,h = image.size
            min_dim = min(w,h)
            max_square_image = crop_center(image, min_dim, min_dim)

            # Resize that square down to 256x256
            augmented_image = resize_to_256_square(max_square_image)

            # Crop the center for the specified network_input_Size
            augmented_image = crop_center(augmented_image, network_input_size, network_input_size)

            # RGB -> BGR
            r,g,b = np.array(augmented_image).T
            bgr_image = np.array([b,g,r]).transpose()

        except Exception as e:
            log_msg(str(e))
            return 'Error: Could not preprocess image for prediction. ' + str(e)

        predictions, = sess.run(prob_tensor, {input_node: [bgr_image] })

        result = []
        idx = 0
        for p in predictions:
            truncated_probablity = np.float64(round(p,8))
            if (truncated_probablity > 1e-8):
                result.append({
                    'tagName': labels[idx],
                    'probability': truncated_probablity,
                    'tagId': '',
                    'boundingBox': None })
            idx += 1

        response = { 
            'id': '',
            'project': '',
            'iteration': '',
            'created': datetime.utcnow().isoformat(),
            'predictions': result 
        }

        log_msg("Results: " + str(response))
        return response
