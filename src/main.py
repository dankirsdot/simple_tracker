import os
import argparse

import cv2
import numpy as np
from tracker import Tracker
from inference import Network

def preprocess(image, height, width):
    image = cv2.resize(image, (height, width)) # to net input size
    image = np.transpose(image, (2, 0, 1))     # from H,W,C view to C,H,W
    image = np.expand_dims(image, axis=0)      # from C,H,W view to B,C,H,W
    return image

def postprocess(output, anchors, n_cells, n_classes):
    n_anchors = anchors.shape[0] // 2

    # tuning constants
    class_conf_thresh = 0.8
    score_thresh = 0.95
    nms_thresh = 0.0 

    boxes = []
    confidences = []

    # decode yolo output
    for cx in range(n_cells):
        for cy in range(n_cells):
            for b in range(n_anchors):
                n = b * (n_classes + 5)
                
                confidence = output[0, n + 4, cx, cy]
                class_prob = output[0, n + 5 : n + 5 + n_classes, cx, cy]
                class_confidence = class_prob * confidence

                detected_class = np.argmax(class_prob)

                if class_confidence[detected_class] > class_conf_thresh:
                    # get relative coordinates
                    tx = output[0, n    , cx, cy]
                    ty = output[0, n + 1, cx, cy]
                    th = output[0, n + 2, cx, cy]
                    tw = output[0, n + 3, cx, cy]

                    # calculate actual coordinates
                    x = (cx + tx) * 16
                    y = (cy + ty) * 16
                    w = np.exp(tw) * anchors[2*b    ]
                    h = np.exp(th) * anchors[2*b + 1]
                    y, x, h, w = x, y, w, h

                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
    
    # non-maximum suppression
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, score_thresh, nms_thresh)
    idxs = np.asarray(idxs).flatten()
    return [boxes[i] for i in idxs]


def main(model_path, anchors, video_path):
    path_xml = model_path + ".xml"
    path_bin = model_path + ".bin"

    # load model and get its parameters
    net = Network()
    net.load_network(path_xml, path_bin)
    _, _, net_h, net_w = net.get_input_shape()
    _, n, _, n_cells = net.get_output_shape()
    n_anchors = anchors.shape[0] // 2
    n_classes = n // n_anchors - 5

    # initialize tracker
    tracker = Tracker()

    # open input video stream
    video = cv2.VideoCapture(video_path)
    video_h = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_w = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    fps = video.get(cv2.CAP_PROP_FPS)

    # open output video stream
    fourcc = cv2.VideoWriter_fourcc(*'MPEG')
    output = cv2.VideoWriter('result.avi', fourcc, fps, (video_w, video_h))

    # remember ratio to scale bounding boxes back
    x_scale = video_w / net_w
    y_scale = video_h / net_h

    while(video.isOpened()):
        ret, image = video.read()
        if ret == True:
            net_image = preprocess(image, net_h, net_w)
            predictions = net.sync_inference(net_image)
            boxes = postprocess(predictions, anchors, n_cells, n_classes)
            boxes, tracks = tracker.update(boxes)
            
            # plot boxes and tracks
            for i in range(len(boxes)):
                x1, y1, x2, y2 = boxes[i]
                # scale bounding boxes to output size
                x1, x2 = int(x1 * x_scale), int(x2 * x_scale)
                y1, y2 = int(y1 * y_scale), int(y2 * y_scale)
                image = cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

                track = np.array(tracks[i]).reshape(-1, 2)
                track[:, 0] = track[:, 0] * x_scale
                track[:, 1] = track[:, 1] * y_scale
                for j in range(track.shape[0] - 1):
                    x1, y1 = track[j, :]
                    x1, y1 = int(x1), int(y1)
                    x2, y2 = track[j + 1, :]
                    x2, y2 = int(x2), int(y2)
                    image = cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

            output.write(image)
            # cv2.imwrite("image.jpg", image) # debug line
        else:
            break
    
    video.release()
    output.release()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = 'Simple vehicle detector')
    parser.add_argument('-n', '--name', default='yolo-v3-tiny-tf', help='model name')
    parser.add_argument('-f', '--folder', default='/home/openvino/simple_tracker/', help='path to the models folder')
    parser.add_argument('-p', '--precision', default='FP32', help='model precision')
    parser.add_argument('-v', '--video', default='/home/openvino/simple_tracker/video.mp4', help='video to process')
    args = parser.parse_args()

    model_path = os.path.join(args.folder, args.name, args.precision, args.name)
    
    anchors = np.array([23,27, 37,58, 81,82])

    main(model_path, anchors, args.video)
