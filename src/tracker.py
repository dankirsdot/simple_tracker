import cv2
import numpy as np
from collections import OrderedDict

class Tracker:
    def __init__(self, alive_duration=5):
        self.next_id = 0
        self.boxes = OrderedDict()
        self.tracks = OrderedDict()
        self.alive_counts = OrderedDict()
        self.alive_duration = alive_duration

    def add_object(self, box, center):
        self.boxes[self.next_id] = box
        self.tracks[self.next_id] = center
        self.alive_counts[self.next_id] = self.alive_duration
        self.next_id += 1

    def del_object(self, i):
        del self.boxes[i]
        del self.tracks[i]
        del self.alive_counts[i]

    def iou(self, box_i, box_j):
        x1_i, y1_i, x2_i, y2_i = box_i
        x1_j, y1_j, x2_j, y2_j = box_j
        area_i = (x2_i - x1_i) * (y2_i - y1_i)
        area_j = (x2_j - x1_j) * (y2_j - y1_j)

        x1_inter = max(x1_i, x1_j)
        y1_inter = max(y1_i, y1_j)
        x2_inter = min(x2_i, x2_j)
        y2_inter = min(y2_i, y2_j)
        intersection = max(0, x2_inter - x1_inter) * max(0, y2_inter - y1_inter)

        iou = intersection / (area_i + area_j - intersection)
        return iou

    def update(self, boxes):
        input_boxes = []
        input_centers = []
        
        for box in boxes:
            x, y, w, h = box

            x1 = x - w / 2
            y1 = y - h / 2
            x2 = x1 + w
            y2 = y1 + h

            input_centers.append([x, y])
            input_boxes.append([x1, y1, x2, y2])

        if len(input_centers) == 0:
            for i in list(self.alive_counts.keys()):
                self.alive_counts[i] -= 1
                if self.alive_counts[i] == 0:
                    self.del_object(i)
            return list(self.boxes.values()), list(self.tracks.values())
        
        if len(self.boxes) == 0:
            for i in range(len(input_centers)):
                self.add_object(input_boxes[i], input_centers[i])
        else:
            ids = list(self.boxes.keys())
            boxes = list(self.boxes.values())

            IOU = np.zeros((len(boxes), len(input_boxes)))
            for i in range(IOU.shape[0]):
                for j in range(IOU.shape[1]):
                    IOU[i][j] = self.iou(boxes[i], input_boxes[j])

            rows = IOU.max(axis=1).argsort()
            cols = IOU.argmax(axis=1)[rows]

            used_rows = set()
            used_cols = set()

            for (row, col) in zip(rows, cols):
                if row in used_rows or col in used_cols:
                    continue
                    
                i = ids[row]
                self.boxes[i] = input_boxes[col]
                self.tracks[i] = self.tracks[i] + input_centers[col]
                self.alive_counts[i] = self.alive_duration
                
                used_rows.add(row)
                used_cols.add(col)

            unused_rows = set(range(IOU.shape[0])).difference(used_rows)
            unused_cols = set(range(IOU.shape[1])).difference(used_cols)

            if IOU.shape[0] >= IOU.shape[1]:
                for row in unused_rows:
                    i = ids[row]
                    self.alive_counts[i] -= 1
                    if self.alive_counts[i] == 0:
                        self.del_object(i)
            else:
                for col in unused_cols:
                    self.add_object(input_boxes[col], input_centers[col])

        return list(self.boxes.values()), list(self.tracks.values())