import cv2
import numpy as np
from .utils import (
    store_image_on_google_drive,
    FuzzyComparator,
    detect_words,
    get_similar_words_vertices,

)
from .vision import OCRClient
from werkzeug.datastructures import FileStorage



class IUtils():
    def __init__(self, image_content):
        self.image_content = image_content
        self.vision_client = False
        self.rotated_img = None
        self.start_x, self.start_y = 0, 0


    def call_vision(self):
        self.image = cv2.imread(self.image_content)
        with open(self.image_content, 'rb') as bytefile:
            self.vision_client = OCRClient(FileStorage(bytefile))

    def get_straightening_reference_bounds(self, detected_words, keywords):
        for keyword in keywords:
            if keyword in FuzzyComparator(detected_words) and len(keyword) >= 5:
                vertices = get_similar_words_vertices(self.response.text_annotations, FuzzyComparator([keyword]))
                if vertices:
                    return vertices

    def get_rotation_angle(self, reference_bounds):
        angle_to_rot_img = np.arctan2(
            reference_bounds[0].vertices[0].y - reference_bounds[-1].vertices[1].y,
            reference_bounds[0].vertices[0].x - reference_bounds[-1].vertices[1].x, )
        horizontal_angle = np.arctan2(
            reference_bounds[0].vertices[0].y - reference_bounds[0].vertices[0].y,
            reference_bounds[0].vertices[0].x - reference_bounds[-1].vertices[1].x, )
        return np.degrees(angle_to_rot_img) - np.degrees(horizontal_angle)

    def flip_if_reverse(self, ref_set1_boundpoly):
        horizontal_angle = np.arctan2(
            ref_set1_boundpoly[0].vertices[0].y -
            ref_set1_boundpoly[0].vertices[0].y,
            ref_set1_boundpoly[0].vertices[0].x
            - ref_set1_boundpoly[len(ref_set1_boundpoly) - 1].vertices[1].x,
        )
        if np.degrees(horizontal_angle) == 0.0:
            return True

    def rotate_box(self, bb, cx, cy, h, w, theta):
        new_bb = list(bb)
        for i, coord in enumerate(bb):
            M = cv2.getRotationMatrix2D((cx, cy), theta, 1.0)

            cos = np.abs(M[0, 0])
            sin = np.abs(M[0, 1])

            nW = int((h * sin) + (w * cos))
            nH = int((h * cos) + (w * sin))

            M[0, 2] += (nW / 2) - cx
            M[1, 2] += (nH / 2) - cy

            v = [coord[0], coord[1], 1]

            calculated = np.dot(M, v)
            new_bb[i] = (calculated[0], calculated[1])
        return new_bb

    def rotate_bound(self, image, angle):
        # grab the dimensions of the image and then determine the
        # centre
        (h, w) = image.shape[:2]
        (cX, cY) = (w // 2, h // 2)

        # grab the rotation matrix (applying the negative of the
        # angle to rotate clockwise), then grab the sine and cosine
        # (i.e., the rotation components of the matrix)
        M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])

        # compute the new bounding dimensions of the image
        nW = int((h * sin) + (w * cos))
        nH = int((h * cos) + (w * sin))

        # adjust the rotation matrix to take into account translation
        M[0, 2] += (nW / 2) - cX
        M[1, 2] += (nH / 2) - cY

        # perform the actual rotation and return the image
        return cv2.warpAffine(image, M, (nW, nH))

    def rotate_response_bounds(self, image, doc_page_id):
        self.response = self.vision_client.response

        detected_words = FuzzyComparator(detect_words(self.response.full_text_annotation.text))
        if not detected_words:
            return {"status":"NOT_PROCESSED", "status_code": "412", "error_message": "Error reading image file"}

        reference_bounds = self.get_straightening_reference_bounds(detected_words, ['REQUISITION','FORM','Healthcare','Patient'])
        if not reference_bounds:
            self.rotated_img = self.image
            return {"status":"NOT_PROCESSED","status_code": "412",  "error_message" : "Poor/Invalid/Not Supported. Cannot identify keywords"}

        theta = self.get_rotation_angle(reference_bounds)

        if self.flip_if_reverse(reference_bounds):
            theta += 180
        # --------------------- TEXT ANNOTATIONS ---------------------?
        bb1 = {}
        for i, doc in enumerate(self.response.text_annotations):
            x0 = doc.bounding_poly.vertices[0].x
            y0 = doc.bounding_poly.vertices[0].y

            x1 = doc.bounding_poly.vertices[1].x
            y1 = doc.bounding_poly.vertices[1].y

            x2 = doc.bounding_poly.vertices[2].x
            y2 = doc.bounding_poly.vertices[2].y

            x3 = doc.bounding_poly.vertices[3].x
            y3 = doc.bounding_poly.vertices[3].y
            bb1[i] = [(float(x0), float(y0)), (float(x1), float(y1)),
                      (float(x2), float(y2)), (float(x3), float(y3))]

        # img_orig = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
        img_orig = image
        self.rotated_img = self.rotate_bound(img_orig, theta)

        (heigth, width) = img_orig.shape[:2]
        (cx, cy) = (width // 2, heigth // 2)
        (new_height, new_width) = self.rotated_img.shape[:2]
        (new_cx, new_cy) = (new_width // 2, new_height // 2)
        # print(bb1)
        for b in bb1:
            # if b == 0:
            #     continue
            new_coordinates = self.rotate_box(bb1[b], cx, cy, heigth, width, theta)
            for n, corner in enumerate(new_coordinates):
                self.response.text_annotations[b].bounding_poly.vertices[n].x = int(corner[0])
                self.response.text_annotations[b].bounding_poly.vertices[n].y = int(corner[1])

        # --------------------- FULL TEXT ANNOTATIONS ---------------------?
        bb1 = []
        for p, page in enumerate(self.response.full_text_annotation.pages):
            for b, block in enumerate(page.blocks):
                for pa, paragraph in enumerate(block.paragraphs):
                    for w, word in enumerate(paragraph.words):
                        for d, doc in enumerate(word.symbols):
                            x0 = doc.bounding_box.vertices[0].x
                            y0 = doc.bounding_box.vertices[0].y

                            x1 = doc.bounding_box.vertices[1].x
                            y1 = doc.bounding_box.vertices[1].y

                            x2 = doc.bounding_box.vertices[2].x
                            y2 = doc.bounding_box.vertices[2].y

                            x3 = doc.bounding_box.vertices[3].x
                            y3 = doc.bounding_box.vertices[3].y
                            bb1.append([p, b, pa, w, d, [(float(x0), float(y0)), (float(x1), float(y1)),
                                                         (float(x2), float(y2)), (float(x3), float(y3))]])
        # print(bb1)
        for i, b in enumerate(bb1):
            new_coordinates = self.rotate_box(b[-1], cx, cy, heigth, width, theta)
            for n, corner in enumerate(new_coordinates):
                self.response.full_text_annotation.pages[b[0]].blocks[b[1]].paragraphs[b[2]].words[b[3]].symbols[b[4]].bounding_box.vertices[n].x = int(corner[0])
                self.response.full_text_annotation.pages[b[0]].blocks[b[1]].paragraphs[b[2]].words[b[3]].symbols[b[4]].bounding_box.vertices[n].y = int(corner[1])

        return False

