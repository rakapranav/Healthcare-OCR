import datetime
import glob
import locale
import os
import io
import random
import re
import string
import tempfile
from collections import OrderedDict
from collections import defaultdict
import requests
import traceback
import cv2
import ghostscript
import magic
from PIL import Image, ImageDraw, ExifTags
from wand.image import Image as wi
import numpy as np
from decouple import config
from django.utils.datastructures import MultiValueDictKeyError
from fuzzywuzzy import fuzz
from google.cloud import storage
from numpy import ndarray
from shapely.geometry.polygon import Polygon
from werkzeug.exceptions import BadRequest
import json
import sqlalchemy
from django.conf import settings
from django.http import HttpResponse
import ast
#from kyc_docs.kyc.documents.india.aadhar import AADHAAR
#from .documents.india.aadhar import AADHAAR

valid_uri = re.compile(
    r"^(?:http|ftp)s?://"  # http:// or https://
    # domain...
    r"(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+(?:[A-Z]{2,6}\.?|[A-Z0-9-]{2,}\.?)|"
    r"localhost|"  # localhost...
    r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})"  # ...or ip
    r"(?::\d+)?"  # optional port
    r"(?:/?|[/?]\S+)$",
    re.IGNORECASE,
)


def document_selection(domain_url):
    selection_query = sqlalchemy.text(
        "select B.critical_fields from adminapp_client as A inner join document_app_clientdocuments as B on A.id=B.client_id_id where A.domain_url='{}'".format(
            domain_url))
    try:
        with DB.connect() as conn:
            selected_data = conn.execute(selection_query)
            for row_data in selected_data:
                data = row_data['critical_fields']

    except Exception as err:
        return None
    return data


def run_query(query):
    selection_query = sqlalchemy.text(query)
    try:
        with DB.connect() as conn:
            selected_data = conn.execute(selection_query)
    except Exception as err:
        return HttpResponse(json.dumps({'system_status_code': 400, 'Error': str(err)}), content_type='application/json')
    return selected_data


def mask_aadhaar_no(aadhar_no):
    """masks aadhar number

    param aadhar_no:  aadhar number to be masked
    return: masked aadhar number

    """
    check_digits_regex = r'\d{12}|[X]{8}\d{4}'
    try:
        if len(aadhar_no) != 12:
            aadhar_no = ""
        elif not re.match(check_digits_regex, aadhar_no):
            aadhar_no = ""
        else:
            mask_digits = "XXXX-XXXX-"
            last_four_no = aadhar_no[-4:]
            aadhar_no = mask_digits + last_four_no
    except:
        aadhar_no = ""
    return aadhar_no


# Check if a string is valid uri
def is_uri(data):
    return re.match(valid_uri, data) if isinstance(data, str) else False


# Downloads the image, buffer it and returns a handle
def download_image(uri):
    buffer = io.BytesIO()
    response = requests.get(uri, stream=True)
    if response.status_code == 200:
        for chunk in response.iter_content():
            buffer.write(chunk)
        buffer.seek(0)
        response.close()
        return buffer
    raise BadRequest()


class FuzzyComparator(object):
    def __init__(self, value, threshold=1):
        self.value = value
        self.threshold = threshold

    def __str__(self):
        return str(self.value)

    def __contains__(self, query):
        # Handle warnings thrown by FuzzyWuzzy for symbols
        if len(query) == 1 and not str(query).isalpha():
            return False
        for item in self.value:
            if (
                    fuzz.token_set_ratio(query, item)
                    >= min(
                config("FUZZ_THRESHOLD", default=85, cast=int),
                ((len(query) - self.threshold) / len(query)) * 100,
            )
                    and abs(len(query) - len(item)) <= self.threshold
            ):
                return True
        return False

    def __eq__(self, other):
        for index, item in enumerate(other):
            # Handle warnings thrown by FuzzyWuzzy for symbols
            if len(item) == 1 and not str(item).isalpha():
                return False
            if (
                    fuzz.partial_ratio(item, self.value[index])
                    >= min(
                config("FUZZ_THRESHOLD", default=85, cast=int),
                ((len(item) - 1) / len(item)) * 100,
            )
                    and not abs(len(item) - len(self.value[index])) <= 1
            ):
                return False
            elif not fuzz.partial_ratio(item, self.value[index]) >= min(
                    config("FUZZ_THRESHOLD", default=85, cast=int), ((
                                                                             len(item) - 1) / len(item)) * 100
            ):
                return False
        return True

    def __len__(self):
        return len(self.value)

    def __getitem__(self, index):
        return self.value[index]


def get_passport_signature_coordinates(photo_coordinates):
    height_of_photo = photo_coordinates["bottom"] - photo_coordinates["top"]
    width_of_photo = photo_coordinates["right"] - photo_coordinates["left"]
    top = photo_coordinates["bottom"] + (height_of_photo * 0.25)
    bottom = photo_coordinates["bottom"] + (height_of_photo * 0.85)
    left = photo_coordinates["left"] - (width_of_photo * 0.3)
    right = photo_coordinates["right"] + (width_of_photo * 0.25)

    return {"top": int(top), "right": int(right), "bottom": int(bottom), "left": int(left)}


def get_pan_signature_coordinates(response, signature_location):
    if not signature_location:
        return None

    length_of_signature = signature_location[0].vertices[
                              1].x - signature_location[0].vertices[0].x
    pan_old_keywords = ["NAME", "FATHERS", "FATHER", "DATE", "BIRTH", "CARD"]
    detected_words = FuzzyComparator(
        detect_words(response.full_text_annotation.text))
    if any_of_these_exists_in(pan_old_keywords, detected_words):
        if any_of_these_exists_in(["DEPARTMENT", "GOVT", "INDIA"], detected_words):
            left_x_cord = signature_location[0].vertices[
                              0].x - (length_of_signature * 1.4)
            right_x_cord = signature_location[0].vertices[
                               1].x + (length_of_signature * 0.4)
            top_y_cord = signature_location[0].vertices[
                             0].y - (length_of_signature * 0.9)
            bottom_y_cord = signature_location[0].vertices[0].y

        else:
            left_x_cord = signature_location[
                              0].vertices[0].x - length_of_signature
            right_x_cord = signature_location[0].vertices[1].x
            top_y_cord = signature_location[0].vertices[3].y
            bottom_y_cord = signature_location[0].vertices[
                                3].y + (length_of_signature * 0.8)

    else:
        left_x_cord = signature_location[0].vertices[0].x
        right_x_cord = signature_location[0].vertices[
                           1].x + (length_of_signature * 1.6)
        top_y_cord = signature_location[0].vertices[
                         3].y - (length_of_signature * 1.2)
        bottom_y_cord = signature_location[0].vertices[0].y

    return {
        "top": int(top_y_cord),
        "right": int(right_x_cord),
        "bottom": int(bottom_y_cord),
        "left": int(left_x_cord)
    }


def crop_image(image, coordinates):
    if not coordinates:
        return None
    # cropped_image = Image.open(image)
    cropped_image = image[coordinates["top"]:coordinates[
        "bottom"], coordinates["left"]:coordinates["right"]]
    # cropped_image = cropped_image.crop(
    #     (coordinates["left"],
    #      coordinates["top"],
    #      coordinates["right"],
    #      coordinates["bottom"]))
    return cropped_image


def detect_words(detected_text):
    try:
        if detected_text.ListFields()[-1][1].code == 3:
            return False
    except:
        pass
    separators = [",", "'", "/", ":", "!"]  # "-",
    for separator in separators:
        detected_text = " ".join(detected_text.split(separator))
    return detected_text.split()


def generate_random_file_name(length=20, extension=None):
    """
    Random string Generator.
    :param length: Length of the Output string
    :param extension: File extension
    :return: random file name when extension passed otherwise a random string
    """
    timestamp = datetime.datetime.utcnow().strftime("%Y-%m-%d-%H%M%S")
    salt = "".join(
        random.choice(string.ascii_uppercase +
                      string.digits + string.ascii_lowercase)
        for _ in range(length - len(timestamp))
    )
    if extension:
        return "{}{}.{}".format(timestamp, salt, extension).replace("..", ".")

    return "{}{}".format(timestamp, salt)


def upload_file_to_google_cloud(filename, format, content_type='image/jpeg'):
    if format.lower() not in ALLOWED_EXTENSIONS:
        raise BadRequest

    storage_client = storage.Client()

    bucket = storage_client.get_bucket(LABS_CLOUD_STORAGE_BUCKET if DEBUG else PRODUCTION_CLOUD_STORAGE_BUCKET)

    blob = bucket.blob(filename)
    blob.upload_from_filename(filename, content_type=content_type)
    blob.make_public()

    return blob.public_url


def store_image_on_google_drive(cropped_image, format='JPEG'):
    if not isinstance(cropped_image, ndarray):
        return None
    name = "/tmp/" + generate_random_file_name(extension=format.lower())
    cv2.imwrite(name, cropped_image)
    cropped_image_google_bucket_url = upload_file_to_google_cloud(filename=name, format=format)

    if not config("DEBUG", default=False, cast=bool):
        os.remove(name)
    return cropped_image_google_bucket_url


def any_of_these_exists_in(queries, detected_words):
    assert type(detected_words) == FuzzyComparator

    for query in queries:
        if query in detected_words:
            return True
    return False


# Returns the bounding_boxes of words fulfilling fuzzy comparison
def get_similar_words_vertices(texts, options, bounded=True):
    # Ensure options is of type FuzzyComparator
    assert type(options) == FuzzyComparator

    vertices = []
    for text in texts:
        # Prevent capturing of extra data
        if len(vertices) == len(options) and bounded:
            break
        if text.description in options:
            # print(text, options)
            vertices.append(text.bounding_poly)
    return vertices


def fix_image_orientation_using_exif(image, filename):
    if isinstance(image, io.BytesIO):
        image.seek(0)
    pil_image = Image.open(image)
    try:
        exif = dict((ExifTags.TAGS[k], v) for k, v in pil_image._getexif().items() if k in ExifTags.TAGS)
        if exif['Orientation'] == 3:
            pil_image = pil_image.rotate(180, expand=True)
        elif exif['Orientation'] == 8:
            pil_image = pil_image.rotate(90, expand=True)
        elif exif['Orientation'] == 6:
            pil_image = pil_image.rotate(270, expand=True)
        else:
            raise AttributeError
        pil_image.save(filename[1], format='JPEG', optimize=True, quality=95)
    except:
        with os.fdopen(filename[0], 'wb') as f:
            image.seek(0) if isinstance(image, io.BytesIO) else image.open()
            f.write(image.read())


def process_input_from_request(request):
    file_name = mime_type = ""
    try:
        image = request.FILES['image']
        print("IMAGE ",image)
        # import pdb;pdb.set_trace()

    except MultiValueDictKeyError:
        try:
            file_url = request.POST['image']
            if not file_url:
                raise MultiValueDictKeyError

            file_name = file_url.split('/')[-1]
            image = download_image(file_url) if is_uri(file_url) else False
            if not image:
                raise Exception

        except MultiValueDictKeyError:
            return HttpResponse({
                       'status':'FAIL',
                        'status_code':204,
                        'message':"No Content",
                        'file_name':None,
                        'file_type':None,
                        'url':str(request.path)
                    }) , [], "", "", ""

        except Exception as err:
            return HttpResponse({
                        'status':'FAIL',
                        'status_code':204,
                        'message':"Invalid URI",
                        'file_name':None,
                        'file_type':None,
                        'url':str(request.path)
                    }), [], "", "", ""

    file_name = image.name if not file_name else file_name
    mime_type = magic.from_buffer(image.read(1024), mime=True)
    file_size = image.getbuffer().nbytes if isinstance(image, io.BytesIO) else image.size

    try:
        assert hasattr(image, "read")
    except AssertionError:
        return HttpResponse({
                   'status':'FAIL',
                    'status_code':204,
                    'message':"File Not Readable",
                    'file_name':file_name,
                    'file_type':mime_type.split('/')[-1],
                    'url':str(request.path)
                }), [], "", "", ""

    if mime_type in ['application/pdf']:
        image.seek(0) if isinstance(image, io.BytesIO) else image.open()
        pages = []
        max_confidence_index = ''
        int_, temp_local_filename = tempfile.mkstemp()

        f = os.fdopen(int_, 'wb')
        f.write(image.read())  # write the tmp file
        f.close()

        temp_local_dir = tempfile.mkdtemp()
        gs_args = ["pdf2png",
                   "-dSAFER -dBATCH -dNOPAUSE",
                   "-r300",
                   "-sDEVICE=pnggray",
                   "-dTextAlphaBits=4 -sPAPERSIZE=a4",
                   "-o", temp_local_dir + "page-%02d.png",
                   temp_local_filename,
                   ]

        encoding = locale.getpreferredencoding()
        gs_args = [gs_arg.encode(encoding) for gs_arg in gs_args]
        with ghostscript.Ghostscript(*gs_args) as g:
            ghostscript.cleanup()
        files = sorted(glob.glob(temp_local_dir + "*.png"))
        print(files)
        return None, files, file_name, mime_type, file_size

    elif mime_type == 'image/tiff':
        file = []
        _, temp_local_filename = tempfile.mkstemp()
        image.seek(0) if isinstance(image, io.BytesIO) else image.open()
        pdf_tiff = wi(file=image, resolution=180)

        for i, page in enumerate(pdf_tiff.sequence):
            with wi(page) as page_image:
                page_image.alpha_channel = False
                img_buffer = np.asarray(
                    bytearray(page_image.make_blob(format='jpeg')), dtype='uint8')
                bytesio = io.BytesIO(img_buffer)

                image = Image.open(bytesio)
                image.save(temp_local_filename+'_' + str(i)+'.jpeg')
                file.append(temp_local_filename+'_' + str(i)+'.jpeg')

        # import pdb;pdb.set_trace()
        return None, file, file_name, mime_type, file_size
        # return None, [temp_local_filename], file_name, mime_type, file_size

    elif mime_type == 'text/html':
        return HttpResponse(status='FAIL', status_code=204, message="Not a file object", file_name=None, file_type=None,
                               url=str(request.path)).to_dict(), [], file_name, mime_type, file_size

    else:
        temp_local_filename = tempfile.mkstemp()
        fix_image_orientation_using_exif(image, temp_local_filename)
        return None, [temp_local_filename[1]], file_name, mime_type, file_size


def list_duplicates(seq):
    tally = defaultdict(list)
    for i, item in enumerate(seq):
        tally[item].append(i)
    return ((key, locs) for key, locs in tally.items()
            if len(locs) > 1)



def is_english(text):
    if all(ord(i) < 128 for i in text):
        return True
    elif all(ord(i) >= 128 for i in text):
        return False
    else:
        return ''.join([x for x in text if ord(x) < 128])


def check_non_horizontal(word):
    try:
        y_vertices = [x1['y'] for x1 in word['bounding_poly']['vertices']]
        x_vertices = [x1['x'] for x1 in word['bounding_poly']['vertices']]
    except KeyError:
        return True
    angle_to_rot_img = np.arctan2(
        y_vertices[1] - y_vertices[0], x_vertices[1] - x_vertices[0]) * 57.2958
    if abs(angle_to_rot_img) > 45:
        return True
    return False


def get_non_english_low_confidence_words(response, non_english_flag, low_confidence_flag, confidence_threshold):
    c_thres = 1
    if low_confidence_flag:
        c_thres = confidence_threshold
    non_english_or_low_confidence_words = []
    for page in response['full_text_annotation']['pages']:
        for block in page['blocks']:
            for para in block['paragraphs']:
                for word in para['words']:
                    c = word['confidence']
                    word_text = ''.join([symbol['text']
                                         for symbol in word['symbols']])
                    if (low_confidence_flag and c <= c_thres) or not (
                            is_english(word_text) if non_english_flag else True):
                        non_english_or_low_confidence_words.append(word_text)
    return non_english_or_low_confidence_words


def check_box_height(word, flag, height_threshold):
    if not flag:
        return False
    y_vertices = [x1['y'] for x1 in word['bounding_poly']['vertices']]
    if (((y_vertices[3] - y_vertices[0]) + (y_vertices[2] - y_vertices[1])) / 2) < height_threshold:
        return True
    return False


def find_overlapping_words(document):
    words = [word['description'] for word in document]
    indexes_to_drop = []
    for dup in sorted(list_duplicates(words)):
        if dup[0].isalpha():
            polygon = Polygon(
                [(vertex['x'], vertex['y']) for vertex in document[dup[1][0]]['bounding_poly']['vertices']])
            second_polygon = Polygon(
                [(vertex['x'], vertex['y']) for vertex in document[dup[1][1]]['bounding_poly']['vertices']])
            if polygon.area > second_polygon.area:
                i = 0
                bigger_polygon = polygon
                smaller_polygon = second_polygon
            else:
                i = 1
                bigger_polygon = second_polygon
                smaller_polygon = polygon

            if bigger_polygon.intersection(smaller_polygon).area / smaller_polygon.area > 0.65:
                indexes_to_drop.append(dup[1][i])

    for ind in sorted(indexes_to_drop, reverse=True):
        del document[ind]
    return document


def any_of_these_exists_in(queries, detected_words):
    assert type(detected_words) == FuzzyComparator

    for query in queries:
        if query in detected_words:
            return True
    return False


def get_straight_line_coordinates(response, image_shape, reduction_ratio=0.3, remove_non_english_words=False,
                                  remove_low_confidence_words=False, check_box_height_flag=False, face=None,
                                  confidence_threshold=1,
                                  height_threshold=40, delete_overlapping_words=False):
    document = response['text_annotations'][1:]

    non_english_or_low_confidence_words = []
    if remove_non_english_words or remove_low_confidence_words:
        non_english_or_low_confidence_words = get_non_english_low_confidence_words(
            response, remove_non_english_words, remove_low_confidence_words, confidence_threshold)

    if delete_overlapping_words:
        document = find_overlapping_words(document)

    updated_document = []
    new_image = Image.new('1', (int(image_shape[1]), int(image_shape[0])))
    draw = ImageDraw.Draw(new_image)

    for i, doc in enumerate(document):
        if check_non_horizontal(doc) or check_box_height(doc, check_box_height_flag, height_threshold) or doc[
            'description'] in non_english_or_low_confidence_words:
            continue

        updated_document.append(doc)
        pil_input = []
        xs = [x1['x'] for x1 in doc['bounding_poly']['vertices']]
        ys = [x1['y'] for x1 in doc['bounding_poly']['vertices']]
        ht = ((ys[3] - ys[0]) + (ys[2] - ys[1])) / 2

        ys[0] = ys[0] + ht * reduction_ratio
        ys[1] = ys[1] + ht * reduction_ratio
        ys[2] = ys[2] - ht * reduction_ratio
        ys[3] = ys[3] - ht * reduction_ratio

        # if not (xs[1] > xs[0] and ys[2] > ys[1]):
        #     continue
        for i_ in range(4):
            pil_input.append(xs[i_])
            pil_input.append(ys[i_])
        draw.polygon(pil_input, fill=1, outline=1)

    pil_image = new_image.convert('RGB')

    img = np.array(pil_image)
    if face:
        width_of_photo = face['right'] - face['left']
        right = face['right'] + (width_of_photo * 0.5)
        img[face['top']:, 0:int(right)] = np.zeros(
            img[face['top']:, 0:int(right)].shape)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hist = cv2.reduce(gray, 1, cv2.REDUCE_AVG).reshape(-1)
    th = 2
    H, W = img.shape[:2]
    lowers = [y for y in range(H - 1) if hist[y] > th >= hist[y + 1]]

    # for y in lowers:
    #    img=cv2.line(img, (0,y), (W, y), (0,255,0), 1)
    # cv2.imwrite("/Users/monarkunadkat/Desktop/shubham_shubham_result.png", img)

    return lowers, updated_document


def generate_sentences(document, lowers, shape, min_x=0, min_y=0, max_x=9999, max_y=9999, ignore_words=[]):
    words = []
    centers = []
    for doc in document:
        if doc['description'] in FuzzyComparator(ignore_words):
            continue
        xc = sum([x1['x'] for x1 in doc['bounding_poly']['vertices']]) / 4
        yc = sum([x1['y'] for x1 in doc['bounding_poly']['vertices']]) / 4
        words.append(doc['description'])
        centers.append([xc, yc])

    sentences = {}
    i = 0
    lowers.append(shape)
    lowers.sort

    for i, y in enumerate(lowers):
        s = []
        center_x = []
        for key, value in zip(words, centers):
            if i == 0:
                if value[1] <= y and min_x < value[0] < max_x and max_y > value[1] > min_y:
                    s.append(key)
                    center_x.append(value[0])
            else:
                if lowers[i - 1] <= value[1] <= y and max_x > value[0] > min_x and max_y > value[1] > min_y:
                    s.append(key)
                    center_x.append(value[0])
        z = [x for _, x in sorted(zip(center_x, s))]
        sentences[str(y)] = z

    final_sentences = []
    # for value in sentences.values():
    #     sent = ''
    #     for i, v in enumerate(value):
    #         try:
    #             if v.isalnum() and value[i + 1].isalnum():
    #                 sent += v + ' '
    #             elif v.isalnum() and not value[i + 1].isalnum():
    #                 sent += v
    #             elif not v.isalnum() and value[i + 1].isalpha():
    #                 sent += v
    #             elif not v.isalnum() and value[i + 1].isdigit():
    #                 if v == '.' or v == '-':
    #                     sent += v
    #                 else:
    #                     sent += v + ' '
    #             elif not v.isalnum() and not value[i + 1].isalnum():
    #                 sent += v + ' '
    #         except IndexError:
    #             sent += v
    #     sent = sent.replace('\n', ' ')
    #     if len(sent) < 4:
    #         continue
    #     final_sentences.append(sent)

    for value in sentences.values():
        sent = ''
        for i, v in enumerate(value):
            try:
                if re.sub('[^0-9a-zA-Z]+', '', v).isalnum() and re.sub('[^0-9a-zA-Z]+', '', value[i + 1]).isalnum():
                    sent += v + ' '
                elif re.sub('[^0-9a-zA-Z]+', '', v).isalnum() and not value[i + 1].isalnum():
                    sent += v
                elif not v.isalnum() and value[i + 1].isalpha():
                    sent += v
                elif not v.isalnum() and value[i + 1].isdigit():
                    if v == '.' or v == '-':
                        sent += v
                    else:
                        sent += v + ' '
                elif not v.isalnum() and not value[i + 1].isalnum():
                    sent += v + ' '
            except IndexError:
                sent += v
        sent = sent.replace('\n', ' ')
        if len(sent) < 4:
            continue
        final_sentences.append(sent)

    return final_sentences


def drivers_lines(document, lowers, shape):
    words = []
    centers = []
    for doc in document:
        xc = sum([x1['x'] for x1 in doc['bounding_poly']['vertices']]) / 4
        yc = sum([x1['y'] for x1 in doc['bounding_poly']['vertices']]) / 4
        words.append(doc['description'])
        centers.append([xc, yc])

    sentences = {}
    i = 0
    lowers.append(shape)
    lowers.sort

    for i, y in enumerate(lowers):
        s = []
        center_x = []
        for key, value in zip(words, centers):
            if i == 0:
                if value[1] < y:
                    s.append(key)
                    center_x.append(value[0])
            else:
                if value[1] > lowers[i - 1] and value[1] < y:
                    s.append(key)
                    center_x.append(value[0])

        z = [x for _, x in sorted(zip(center_x, s))]
        sentences[str(y)] = z

    final_sentences = []
    for value in sentences.values():
        sent = ''
        for i, v in enumerate(value):
            try:
                if v.isalnum() and value[i + 1].isalnum():
                    sent += v + ' '
                elif v.isalnum() and not value[i + 1].isalnum():
                    sent += v
                elif not v.isalnum() and value[i + 1].isalnum():
                    sent += v
            except IndexError:
                sent += v
        final_sentences.append(sent)
    return final_sentences


def clean_sentences(block_lines):
    final_sentences = []
    for value in block_lines:
        value = value.split()
        sent = ''
        for i, v in enumerate(value):
            try:
                if v.isalnum() and value[i + 1].isalnum():
                    sent += v + ' '
                elif v.isalnum() and not value[i + 1].isalnum():
                    sent += v
                elif not v.isalnum() and value[i + 1].isalpha():
                    sent += v
                elif not v.isalnum() and value[i + 1].isdigit():
                    if v == '.' or v == '-':
                        sent += v
                    else:
                        sent += v + ' '
            except IndexError:
                if value[i - 1].isalnum():
                    sent += v
                else:
                    sent += ' ' + v
        sent = sent.replace('\n', ' ')
        if len(sent) < 4:
            continue
        final_sentences.append(sent)
    return final_sentences


def get_complete_data_by_lines(words):
    y_dict = {}

    for word in words:
        if word == words[0] or check_non_horizontal(word):
            continue
        location = word['bounding_poly']
        top_left_coordinate = location['vertices'][0]
        bottom_right_coordinate = location['vertices'][2]
        mid = int((bottom_right_coordinate[
                       'y'] + top_left_coordinate['y']) // 2)
        size = int(bottom_right_coordinate['y'] - top_left_coordinate['y'])

        flag = False
        for key in y_dict.keys():
            y = int(key)
            if abs(y - mid) <= size // 2 and is_english(word['description']):
                y_dict[key].append(word)
                flag = True
                break
        if not flag and is_english(word['description']):
            temp = []
            temp.append(word)
            y_dict[str(mid)] = temp
    return y_dict


def get_sorted_lines_with_x_axis(to_sort):
    for i in to_sort.keys():
        a = sorted(to_sort[str(i)], key=lambda item: item[
            'bounding_poly']['vertices'][0]['x'])
        to_sort[str(i)] = a
    return to_sort


def get_lines_within_bounds(response, bounds):
    text = []
    for word in response['text_annotations']:
        if check_non_horizontal(word):
            continue
        if (word['bounding_poly']['vertices'][0]['y'] > bounds["top_y"]
                and word['bounding_poly']['vertices'][0]['x'] > bounds["left_x"]
                and word['bounding_poly']['vertices'][0]['x'] < bounds["right_x"]
                and word['bounding_poly']['vertices'][0]['y'] < bounds["bottom_y"]):
            text.append(word)

    lines = get_complete_data_by_lines(text)
    lines = get_sorted_lines_with_x_axis(lines)

    return lines


def sort_lines(lines):
    keys = sorted([int(key) for key in lines.keys()])
    sorted_lines = OrderedDict()

    for key in keys:
        sorted_lines[str(key)] = lines[str(key)]
    return sorted_lines


def construct_sentences(lines):
    for key in lines:
        temp = lines[key]
        sentence = ' '.join([word['description'] for word in lines[key]])
        lines[key] = {
            'sentence': sentence,
            'annotation': temp
        }
    return lines


def get_words_with_distance_below(response, top_boundary, right_boundary=999999, left_boundary=0):
    result = []
    for index, keyword in enumerate(detect_words(str(response['full_text_annotation']['text']))):
        old_location = None
        for word in result:
            if word[0] == keyword:
                old_location = word[2]
        location = find_word_location(
            response['full_text_annotation'], keyword, old_location, below=top_boundary)

        if location:
            if (
                    location['vertices'][0]['y'] >= top_boundary
                    and location['vertices'][1]['y'] >= top_boundary
                    and location['vertices'][1]['x'] <= right_boundary
                    and location['vertices'][0]['x'] >= left_boundary
            ):
                result.append(
                    [
                        keyword,
                        (
                                abs(location['vertices'][0]['y'] - top_boundary)
                                + abs(location['vertices'][1]['y'] - top_boundary)
                        )
                        // 2,
                        location,
                    ]
                )
    return result


def get_words_with_distance_above(
        response, bottom_boundary, right_boundary=999999, left_boundary=0
):
    result = []
    for index, keyword in enumerate(
            detect_words(str(response['full_text_annotation']['text']))
    ):
        old_location = None
        for word in result:
            if word[0] == keyword:
                old_location = word[2]
        location = find_word_location(
            response[
                'full_text_annotation'], keyword, old_location, above=bottom_boundary
        )

        if location:
            if (
                    location['vertices'][2].y <= bottom_boundary
                    and location['vertices'][3].y <= bottom_boundary
                    and location['vertices'][1].x <= right_boundary
                    and location['vertices'][0].x >= left_boundary
            ):
                result.append(
                    [
                        keyword,
                        (
                                abs(location['vertices'][2]['y'] - bottom_boundary)
                                + abs(location['vertices'][3]
                                      ['y'] - bottom_boundary)
                        )
                        // 2,
                        location,
                    ]
                )
    return result


def assemble_word(word):
    assembled_word = ""
    for symbol in word['symbols']:
        assembled_word += symbol['text']
    return assembled_word


def find_word_location(
        document, word_to_find, old_position=None, below=0, above=999999
):
    word_to_find = word_to_find.replace(".", "")
    word_to_find = word_to_find.replace(",", "")
    word_to_find = word_to_find.replace("!", "")
    for page in document['pages']:
        for block in page['blocks']:
            for paragraph in block['paragraphs']:
                for word in paragraph['words']:
                    assembled_word = assemble_word(word)
                    if (
                            assembled_word == word_to_find
                            and old_position != word['bounding_box']
                            and word['bounding_box']['vertices'][3]['y'] >= below
                            and word['bounding_box']['vertices'][0]['y'] <= above
                    ):
                        return word['bounding_box']


def assemble_word(word):
    assembled_word = ""
    for symbol in word['symbols']:
        assembled_word += symbol['text']
    return assembled_word


def get_mean_std(document, keywords_list=[]):
    heights = []
    for doc in document:
        if doc['description'].lower() in keywords_list or len(keywords_list) == 0:
            heights.append(abs(doc['bounding_poly']['vertices'][
                                   2]['y'] - doc['bounding_poly']['vertices'][0]['y']))
    mean = np.mean(heights)
    std = np.std(heights)
    return std, mean


def text_within(document, x1, y1, x2, y2):
    text = ""
    for page in document['pages']:
        for block in page['blocks']:
            for paragraph in block['paragraphs']:
                for word in paragraph['words']:
                    for symbol in word['symbols']:
                        min_x = min(symbol['bounding_box']['vertices'][0]['x'],
                                    symbol['bounding_box']['vertices'][1]['x'],
                                    symbol['bounding_box']['vertices'][2]['x'],
                                    symbol['bounding_box']['vertices'][3]['x'])
                        max_x = max(symbol['bounding_box']['vertices'][0]['x'],
                                    symbol['bounding_box']['vertices'][1]['x'],
                                    symbol['bounding_box']['vertices'][2]['x'],
                                    symbol['bounding_box']['vertices'][3]['x'])
                        min_y = min(symbol['bounding_box']['vertices'][0]['y'],
                                    symbol['bounding_box']['vertices'][1]['y'],
                                    symbol['bounding_box']['vertices'][2]['y'],
                                    symbol['bounding_box']['vertices'][3]['y'])
                        max_y = max(symbol['bounding_box']['vertices'][0]['y'],
                                    symbol['bounding_box']['vertices'][1]['y'],
                                    symbol['bounding_box']['vertices'][2]['y'],
                                    symbol['bounding_box']['vertices'][3]['y'])

                        if (min_x >= x1 and max_x <= x2 and min_y >= y1 and max_y <= y2):
                            text += symbol['text']
                            try:
                                if (symbol['property']['detected_break']['type'] == 1 or
                                        symbol['property']['detected_break']['type'] == 3):
                                    text += ' '
                            except:
                                continue

    return text


def check_pan_regex(input_text):
    if not input_text:
        return ""
    input_text = re.search(r'[A-Z]{5}\d{4}[A-Z]', input_text.upper())
    if input_text:
        input_text = input_text.group()
    else:
        input_text = ''

    return input_text


def check_tan_regex(input_text):
    input_text = re.search(r'[A-Z]{4}\d{5}[A-Z]', input_text.upper())

    if input_text:
        input_text = input_text.group()
    else:
        input_text = ''

    return input_text


def generate_sentences_for_med_reports(document, lowers, shape):
    words = []
    centers = []
    for doc in document:
        xc = sum([x1['x'] for x1 in doc['bounding_poly']['vertices']]) / 4
        yc = sum([x1['y'] for x1 in doc['bounding_poly']['vertices']]) / 4
        words.append(doc['description'])
        centers.append([xc, yc])

    sentences = {}
    i = 0
    lowers.append(shape)
    lowers.sort

    for i, y in enumerate(lowers):
        s = []
        center_x = []
        for key, value in zip(words, centers):
            if i == 0:
                if value[1] <= y:
                    s.append(key)
                    center_x.append(value[0])
            else:
                if lowers[i - 1] <= value[1] <= y:
                    s.append(key)
                    center_x.append(value[0])
        z = [x for _, x in sorted(zip(center_x, s))]
        sentences[str(y)] = z

    final_sentences = []
    for value in sentences.values():
        sent = ''
        for i, v in enumerate(value):
            try:
                if v.isalnum() and value[i + 1].isalnum():
                    sent += v + ' '
                elif v.isalnum() and not value[i + 1].isalnum():
                    sent += v
                elif not v.isalnum() and value[i + 1].isalpha():
                    sent += v
                elif not v.isalnum() and value[i + 1].isdigit():
                    if v == '.' or v == '-':
                        sent += v
                    else:
                        sent += v + ' '
                elif not v.isalnum() and not value[i + 1].isalnum():
                    sent += v + ' '
            except IndexError:
                sent += v
        sent = sent.replace('\n', ' ')
        if sent != '':
            final_sentences.append(sent)
    return final_sentences


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        try:
            float(s[:-1])
            return True
        except ValueError:
            pass
        pass
    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass
    return False


def get_assembled_word(word):
    assembled_word = ""
    for symbol in word.symbols:
        assembled_word += symbol.text
    return assembled_word


def get_words_with_confidence(response):
    words = {}
    for page in response.full_text_annotation.pages:
        for block in page.blocks:
            for para in block.paragraphs:
                for word in para.words:
                    words[get_assembled_word(word)] = word.confidence
    return words


def add_confidence(response, result, qr=False):
    if not response:
        for field in result:
            if field not in ['num_expected_fields', 'num_extracted_fields', 'fields_not_detected']:
                value = result[field]
                result[field] = {"value": value, "confidence": "{0:.5f}".format(100) if value else ""}
        return result

    confidence_result = dict()
    words = get_words_with_confidence(response)
    for field in result:
        if field in ['quarter', 'bank_stat_salary', 'photo', 'signature']:
            value = result[field]
            confidence_result[field] = {'value': value, 'confidence': "{0:.5f}".format(95) if value else ""}

        confidence = []
        for i, field_word in enumerate(str(result[field]).split()):
            if field in ['micr_code', 'bank_name', 'branch_name', 'branch_code']:
                confidence.append(100)
                continue
            try:
                confidence.append(words[field_word] * 100)
            except KeyError:
                confidence.append(90)

        cfd = np.average(confidence) if confidence else ""
        confidence_result[field] = {'value': result[field], 'confidence': "{0:.5f}".format(cfd) if cfd else cfd}

    return confidence_result


def merge_result(api_response, doc_class, domain_url,doc_class_list):
    fields = {
        'FORM_16': ['certificate_no', 'last_updated', 'pan_number', 'deducter_pan_number', 'deducter_tan_number',
                    'assessment_year', 'period_w_employer', 'name_address_employer', 'name_address_employee', 'quarter',
                    'aggregate', 'net_salary', 'gross_salary', 'total_tax_payable'],
        'ITR_SAHAJ': ['itr_type', 'name_of_customer', 'pan_number', 'town_name', 'state_name', 'pincode', 'filing_type',
                      'date_of_birth','mobile_no', 'country', 'verification_date', 'verification_place', 'gross_salary',
                      'deduction','net_salary','assessment_year', 'ifsc_code', 'bank_name', 'account_number',
                      'tax_deduct_acc_no','name_address_employer','income_chargeble', 'exempt_income1', 'exempt_income2',
                      'exempt_income3', 'acknowledgement_no']
    }
    doc_id = {'FORM_16': '7','ITR_SAHAJ':'4' }
    ds = [json_['result'] for json_ in api_response['response'] if json_['result']]
    ds = ds if ds else [{}]
    tagged_image_urls = [json_.pop('tagged_image_url') for json_ in api_response['response']]
    raw_ocr_response_urls = [json_.pop('raw_ocr_response') for json_ in api_response['response']]
    total_response_time = str(sum([float(json_.pop('response_time')) for json_ in api_response['response']]))

    merged_result = {}
    for k in ds[0].keys():
        value = [str(d[k]['value']) for d in ds]
        confidence = [d[k]['confidence'] for d in ds]
        index = np.argmax(confidence)

        merged_result[k] = {"value": value[index] if k != 'quarter' else ast.literal_eval(value[index]) if value else [],
                            "confidence": str(confidence[index])}

    not_detected_fields = []
    if merged_result:
        not_detected_count = 0
        for field in fields[doc_class]:
            if len(merged_result[field]['value']) == 0 and len(merged_result[field]['confidence']) == 0:
                not_detected_count += 1
                not_detected_fields.append(field)
    else:
        not_detected_count = len(fields[doc_class])

    api_response['response'] = api_response['response'][:1]
    if merge_result:
        api_response['response'][0]['status'] = "SUCCESS"
        api_response['response'][0]['system_status_code'] = "200"
        api_response['response'][0]['message'] = ""
        api_response['response'][0]['document_type'] = doc_id[doc_class]

    api_response['response'][0]['result'] = merged_result
    api_response['response'][0]['tagged_image_url'] = tagged_image_urls
    api_response['response'][0]['raw_ocr_response'] = raw_ocr_response_urls
    api_response['response'][0]['response_time'] = total_response_time
    api_response['response'][0]['num_expected_fields'] = str(len(fields[doc_class]))
    api_response['response'][0]['num_extracted_fields'] = str(len(fields[doc_class]) - not_detected_count)
    api_response['response'][0]['fields_not_detected'] = str(not_detected_fields)

    if merged_result:
        selected_data = document_selection(domain_url)
        if selected_data:
            selected_data = ast.literal_eval(selected_data)
            category_name_list = []
            for i in range(0, len(selected_data['domain_data'][0]['category_data'])):
                for j in range(0, len(selected_data['domain_data'][0]['category_data'][i]['document_data'])):
                    for k in range(0, len(
                            selected_data['domain_data'][0]['category_data'][i]['document_data'][j]['fields'])):
                        category_name_list.append(
                            selected_data['domain_data'][0]['category_data'][i]['document_data'][j]['fields'][
                                k][
                                'label'])

            fields_not_detected_list = ast.literal_eval(api_response['response'][0]['fields_not_detected'])
            if any(x in fields_not_detected_list for x in category_name_list):
                api_response['response'][0]['status'] = 'PARTIAL SUCCESS'
                api_response['response'][0]['system_status_code'] = 206
                api_response['response'][0]['message'] = 'Critical Field(s) not detected'

    return api_response


def get_uk_passport_signature_coordinates(response, signature_location):
    """
            Summary line.

            get the signature coordinates for UK  Passport

            Parameters:
            arg1 (photo coordinates): photo coordinates

            Returns:
            dict : signature coordinates with respect to photo coordinates

            """
    if not signature_location:
        return None

    length_of_signature = signature_location[0].vertices[
                              1].x - signature_location[0].vertices[0].x

    left_x_cord = signature_location[0].vertices[0].x
    right_x_cord = signature_location[0].vertices[
                       1].x + (length_of_signature * 5)
    top_y_cord = signature_location[0].vertices[0].y + length_of_signature
    bottom_y_cord = signature_location[0].vertices[1].y + (length_of_signature * 2.5)

    return {
        "top": int(top_y_cord),
        "right": int(right_x_cord),
        "bottom": int(bottom_y_cord),
        "left": int(left_x_cord)
    }


def get_uk_dl_signature_coordinates(photo_coordinates):
    """
        Summary line.

        get the signature coordinates for UK  Driving License

        Parameters:
        arg1 (photo coordinates): photo coordinates

        Returns:
        dict : signature coordinates with respect to photo coordinates

        """

    height_of_photo = photo_coordinates["bottom"] - photo_coordinates["top"]
    width_of_photo = photo_coordinates["right"] - photo_coordinates["left"]
    top = photo_coordinates["right"] + height_of_photo*0.35
    bottom = photo_coordinates["bottom"] + height_of_photo*0.1
    left = photo_coordinates["right"] + (width_of_photo * 0.75)
    right = photo_coordinates["right"] + (width_of_photo * 3)

    return {"top": int(top), "right": int(right), "bottom": int(bottom), "left": int(left)}


