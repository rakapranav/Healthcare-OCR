import json
import re

# from recognic_utilities.core_utils.output_json import JSON_STANDARD
from .utils import (
    detect_words,
    get_straight_line_coordinates,
    generate_sentences,
)


def uniquify(string):
    output = []
    seen = set()
    string = string.replace('"','')

    for word in string.split():
        if word not in seen:
            output.append(word)
            seen.add(word)
    return ' '.join(output)


class REQ_FORM:
    def __init__(self, vision_response, new_shape):
        self.vision_response = vision_response
        self.new_shape = new_shape

    def splitString(self, str_):
        alpha = ""
        num = ""
        special = ""
        for i in range(len(str_)):
            if str_[i].isdigit():
                num = num + str_[i]
            elif (('A' <= str_[i] <= 'Z') or
                  ('a' <= str_[i] <= 'z')):
                alpha += str_[i]
            elif str_[i] == ' ' or str_[i] == '.':
                alpha += str_[i]
            else:
                special += str_[i]
        return alpha

    def remember(self, x):
        self.memory = x
        return True

    def perform_ocr(self):
        document = self.vision_response['text_annotations']
        d_text = document[0]['description']
        # print("D Text", d_text)

        lowers, updated_document = get_straight_line_coordinates(self.vision_response, self.new_shape)
        final_sentences = generate_sentences(updated_document, lowers, int(self.new_shape[0]))

        detected_words = detect_words(self.vision_response['full_text_annotation']['text'])
        # print('Detected ',detected_words)

        for i, sentence in enumerate(final_sentences):
            print(i, sentence)



        return None

    def as_dict(self):
        return self.perform_ocr()

    def as_json(self):
        pass

    def as_json_with_ocr(self):
        pass

    def as_unified_json(self):
        return json.dumps(self.as_dict())
