from google.cloud import vision


class OCRClient(object):
    def __init__(self, image, language=['en']):
        self.client = vision.ImageAnnotatorClient()
        # import pdb;pdb.set_trace()
        self.language = language
        self.image_context = None
        self._ocr_cache = None

        self.set_image(image)

        self.image_context = vision.types.ImageContext(language_hints=['en'])

    def set_image(self, image):
        image.seek(0)
        self.image = vision.types.Image(content=image.read())
        self.clear()

    def load(self):
        return self.client.text_detection(image=self.image, image_context=self.image_context)

    def clear(self):
        self._ocr_cache = None

    def _get_response(self):
        if self._ocr_cache:
            return self._ocr_cache

        # Lazy loading
        self._ocr_cache = self.load()
        return self._ocr_cache

    response = property(_get_response)
