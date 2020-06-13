import json
import cv2
import numpy as np
from rest_framework.decorators import api_view
from django.views.decorators.cache import never_cache
from django.views.decorators.csrf import csrf_exempt
from django.http import HttpResponse
from rest_framework import status
from keras import backend as K
from keras.models import load_model
import tensorflow as tf

model_path = 'healthcare/models/requisition_form.hdf5'
model_kyc = load_model(model_path)
model_kyc._make_predict_function()
graph_kyc = tf.get_default_graph()


def kyc_classifier(image, model, graph):
    module_dict = {
        0: 'UnidentifiedId',
        1: 'Requisition_Form',
    }

    num_channel = 1
    if num_channel == 1:
        if K.common.image_dim_ordering() == 'th':
            image = np.expand_dims(image, axis=0)
            image = np.expand_dims(image, axis=0)
        else:
            image = np.expand_dims(image, axis=-1)
            image = np.expand_dims(image, axis=0)

    else:
        if K.common.image_dim_ordering() == 'th':
            image = np.rollaxis(image, 2, 0)
            image = np.expand_dims(image, axis=0)
        else:
            image = np.expand_dims(image, axis=0)

    with graph.as_default():
        value = model.predict_classes(image)
        prob_doc = model.predict_proba(image)

    if value[0] != 0 and np.amax(prob_doc) < 0.65:
        doc_id = 0
    elif value[0] == 0:
        doc_id = 0
    elif value[0] == 1:
        doc_id = 1

    return module_dict[doc_id], doc_id


# Create your views here.
@csrf_exempt
@never_cache
@api_view(['POST'])
def requisition_form(request):
    try:
        doc_class = ""
        image = cv2.imdecode(np.fromstring(request.FILES['image'].read(), np.uint8), cv2.IMREAD_UNCHANGED)
        test_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        test_image = cv2.resize(test_image, (128, 128))
        test_image = np.array(test_image)
        test_image = test_image.astype('float32')
        test_image /= 255

        if not doc_class:
            doc_class, max_confidence_index = kyc_classifier(test_image, model_kyc, graph_kyc)
        response = {
            "document": doc_class,
        }
        return HttpResponse(json.dumps(response), status=200)

    except Exception as e:
        print(e)
        return HttpResponse("final_json", status.HTTP_400_BAD_REQUEST)
