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
from .image_utils import IUtils
from .utils import process_input_from_request
from google.protobuf.json_format import MessageToJson
from google.api_core.exceptions import InvalidArgument
import traceback

model_path = 'healthcare/models/requisition_form.hdf5'
model_kyc = load_model(model_path)
model_kyc._make_predict_function()
graph_kyc = tf.get_default_graph()


def doc_classifier(image, model, graph):
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

    np.set_printoptions(suppress=True)
    print("Values :", value, prob_doc)

    return module_dict[doc_id], doc_id


# Create your views here.
@csrf_exempt
@never_cache
@api_view(['POST'])
def requisition_form(request):
    # try:
    #     response, image_list, file_name, mime_type, file_size = process_input_from_request(request)
    #
    #     doc_class = ""
    #     image = cv2.imdecode(np.fromstring(request.FILES['image'].read(), np.uint8), cv2.IMREAD_UNCHANGED)
    #     test_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #     test_image = cv2.resize(test_image, (128, 128))
    #     test_image = np.array(test_image)
    #     test_image = test_image.astype('float32')
    #     test_image /= 255
    #
    #     if not doc_class:
    #         doc_class, max_confidence_index = doc_classifier(test_image, model_kyc, graph_kyc)
    #     response = {
    #         "document": doc_class,
    #     }
    #
    #     try:
    #         image = request.FILES['image']
    #         image_client = IUtils(request.FILES['image'])
    #         image_client.call_vision()
    #         import pdb;pdb.set_trace()
    #
    #         image_client.rotate_response_bounds(image_, doc_class)
    #     except:
    #         pass
    #
    #     return HttpResponse(json.dumps(response), status=200)
    #
    # except Exception as e:
    #     print(e)
    #     return HttpResponse("final_json", status.HTTP_400_BAD_REQUEST)

    try:
        resp_json = {'status': 'Pass'}
        response, image_list, file_name, mime_type, file_size = process_input_from_request(request)
        print('OCR')
        doc_class = ''

        for i, image_ in enumerate(image_list):
            try:
                image_client = IUtils(image_)
                image_client.call_vision()

                classifier = doc_classifier(image_client.image, model_kyc, graph_kyc)

                any_error = image_client.rotate_response_bounds(image_, doc_class)
                if any_error:
                    response = any_error.as_dict()
                    continue

                serialized = MessageToJson(image_client.response, doc_class, image_client.rotated_img.shape)


            except InvalidArgument as err:
                response = HttpResponse({
                                'status':'FAIL',
                                'status_code':"413",
                                'message':str(err),
                                'trace_back':""
                            })

            except json.decoder.JSONDecodeError:
                response = HttpResponse({
                                'status':'FAIL',
                                'status_code':"413",
                                'message':"JSONDecodeError, No response from ",
                                'trace_back':str(traceback.format_exc())
                            })

            except Exception as err:
                response = HttpResponse({
                                'status':'FAIL',
                                'status_code':"420",
                                'message':str(err),
                                'trace_back':str(traceback.format_exc())
                            })


        return response

    except Exception as err:
        response = HttpResponse({
                        'status':'FAIL',
                        'status_code':"500",
                        'message':str(err),
                        'trace_back':str(traceback.format_exc())
                    })
        return response

