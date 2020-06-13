from django.conf.urls import include, url
from .views import requisition_form

urlpatterns = [
    url(r'api/v1/healthcare/', requisition_form, name="requisition_form"),

]
