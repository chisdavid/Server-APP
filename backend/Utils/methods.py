import json
from collections import namedtuple


def customDecoder(obj):
    return namedtuple('X', obj.keys())(*obj.values())


def jsonToObj(jsonObject):
    return json.loads(jsonObject, object_hook=customDecoder)
