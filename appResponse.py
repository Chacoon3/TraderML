from fastapi.responses import PlainTextResponse   
import json

def __makeResponseBody(data = None, error = None):
    return json.dumps(dict(
        data=data, error = error
    ))

__BadFormatError = PlainTextResponse(status_code=400, content=__makeResponseBody(error="Data format does not match."))
def badFormatResponse():
    return __BadFormatError

def unhandledErrorResponse(error: Exception):
    if len(error.args) > 0:
        return PlainTextResponse(status_code=500, content=__makeResponseBody(error=error.args[0]))
    return  PlainTextResponse(status_code=500, content=__makeResponseBody(error="Server error."))

def dataResponse(data):
    return PlainTextResponse(status_code=200, content=__makeResponseBody(data=data))