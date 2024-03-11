from fastapi.responses import PlainTextResponse   


__BadFormatError = PlainTextResponse(status_code=400, content="Data format does not match.")
def badFormatError():
    return __BadFormatError

def unhandledError(error: Exception):
    if len(error.args) > 0:
        return PlainTextResponse(status_code=500, content=error.args[0])
    return  PlainTextResponse(status_code=500)