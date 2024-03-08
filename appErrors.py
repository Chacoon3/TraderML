from fastapi.responses import PlainTextResponse   


__BadFormatError = PlainTextResponse(status_code=400, content="Data format does not match.")
def badFormatError():
    return __BadFormatError