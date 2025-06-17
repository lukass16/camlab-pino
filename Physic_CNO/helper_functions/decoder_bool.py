#decoder json
def custom_bool_decoder(obj):
    if isinstance(obj, str):
        if obj.lower() == "true":
            return True
        elif obj.lower() == "false":
            return False
    return obj