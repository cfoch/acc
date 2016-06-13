def classifier_from_settings(classifier_settings):
    """
    Creates a classifier with params set in settings.
    """
    klass = classifier_settings["classifier"]
    args = classifier_settings["args"]
    return klass(**args)
