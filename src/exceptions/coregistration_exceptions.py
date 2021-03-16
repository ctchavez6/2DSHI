

class InvalidROIException(Exception):
    """
    Exception raised when there is an error while trying to create the ROI bounds.

    """

    def __init__(self, message="Exception occurred while generating ROI bounds:"):
        self.message = message
        super().__init__(self.message)


class BeamNotGaussianException(Exception):
    """
    Exception raised when there is an error while trying to create the ROI bounds.

    """

    def __init__(self, message="Exception occurred while generating ROI bounds:"):
        self.message = message
        super().__init__(self.message)