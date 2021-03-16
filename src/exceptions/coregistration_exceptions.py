

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

class ROITooSmallException(Exception):
    """
    Exception raised when there is an error while trying to create the ROI bounds.

    """

    def __init__(self, message="Your ROI does not meet the 50x50 pixel requirements:"):
        self.message = message
        super().__init__(self.message)
