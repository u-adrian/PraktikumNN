class GpuNotFoundError(Exception):
    pass


class NoGeneratorError(Exception):
    pass


class NoDiscriminatorError(Exception):
    pass


class NumEpochsError(Exception):
    pass


class BatchSizeError(Exception):
    pass


class LearningRateError(Exception):
    pass


class InvalidLossError(Exception):
    pass


class InvalidNoiseSizeError(Exception):
    pass


class InvalidArgumentError(Exception):
    pass