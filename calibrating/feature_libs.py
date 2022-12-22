import boxx
import warnings

with boxx.inpkg():
    from .boards import *

"""把 feature_lib 重命名为 board 后, 向前兼容
"""


class MetaFeatureLib(BaseBoard):
    def __init__(self, *args, **argkws):
        new_class_name = self.__class__.__bases__[0].__name__
        message = f"{self.__class__.__name__} rename to {new_class_name}"
        warnings.warn(message, DeprecationWarning)
        print("Warning:", message)
        super().__init__(*args, **argkws)


class CheckboardFeatureLib(Chessboard):
    def __init__(self, *args, **argkws):
        new_class_name = self.__class__.__bases__[0].__name__
        message = f"{self.__class__.__name__} rename to {new_class_name}"
        warnings.warn(message, DeprecationWarning)
        print("Warning:", message)
        super().__init__(*args, **argkws)


class ArucoFeatureLib(PredifinedArucoBoard1):
    def __init__(self, *args, **argkws):
        new_class_name = self.__class__.__bases__[0].__name__
        message = f"{self.__class__.__name__} rename to {new_class_name}"
        warnings.warn(message, DeprecationWarning)
        print("Warning:", message)
        super().__init__(*args, **argkws)


class CharucoFeatureLib(CharucoBoard):
    def __init__(self, *args, **argkws):
        new_class_name = self.__class__.__bases__[0].__name__
        message = f"{self.__class__.__name__} rename to {new_class_name}"
        warnings.warn(message, DeprecationWarning)
        print("Warning:", message)
        super().__init__(*args, **argkws)
