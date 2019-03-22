

class Roots:
    """
    Contains Attributes and Methods useful for finding Left-most and Right-most roots.
    """

class RootInfo:
    """
    contains information about a root of a seed.
    """
    angleWithHorizontal = None


class RootAngle:
    """
    Contains information about the root angle of a seed.
    ...

    Attributes
    ----------
    angle : radians
        Represents a root angle for a seed.
    """
    angle = None

    def __init__(self, left_root: RootInfo, right_root: RootInfo):
        self.__left_root = left_root
        self.__right_root = right_root
        self.angle = self.__calculateAngle(left_root, right_root)


    def __calculateAngle(self, left_root: RootInfo, right_root: RootInfo):
        return left_root.angleWithHorizontal - right_root.angleWithHorizontal