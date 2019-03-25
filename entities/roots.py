from entities.seed import SeedSection

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
        self.angle = self.__calculate_angle(left_root, right_root)


    def __calculate_angle(self, left_root: RootInfo, right_root: RootInfo):
        return left_root.angleWithHorizontal - right_root.angleWithHorizontal

class RootFinder:
    """
    Finds the root in a SeedSection
    """
    @staticmethod
    def find(section: SeedSection) -> RootAngle:
        """
        Finds information about the roots in the section
        Parameters
        ----------
        section : SeedSection
            The seed section containing the roots we want to find.
        """
        left_root = RootInfo()
        right_root = RootInfo()
        return RootAngle(left_root, right_root)
