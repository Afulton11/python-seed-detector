import math

def max_int():
    return 2_147_483_000

def clamp_to_c_int(val) -> int:
    intValue = int(round(val))
    
    return max(-max_int(), min(intValue, max_int()))

class Location:

    @staticmethod
    def from_tuple(tp: tuple):
        return Location(tp[0], tp[1])

    def __init__(self, x: float, y: int):
        self.x = x
        self.y = y

    def distance(self, other) -> float:
        vx = self.x - other.x
        vy = self.y - other.y

        try:
            return math.sqrt(vx**2 + vy**2)
        except ValueError:
            return 2_147_483_000

    def average(self, other):
        def avgValue(x_1, x_2) -> int:
            return (x_1 + x_2) / 2

        return Location(
            avgValue(self.x, other.x),
            avgValue(self.y, other.y)
        )
    
    def to_tuple(self):
        return tuple(self.x, self.y)

    def x_int(self):
        return clamp_to_c_int(self.x)

    def y_int(self):
        return clamp_to_c_int(self.y)
    
    def __str__(self):
        return '(%d, %d)' % (self.x, self.y)

    def __iter__(self):
        yield self.x_int()
        yield self.y_int()
