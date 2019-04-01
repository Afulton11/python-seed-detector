import math

class Location:
    def __init__(self, x: int, y: int):
        self.x = int(round(x))
        self.y = int(round(y))

    def distance(self, other) -> float:
        vx = self.x - other.x
        vy = self.y - other.y
        return math.sqrt(vx**2 + vy**2)
    
    def __str__(self):
        return '(%d, %d)' % (self.x, self.y)

    def __iter__(self):
        yield self.x
        yield self.y
