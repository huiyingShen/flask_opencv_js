symbols = {'pi': 3.1415926, 'e': 2.7182818}
class Circle():
    def __init__(self, radius):
        self.radius = radius

    def scale(self,a):
        self.radius *= a
        return self.radius

    @property
    def area(self):
        a = symbols['pi'] * self.radius**2
        print('radius, area = ', self.radius, a)
        return a
print(Circle(2).area)