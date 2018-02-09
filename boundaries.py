from dolfin import *
from mshr import *
import random
import time
    
L = 10.0
H = 1.0

def Mark_Boundaries():
	# Define boundaries
	class Symmetry(SubDomain):
	    def inside(self, x, on_boundary):
		    return near(x[1], 0.0) or near(x[1], H)

	class Left(SubDomain):
	    def inside(self, x, on_boundary):
		    return near(x[0], 0.0) and on_boundary

	class Right(SubDomain):
	    def inside(self, x, on_boundary):
		    return near(x[0], L) and on_boundary

	class Obstacle(SubDomain):
	    def inside(self, x, on_boundary):
	        return between(x[0], (0.0, L)) and between(x[1], (0.0, H)) and on_boundary

	mesh = Mesh("mesh.xml")
	boundaries = FacetFunction("size_t", mesh)
	boundaries.set_all(0)
	Obstacle().mark(boundaries, 3)
	Symmetry().mark(boundaries, 4)
	Left().mark(boundaries, 1)
	Right().mark(boundaries, 2)

	return boundaries




time1 = time.time()
boundaries = Mark_Boundaries()
time2 = time.time()
# Check generated mesh
# Check boundaries of mesh
plot(boundaries, title = "Boundaries")
print time2-time1
File("boundaries.xml") << boundaries
interactive()

