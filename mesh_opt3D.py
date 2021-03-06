from dolfin import *
from mshr import *
import random
import time

def Generate_Mesh():
	# Create mesh
    
	## Parameters of an in-line tube bundle
	time_c = time.time()
	N = 10             # number of circles
	D_max = 0.1         # max diameter
	D_min = 0.01        # min diameter
	res = 50           # grid resolution
	R_max = D_max/2.0   # max radius
	R_min = D_min/2.0   # min radius
	dd = R_min*2        #(R_max+R_min)/2/2     # distance between holes
	raup = 0.0          #R_max + DOLFIN_EPS   #distance between horizontal boundaries
	ramk = 0.1          #distance between vertical boundaries
	
	
	H = 1.0     # height of domain
	L = 10.0     # lenght of domain
	
	kx = H/2.0
	ky = L/kx
	a = []
	for i in range(0, int(ky)):
	    a.append(i*2)
	
	def segments(R):
	    Rrazn = R_max - R_min 
	    if(R<=Rrazn/5):
	        return 12
	    elif(R>Rrazn/5 and R<=Rrazn/5*2):
	        return 17
	    elif(R>Rrazn/5*2 and R<=Rrazn/5*3):
	        return 22
	    elif(R>Rrazn/5*3 and R<=Rrazn/5*4):
	        return 27
	    else:
	        return 32
	
	def kvadrant(x, y):
	    r = x/kx
	    if(y>=kx): return int(a[int(r)]+1)
	    else: return int(a[int(r)])
	
	def neighbor(kvad):
	    lst = []
	    if(kvad % 2 == 0 and kvad > 0 and kvad < int(ky*2-1)):
	        lst = [kvad-2, kvad-1, kvad, kvad+1, kvad+2, kvad+3]
	    if(kvad % 2 != 0 and kvad > 0 and kvad < int(ky*2-1)):
	        lst = [kvad-3, kvad-2, kvad-1, kvad, kvad+1, kvad+2]
	    if(kvad == 0):
	        lst = [kvad, kvad+1, kvad+2, kvad+3]
	    if(kvad == 1):
	        lst = [kvad-1, kvad, kvad+1, kvad+2]
	    if(kvad == int(ky*2-1)):
	        lst = [kvad-3, kvad-2, kvad-1, kvad]
	    if(kvad == int(ky*2-1)-1):
	        lst = [kvad-2, kvad-1, kvad, kvad+1]
	    return lst
	
	
	
	domain = Box(Point(0.0, 0.0, 0.0), Point(L, H, H))
	
	
	
	X = []
	Y = []
	radius = []
	
	for i in range(0,int(ky)*2):
	    X.append([])
	    Y.append([])
	    radius.append([])
	
	x = random.uniform(ramk, L-ramk)
	y = random.uniform(raup, H-raup)
	R = random.uniform(R_min, R_max)
	x = round(x, 3)
	y = round(y, 3)
	R = round(R, 3)
	
	circ = Cylinder(Point(x, y, H), Point(x, y, 0), R, R, segments(R))
	domain = domain - circ
	
	
	kv = kvadrant(x,y)
	
	X[kv].append(x)
	Y[kv].append(y)
	radius[kv].append(R)
	
	i = 1
	while i<N:
	    x = random.uniform(ramk, L-ramk)
	    y = random.uniform(raup, H-raup)
	    R = random.uniform(R_min, R_max)
	    x = round(x, 3)
	    y = round(y, 3)
	    R = round(R, 3)
	    lst = []
	    kv = kvadrant(x,y)
	    lst = neighbor(kv)
	    k = 0
	    for w in range(0, len(lst)):
	        j = 0
	        while(j<len(X[lst[w]])):
	            if(sqrt((X[lst[w]][j]-x)**2 + (Y[lst[w]][j]-y)**2)<=(radius[lst[w]][j]+R+dd)):
	                k+=1
	                break
	            else:
	                j+=1
	        
	    if(k==0):
	        X[kv].append(x)
	        Y[kv].append(y)
	        radius[kv].append(R)
	        circ = Cylinder(Point(x, y, H), Point(x, y, 0), R, R, segments(R))
	        domain = domain - circ
	        i+=1
	
	time_e = time.time()
	print "Geometry generated by " + str(time_e-time_c) + " seconds." + "\n" + "Starting generate mesh..."
	mesh = generate_mesh(domain, float(res))
	info(mesh)
	return mesh


time1 = time.time()
mesh = Generate_Mesh()
time2 = time.time()
# Check generated mesh
plot(mesh, title = "Mesh")
print "Mesh generated by " + str(time2-time1) + " seconds."
File("mesh.xml") << mesh
interactive()
