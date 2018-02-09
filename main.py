from dolfin import *
import numpy as np
import matplotlib.pyplot as plt
#parameters["form_compiler"]["cpp_optimize"] = True
set_log_level(35)
#parameters["refinement_algorithm"] = "plaza_with_parent_facets"

def error(u, ur, V, name):
    error = open("./error.txt", "a")
    uh = ur#project(ur, V)
    e = project(u - uh, V)
    File(name + ".pvd") << e
    #absolut pogreshnost
    eabs = norm(e, 'L2')
    #otnosit pogreshnost
    erel  = norm(e, 'L2')  / norm(uh, 'L2')
    #plot(eabs, title="Absolute Error for " + name)
    #plot(erel, title="Relative Error for " + name)
    title = 'abs = ' + str(eabs) + ' rel = ' + str(erel)
    e.rename('e' + str(V.mesh().num_vertices()), title)
    plot(e)
    error.write(str(name) + "\t" + str(eabs) + "\t" + str(erel) + "\n")


# Mesh
mesh = Mesh("mesh.xml")
# Extract mesh subdomains and boundaries
boundaries = MeshFunction("size_t", mesh, "boundaries.xml")
#mesh = refine(mesh)
#subdomains = adapt(subdomains, mesh)
#boundaries = adapt(boundaries, mesh)
# get information
mesh.init()
ncells = [  mesh.num_vertices(), mesh.num_edges(), mesh.num_faces(), mesh.num_facets(), mesh.num_cells() ]
print(ncells)


# GVV hydrodynamics
ds = Measure("ds")[boundaries]
n = FacetNormal(mesh)

# Define variational problem
V = VectorFunctionSpace(mesh, "CG", 2)
Q = FunctionSpace(mesh, "CG", 1)

# Make a mixed space
W = MixedFunctionSpace([V, Q])

# Define unknown and test function(s)
w = Function(W)
(u, p) = split(w)
(v, q) = TestFunctions(W)

# Define boundary conditions
zero_vec = Expression(("0.0", "0.0"), degree=1)

bcs = [DirichletBC(W.sub(0).sub(1), Constant(0.0), boundaries, 1),\
       DirichletBC(W.sub(0).sub(0), Constant(1.0), boundaries, 1),\
       DirichletBC(W.sub(0).sub(1), Constant(0.0), boundaries, 4),\
       DirichletBC(W.sub(0), zero_vec, boundaries, 3),\
       DirichletBC(W.sub(1), Constant(0.0), boundaries, 2)]       

# Form definition
Re = 150.0
g = Constant(1.0)

F = inner(grad(u)*u, v)*dx + 1.0/Re*inner(grad(u), grad(v))*dx + div(u)*q*dx - div(v)*p*dx

# Compute solution
solve(F == 0, w, bcs, solver_parameters={"newton_solver":
                      {"relative_tolerance": 1e-8, "relaxation_parameter": 1.0}})
(u, p) = w.split()
plot(u, title = "Velocity")
plot(p, title = "Pressure")

File("./res/u.pvd")<<u
File("./res/p.pvd")<<p


# CHEKING
P = FunctionSpace(mesh, 'CG', 1)


nx = Expression(('1.0','0.0'), degree=1, domain = mesh)
ux = dot(u,nx)

ny = Expression(('0.0','1.0'), degree=1, domain = mesh)
uy = dot(u,ny)

plot(ux, title = "Velocity u_x")
plot(uy, title = "Velocity u_y")

'''
# GVV psi
psi_t = TestFunction(P)
psi = TrialFunction(P)

# Define inlet boundary conditions psi = y
psi_in = Expression("x[1]", degree=1)
psi_bot = 0.0
psi_top = 1.0

bc = [DirichletBC(P, psi_in, boundaries, 1), \
      DirichletBC(P, Constant(psi_bot), boundaries, 5), \
      DirichletBC(P, Constant(psi_bot), boundaries, 6), \
      DirichletBC(P, Constant(psi_top), boundaries, 2), \
      DirichletBC(P, Constant(psi_top), boundaries, 3)]
      
F = inner(grad(psi), grad(psi_t))*dx - (uy.dx(0) - ux.dx(1))*psi_t*dx
a = lhs(F)
L = rhs(F)

psiu = Function(P)

solve(a == L, psiu, bc)

plot(psiu, title = "PSI")

File("./res/psi.pvd")<<psiu
'''

# Perenos primesi GVV

# Define variational problem
V = FunctionSpace(mesh, "CG", 1)

# Make a mixed space
W = MixedFunctionSpace([V, V])

# Define unknown and test function(s)
w = project(Constant((0, 0)), W)
(c, m) = (w[0], w[1])
(v, q) = TestFunctions(W)

Pe = 10.0

tau = 0.01
T = 3.0
t = 0.0
N = int(T/tau)

w_old = Function(W)
w_old.assign(w)
(c0, m0) = (w_old[0], w_old[1])

# Stream function
#psi = Function(V, "psi-.xml")
#u = as_vector([psi.dx(1), -psi.dx(0)])

bc = DirichletBC(W.sub(0), Constant(1.0), boundaries, 1)
ds = Measure("ds")[boundaries]

Fc = (c-c0)/tau*v*dx + 0.5*1/Pe*inner(grad(c+c0), grad(v))*dx - 0.5*(c+c0)*inner(u, grad(v))*dx + 0.5*(c+c0)*inner(u, n)*v*ds(2) + (m-m0)/tau*v*ds(3) + (m-m0)/tau*v*ds(3)


Sh1 = 0.001
Sh2 = 1.0
ISh1 = 1./Sh1
ISh2 = 0.0


K_a = 0.005
K_d = 0.05
m_inf = 0.05

#Langmuir's isotherm
Fm = (m-m0)/tau*q*dx - 0.5*K_a*(c+c0)*(1-0.5*(m+m0)/m_inf)*q*dx + 0.5*K_d*(m+m0)*q*dx

#Henry's isotherm
#Fm = (m-m0)/tau*q*dx - 0.5*K_a*(c+c0)*q*dx + 0.5*K_d*(m+m0)*q*dx

#Sherwood's isotherm (mb nazivaetsya po drugomy? dlya okisleniya)
#Fm = (m-m0)/tau*q*dx - 0.5/(ISh1 + 0.5*ISh2*(m+m0))*(c+c0)*q*dx

F = Fc + Fm

c_pvd = File("./res/c.pvd")
m_pvd = File("./res/m.pvd")

td = np.zeros(N + 1, 'float')
cd = np.zeros(N + 1, 'float')

mtd = np.zeros(N + 1, 'float')
mcd = np.zeros(N + 1, 'float')

(c, m) = w.split()

for i in range(N):
    t += tau
    
    solve(F == 0, w, bc, solver_parameters={"newton_solver":
                      {"relative_tolerance": 1e-8, "relaxation_parameter": 1.0}})
    w_old.assign(w)
    
    c_pvd << c
    m_pvd << m
    
    I = assemble(c * ds(2))
    td[i + 1] = t
    cd[i + 1] = I
    
    mI = assemble(m * ds(3))
    mtd[i + 1] = t
    mcd[i + 1] = mI
    print t, I, mI

plot(c, title="c")
plot(m, title="m")

m1 = m

plt.figure(1)
plt.plot(td,cd)
#plt.axis([0, T, 0.0, 1.])
plt.xlabel('$t$')
plt.ylabel('$c$')

plt.figure(2)
tds = np.zeros(N, 'float')
cds = np.zeros(N, 'float')
for i in range(N):
    tds[i] = 0.5*(td[i + 1] + td[i])
    cds[i] = 0.5*(cd[i + 1] - cd[i])/tau
plt.plot(tds,cds)
#plt.axis([0, T, 0.0, 0.2])
plt.xlabel('$t$')
plt.ylabel('$c$')


plt.figure(3)
plt.plot(mtd,mcd)
#plt.axis([0, T, 0.0, 1.])
plt.xlabel('$t$')
plt.ylabel('$m$')

plt.figure(4)
mtds = np.zeros(N, 'float')
mcds = np.zeros(N, 'float')
for i in range(N):
    mtds[i] = 0.5*(mtd[i + 1] + mtd[i])
    mcds[i] = 0.5*(mcd[i + 1] - mcd[i])/tau
plt.plot(mtds,mcds)
#plt.axis([0, T, 0.0, 0.2])
plt.xlabel('$t$')
plt.ylabel('$m$')

plt.show()
interactive()
