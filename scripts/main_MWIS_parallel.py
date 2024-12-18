from gurobipy import *
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import numpy as np
import math
import pickle
import threading
import time
import concurrent.futures


#


with open('_candidates.pkl', 'rb') as file:
    candidates = pickle.load(file)
with open('_feasReg.pkl', 'rb') as file:
    marg = pickle.load(file)


#


num_radii = len(candidates[0][0])
vertices = []
for z in range(len(candidates)):
    X = candidates[z][0]
    Y = candidates[z][1]
    Z = candidates[z][2]
    for r in range(num_radii):
        if (len(candidates[z][0][r]) != len(candidates[z][1][r])) or (len(candidates[z][0][r]) != len(candidates[z][2][r])):
            print('*** ERROR! ***')
    for r in range(num_radii):
        for p in range(len(candidates[z][0][r])):
            v = [round(candidates[z][0][r][p],2),round(candidates[z][1][r][p],2),round(candidates[z][2][r][p],2),r]
            vertices.append(v)


#


N = len(vertices)
radii = [0.25,0.5,1.0]
vol = [round(1000*(0.25**3),0),round(1000*(0.5**3),0),1000]
w = [0 for k in range(N)]
for k in range(N):
    for r in range(len(radii)):
        if vertices[k][3] == r:
            w[k] = vol[r]


#


ellipsoids_0 = [[0.25,0.25,0.25],[0.53,0.55,0.3],[0.65,0.66,0.54],[0.77,0.78,0.65],[0.88,0.9,0.76],[1,1.04,0.87],[1.26,1.29,0.97],[1.55,1.6,1.11],[2.15,2.32,1.25],[3.42,4.59,1.4],[10,10,10]]
ellipsoids_1 = [[0.5,0.5,0.5],[0.61,0.62,0.58],[0.72,0.72,0.66],[0.83,0.82,0.74],[0.93,0.93,0.82],[1.09,1.06,0.9],[1.33,1.3,0.98],[1.65,1.6,1.12],[2.26,2.3,1.26],[3.62,4.51,1.41],[10,10,10]]
ellipsoids_2 = [[1.0,1.0,1.0],[1.13,1.15,1.08],[1.25,1.28,1.16],[1.37,1.41,1.24],[1.49,1.58,1.32],[1.76,1.85,1.39],[2.05,2.22,1.47],[2.63,2.86,1.6],[3.73,4.27,1.76],[5.75,8.82,1.91],[10,10,10]]
ellipsoids = [ellipsoids_0, ellipsoids_1, ellipsoids_2]
doses = [50,45,40,35,30,25,20,15,10,5,0] 
spacing = 0.3
calib_factor = 0.85
valley_dose = calib_factor*20


#


def is_inside_ellipsoid(point, center, a, b, c):
    x, y, z = point
    dx, dy, dz = center
    return ((x - dx)**2 / a**2 + (y - dy)**2 / b**2 + (z - dz)**2 / c**2) <= 1
#
def find_intersection(ellipsoid, center, point_e):
    a, b, c = ellipsoid
    d_x, d_y, d_z = center
    e_x, e_y, e_z = point_e
    dir_x = e_x - d_x
    dir_y = e_y - d_y
    dir_z = e_z - d_z
    A = (dir_x / a) ** 2 + (dir_y / b) ** 2 + (dir_z / c) ** 2
    B = 0
    C = -1
    t_values = np.roots([A, B, C])
    intersection_points = [(d_x + t * dir_x, d_y + t * dir_y, d_z + t * dir_z) for t in t_values if np.isreal(t)]
    temp = 1000
    final_point = intersection_points[0]
    for p in intersection_points:
        dist = math.sqrt((p[0] - e_x) ** 2 + (p[1] - e_y) ** 2 + (p[2] - e_z) ** 2)
        if dist < temp:
            temp = dist
            final_point = p
    return final_point
#
def interpolate_dose(point, center, ellipsoids, doses):
    inside = [is_inside_ellipsoid(point, center, ellipsoid[0], ellipsoid[1], ellipsoid[2]) for ellipsoid in ellipsoids]
    for i in range(len(inside) - 1):
        if inside[i] != inside[i + 1]:
            dose_diff = doses[i] - doses[i + 1]
            ellipsoid_outer = (ellipsoids[i+1][0], ellipsoids[i+1][1], ellipsoids[i+1][2])
            ellipsoid_inner = (ellipsoids[i][0], ellipsoids[i][1], ellipsoids[i][2])
            point_outer = find_intersection(ellipsoid_outer, center, point)
            point_inner = find_intersection(ellipsoid_inner, center, point)
            dist_outer = math.sqrt((point[0] - point_outer[0]) ** 2 + (point[1] - point_outer[1]) ** 2 + (point[2] - point_outer[2]) ** 2)
            dist_inner = math.sqrt((point[0] - point_inner[0]) ** 2 + (point[1] - point_inner[1]) ** 2 + (point[2] - point_inner[2]) ** 2)
            return doses[i + 1] + dose_diff * (dist_outer) / (dist_outer + dist_inner) 
    return doses[0] if inside[0] else doses[-1]
#
def find_mid_point(v,u):
    dir_x = v[0] - u[0]
    dir_y = v[1] - u[1]
    dir_z = v[2] - u[2]
    mid = [u[0]+0.5*dir_x, u[1]+0.5*dir_y, u[2]+0.5*dir_z]
    return mid
#
def const_gen(i):
    temp = []
    for j in range(i+1,N):
        v = vertices[i]
        u = vertices[j]
        distance = math.sqrt((v[0] - u[0]) ** 2 + (v[1] - u[1]) ** 2 + (v[2] - u[2]) ** 2)
        if distance <= radii[v[3]] + radii[u[3]]:
            temp.append([i,j])
        else:
            point = find_mid_point(v,u)
            dose_at_point_v = interpolate_dose(point, [v[0],v[1],v[2]], ellipsoids[v[3]], doses)
            dose_at_point_u = interpolate_dose(point, [u[0],u[1],u[2]], ellipsoids[u[3]], doses)
            if (dose_at_point_v + dose_at_point_u > valley_dose):
                temp.append([i,j])
    return temp


#


if __name__ == "__main__":
    startTime = time.time()
    #
    edges = []
    try:
        with concurrent.futures.ProcessPoolExecutor() as executor:
            futures = [executor.submit(const_gen, i) for i in range(N-1)]
            AllResults = [future.result() for future in concurrent.futures.as_completed(futures)]
    except Exception as e:
        print(f"An exception occurred: {e}")
    for i in range(N-1):
        eachResult = AllResults[i]
        for ele in eachResult:
            edges.append(ele)
    #
    tumorIS = Model()
    tumorIS.Params.TIME_LIMIT = 1*3600
    #tumorIS.Params.MIPGap = 1*10**(-3)
    tumorIS.Params.NoRelHeurTime = 1*60
    #tumorIS.Params.Presolve = 1
    #tumorIS.Params.Method = 2
    #tumorIS.Params.NodeMethod = 0
    #tumorIS.Params.MIPFocus = 3
    #tumorIS.Params.IntegralityFocus = 1
    #tumorIS.Params.LogFile = "output_log.txt"
    ## VARS
    x = tumorIS.addVars(N, vtype = GRB.BINARY, name = "x")
    ## OBJ
    tumorIS.setObjective(quicksum(w[k]*x[k] for k in range(N)), GRB.MAXIMIZE)
    ## CONSTS
    for edge in edges:
        tumorIS.addConstr(x[edge[0]] + x[edge[1]] <= 1)
    print("Model built in:", time.time() - startTime, "seconds")
    ## SOLVE
    solTime = time.time()
    status = tumorIS.optimize()
    print("Model Solved in:", time.time() - solTime, "seconds")
    #tumorIS.write("model.lp")
    ## OUTPUT
    print("Calibrated Valley Dose Factor = "+str(calib_factor))
    print("Number of Edges = "+str(len(edges))) 
    print('=============================================')
    print('Objective value = '+str(tumorIS.getObjective().getValue()))
    print('=============================================')
    print('Solution (centers of the spheres):')
    C = []
    R = []
    for v in tumorIS.getVars():    
        ind = int(v.varName[v.VarName.find('[')+1:v.VarName.find(']')])
        if v.x > 0.5:
            r = round(radii[vertices[ind][3]],2)
            R.append(r)
            loc = (vertices[ind][0],vertices[ind][1],vertices[ind][2])
            C.append(loc)
            print("(x,y,z) = "+str(loc)+" :  D = "+str(2*r)+" cm")
    #
    tumor = pd.read_csv('CH_GTV_GRID.csv')
    xT = tumor['X']
    yT = tumor['Y']
    zT = tumor['Z']
    xmarg = []
    ymarg = []
    zmarg = []
    for m in range(len(marg)):
        xmarg.append(marg[m][0])
        ymarg.append(marg[m][1])
        zmarg.append(marg[m][2])
    #
    fig = plt.figure(figsize=(16,16))
    p = fig.add_subplot(111, projection='3d')
    p.scatter(xT, yT, zT, color='b', marker='.')
    p.scatter(xmarg, ymarg, zmarg, color='lightcoral', marker='3', s=900*spacing)
    for i in range(len(R)):
        center = C[i]
        radius = R[i]    
        u = np.linspace(0, 2 * np.pi, 100)
        v = np.linspace(0, np.pi, 100) 
        x = center[0] + radius * np.outer(np.cos(u), np.sin(v))
        y = center[1] + radius * np.outer(np.sin(u), np.sin(v))
        z = center[2] + radius * np.outer(np.ones(np.size(u)), np.cos(v)) 
        p.plot_surface(x, y, z, color='r')
    p.set_xlabel('X')
    p.set_ylabel('Y')
    p.set_zlabel('Z')
    p.view_init(elev=15, azim=-45, roll=0)
    p.set_aspect('equal')
    plt.savefig('output_IS_3D.png')
    plt.close(fig)
    #
    fig = plt.figure(figsize=(16,16))
    p = fig.add_subplot(111, projection='3d')
    p.scatter(xT, yT, zT, color='b', marker='.')
    p.scatter(xmarg, ymarg, zmarg, color='lightcoral', marker='3', s=900*spacing, lw = 3)
    for i in range(len(R)):
        center = C[i]
        radius = R[i]    
        u = np.linspace(0, 2 * np.pi, 100)
        v = np.linspace(0, np.pi, 100)    
        x = center[0] + radius * np.outer(np.cos(u), np.sin(v))
        y = center[1] + radius * np.outer(np.sin(u), np.sin(v))
        z = center[2] + radius * np.outer(np.ones(np.size(u)), np.cos(v))    
        p.plot_surface(x, y, z, color='r')
    p.set_xlabel('X')
    p.set_ylabel('Y')
    p.set_zlabel('Z')
    p.view_init(elev=0, azim=270, roll=0)
    p.set_aspect('equal')
    plt.savefig('output_IS_XZ.png') 
    plt.close(fig) 
    #
    fig = plt.figure(figsize=(16,16))
    p = fig.add_subplot(111, projection='3d')
    p.scatter(xT, yT, zT, color='b', marker='.')
    p.scatter(xmarg, ymarg, zmarg, color='lightcoral', marker='3', s=900*spacing, lw = 3)
    for i in range(len(R)):
        center = C[i]
        radius = R[i]
        u = np.linspace(0, 2 * np.pi, 100)
        v = np.linspace(0, np.pi, 100)    
        x = center[0] + radius * np.outer(np.cos(u), np.sin(v))
        y = center[1] + radius * np.outer(np.sin(u), np.sin(v))
        z = center[2] + radius * np.outer(np.ones(np.size(u)), np.cos(v))    
        p.plot_surface(x, y, z, color='r')
    p.set_xlabel('X')
    p.set_ylabel('Y')
    p.set_zlabel('Z')
    p.view_init(elev=0, azim=0, roll=0)
    p.set_aspect('equal')
    plt.savefig('output_IS_YZ.png') 
    plt.close(fig) 
    #
    fig = plt.figure(figsize=(16,16))
    p = fig.add_subplot(111, projection='3d')
    p.scatter(xT, yT, zT, color='b', marker='.')
    p.scatter(xmarg, ymarg, zmarg, color='lightcoral', marker='3', s=900*spacing, lw = 3)
    for i in range(len(R)):
        center = C[i]
        radius = R[i]
        u = np.linspace(0, 2 * np.pi, 100)
        v = np.linspace(0, np.pi, 100)    
        x = center[0] + radius * np.outer(np.cos(u), np.sin(v))
        y = center[1] + radius * np.outer(np.sin(u), np.sin(v))
        z = center[2] + radius * np.outer(np.ones(np.size(u)), np.cos(v))    
        p.plot_surface(x, y, z, color='r')
    p.set_xlabel('X')
    p.set_ylabel('Y')
    p.set_zlabel('Z')
    p.view_init(elev=90, azim=90, roll=180)
    p.set_aspect('equal')
    plt.savefig('output_IS_XY.png') 
    plt.close(fig) 

