from __future__ import division
from solid import *
from solid.utils import *
import cv2
import itertools
import numpy as np
import heapq
from euclid3 import *
from os import listdir
from subprocess import call


plan = "./plan_predict/1_predict.png"
north = "./elevs_predict/2_predict.png"
south = "./elevs_predict/3_predict.png"
east = "./elevs_predict/4_predict.png"
west = "./elevs_predict/1_predict.png"

def merging(contour_array):
    return list(itertools.chain(*contour_array))

def plan_contouring(file_name):
    img = cv2.imread(file_name)
    imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret,thresh = cv2.threshold(imgray,127,255,cv2.THRESH_BINARY)
    im2, cnts, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    areas = []
    for cnt in cnts:
        areas.append(cv2.contourArea(cnt))
    
    contours_all = []
    for cnt in cnts:
        if cv2.contourArea(cnt) < max(areas) - 1:
            contours_all.append(cnt)

    areas.remove(max(areas))
    
    contours_wall = []
    contours_floor = []
    contours_column = []

    for cnt in cnts:
        if cv2.contourArea(cnt) < max(areas) - 1 and cv2.contourArea(cnt) > 500: #columns should be reexamined!
            contours_wall.append(cnt)
        elif cv2.contourArea(cnt) == max(areas):
            contours_floor.append(cnt)
        elif cv2.contourArea(cnt) < 500:
            contours_column.append(cnt)

    return contours_wall, contours_floor, contours_column, contours_all

plan_wall = plan_contouring(plan)[0]
plan_floor = plan_contouring(plan)[1]
plan_column = plan_contouring(plan)[2]
plan_all = plan_contouring(plan)[3]

def elevation_contouring(elev):
    img = cv2.imread(elev)

    imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret,thresh = cv2.threshold(imgray,127,255,cv2.THRESH_BINARY)
    im2, cnts, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    
    areas = []
    for cnt in cnts:
        areas.append(cv2.contourArea(cnt))

    areas.remove(max(areas))
        
    contours_wall = []
    contours_opening = []

    for cnt in cnts:
        if cv2.contourArea(cnt) < max(areas) + 1  and cv2.contourArea(cnt) > np.mean(areas):
            contours_wall.append(cnt)
        elif cv2.contourArea(cnt) < np.mean(areas):
            contours_opening.append(cnt)

    return contours_wall, contours_opening

north_elev = elevation_contouring(north)[0]
north_open = elevation_contouring(north)[1]

south_elev = elevation_contouring(south)[0]
south_open = elevation_contouring(south)[1]

east_elev = elevation_contouring(east)[0]
east_open = elevation_contouring(east)[1]

west_elev = elevation_contouring(west)[0]
west_open = elevation_contouring(west)[1]

def dimensions(points):
    all_x = []
    all_y = []
    for p in merging(points):
        all_x.append(p[0][0])
        all_y.append(p[0][1])
    dimx = max(all_x) - min(all_x)
    dimy = max(all_y) - min(all_y)
    return round(dimx), round(dimy)

def scaling(plan, elev):
    plan_dim = dimensions(plan)
    elev_dim = dimensions(elev)
    if elev == north_elev or elev==north_open or elev == south_elev or elev == south_open:
        scale = plan_dim[1] / elev_dim[0]
    else:
        scale = plan_dim[0] / elev_dim[0]
    return scale

scaled_north_elev = [i * scaling(plan_all, north_elev) for i in north_elev]
scaled_north_open = [i * scaling(plan_all, north_elev) for i in north_open]

scaled_south_elev = [i * scaling(plan_all, south_elev) for i in south_elev]
scaled_south_open = [i * scaling(plan_all, south_elev) for i in south_open]

scaled_east_elev = [i * scaling(plan_all, east_elev) for i in east_elev]
scaled_east_open = [i * scaling(plan_all, east_elev) for i in east_open]

scaled_west_elev = [i * scaling(plan_all, west_elev) for i in west_elev]
scaled_west_open = [i * scaling(plan_all, west_elev) for i in west_open]

def plan_poly(coordinates):
    poly=[]
    for coord in coordinates:
        poly.append(mirror([0,1,0])(polygon(merging(coord.tolist()))))
    return poly

def b_box(points):
    all_x = []
    all_y = []
    for p in merging(points):
        all_x.append(p[0][0])
        all_y.append(p[0][1])
    return [[min(all_x), min(all_y)], [max(all_x), min(all_y)], [max(all_x), max(all_y)], [min(all_x), max(all_y)]]

poly_wall = plan_poly(plan_wall)
poly_floor = plan_poly(plan_floor)
poly_column = plan_poly(plan_column)

wall_height = dimensions(scaled_north_elev)[1]

model = mirror([0,1,0])(linear_extrude(wall_height)(poly_floor) - linear_extrude(wall_height)(poly_wall) + \
        linear_extrude(wall_height)(poly_column) + color(Red)(linear_extrude(5)(poly_floor)))

bb_plan_coords = b_box(plan_all)

bb_plan_ext = linear_extrude(10)(polygon(bb_plan_coords))

bb_north_elev_coords = b_box(scaled_north_elev)
bb_south_elev_coords = b_box(scaled_south_elev)
bb_east_elev_coords = b_box(scaled_east_elev)
bb_west_elev_coords = b_box(scaled_west_elev)

random_elev_width = 20 #this can be any number since we do not need a specific elevation width.

def opening_poly(opening_coords):
    opening_poly=[]
    for coord in opening_coords:
        opening_poly.append(polygon(merging(coord.tolist())))
    return opening_poly

north_ext = rotate(90, [-1,0,0])(rotate(180 , [0,1,0])(linear_extrude(200, center=True)(opening_poly(scaled_north_open))))
south_ext = rotate(90, [1,0,0])(rotate(180 , [1,0,0])(linear_extrude(200, center=True)(opening_poly(scaled_south_open))))
east_ext = rotate(90,[-1,0,0])(rotate(90 , [0,1,0])(linear_extrude(200, center=True)(opening_poly(scaled_east_open))))
west_ext = rotate(90,[-1,0,0])(rotate(90 , [0,-1,0])(linear_extrude(200, center=True)(opening_poly(scaled_west_open))))

def elevation_extrusion_rotation(obj):
    rot = []
    if obj == east_ext:
        east_elev_vec = bb_east_elev_coords[2]
        plan_vec = bb_plan_coords[1]
        target = plan_vec + [0]
        source = [0] + [-east_elev_vec[0]] + [-east_elev_vec[1]]
        rot = [(target[0] - source[0] + random_elev_width) , (target[1] - source[1]) , \
                (target[2] - source[2])]
    if obj == west_ext:
        west_elev_vec = bb_west_elev_coords[3]
        plan_vec = bb_plan_coords[0]
        target = plan_vec + [0]
        source = [0] + [west_elev_vec[0]] + [-west_elev_vec[1]]
        rot = [(target[0] - source[0] - random_elev_width) , (target[1] - source[1]) , \
                (target[2] - source[2])]
    if obj == south_ext:
        south_elev_vec = bb_south_elev_coords[3]
        plan_vec = bb_plan_coords[3]
        target = plan_vec + [0]
        source = [south_elev_vec[0]] + [0] + [-south_elev_vec[1]]
        rot = [(target[0] - source[0]) , (target[1] - source[1] + random_elev_width ) , \
                (target[2] - source[2])]
    if obj == north_ext:
        north_elev_vec = bb_north_elev_coords[2]
        plan_vec = bb_plan_coords[0]
        target = plan_vec + [0]
        source = [-north_elev_vec[0]] + [0] + [-north_elev_vec[1]]
        rot = [(target[0] - source[0]) , (target[1] - source[1] - random_elev_width) , \
                (target[2] - source[2])]
    return rot

n_ext = translate(elevation_extrusion_rotation(north_ext))(north_ext)
s_ext = translate(elevation_extrusion_rotation(south_ext))(south_ext)
e_ext = translate(elevation_extrusion_rotation(east_ext))(east_ext)
w_ext = translate(elevation_extrusion_rotation(west_ext))(west_ext)

elevations = union()(n_ext + s_ext + e_ext + w_ext)
inters = color([0,0,1,0.5])(intersection()(model, elevations))
subt = color([1,0,0])(difference()(model, elevations))

scad_render_to_file(mirror([0,1,0])(subt), \
                     "./mesh/out.scad", include_orig_code=False)

# files = listdir("./mesh/")
# for f in files:
#     if f.find(".scad") >= 0:            # get all .scad files in directory
#         of = f.replace('.scad', '.stl') # name of the outfile .stl
#         cmd = 'call (["openscad",  "-o", "{}",  "{}"])'.format(of, f)   #create openscad command
#         exec(cmd)