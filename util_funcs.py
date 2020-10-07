# -*- coding: utf-8 -*-
"""
Created on Wed Sep 30 11:04:41 2020

@author: Dillo
"""

import cv2
import numpy as np

# =============================================================================
# The dimensions of output images after processing by CNN.
# =============================================================================
h = 192
w = 256





# =============================================================================
# Use openCV to open image. Rescale to fit dimensions of CNN outputs.
# =============================================================================
def load_image(img_src, rescale = False):
    
    img = cv2.imread(img_src)
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #rgb_img = img
    if rescale:
        rgb_img = cv2.resize(rgb_img, (w, h))
        
    return rgb_img
    




# =============================================================================
# Passes image file (.jpg, etc.) through CNN to estimate depth. Saves to disk
#   (at 'out_loc') a numpy array of size (1, 128, 160, 1) with each entry being 
#   the estimated depth of that pixel in the image.
# =============================================================================
def build_depth(img_src, out_loc, ckpt_file_loc):
    
    from Depth_Model.predict import predict as depth
    
    depth(ckpt_file_loc,
          img_src,
          out_loc,
         )
    
    return None





# =============================================================================
#  Passes image file (.jpg, etc.) through CNN to estimate semantic labelling
#   of pixels. Saves to disk (at 'out_loc') a numpy array of size (h,w) with 
#   each entry being the estimated object label of that pixel in the image.
# =============================================================================
def build_semantics(img_src, out_loc):
    
    from PlanarReconstruction.predict import predict as semantics
    
    semantics(img_src,
              out_loc,
             )
    
    return None





# =============================================================================
# Load numpy array from the outputs of one of the two CNN's, pass one of two 
#   keywords: either 'depth' or 'semantics.' If 'depth,' optionally squeeze and 
#   resize image to (h,w).
# =============================================================================
def load_CNN_results(file_loc, type = 'depth', resize = True, h = 192, w = 256):
    
    type_dict = {'depth': '_depth.npy',
                 'semantics': '_segmentation.npy',
                 }
    
    img_array = np.load( file_loc + type_dict[type] )
    
    if (type == 'depth'):
        img_array = img_array.squeeze()
        if (resize == True):
            img_array = cv2.resize( img_array, (w,h) )
    
    return img_array





# =============================================================================
# Use OpenCV to find edges in the photo along which the semantic class changes.
# =============================================================================
def edges_from_semantics(semantics_array):
    
    seg = semantics_array.astype(np.uint8)
    edges = cv2.Canny(seg, 0, 0)

    return edges




# =============================================================================
# Convert edges image to coordinate pairs - determining the coordinate of 
#    each pixel that lies along the edge of a class.
# =============================================================================
def extract_coords(edges_array, strip_size = 5):
    
    # First strip edge from photo to remove edge-artifacts
    edges_stripped = edges_array[strip_size : -strip_size,
                                 strip_size : -strip_size,
                                ]

    # Find index locations where array has value 255 (indicating presence of
    #   an edge). 
    bool_array = np.where( edges_stripped == 255 )
    row_vals = bool_array[0]
    col_vals = bool_array[1]
    num_points = col_vals.shape[0]
    
    # Initialize array that will hold (x,y) coordinates of all edge pixels.
    coords = np.zeros([num_points, 2])
    
    for idx in range(num_points):
        point = [col_vals[idx] + strip_size,
                 row_vals[idx] + strip_size,
                ]
        coords[idx, :] = point
        
        
    
    
    return coords





# =============================================================================
# Find the pairwise distance between every coordinate. If input is length 'n', 
#   output will be 'n' by 'n' array. 
# =============================================================================
def coord_distances(coords):
    
    x = coords[:,0]
    y = coords[:,1]
    
    delta_x_squared = np.square( x[:,np.newaxis] - x )
    delta_y_squared = np.square( y[:,np.newaxis] - y )  
    
    distances = np.sqrt( delta_x_squared +  delta_y_squared)
    
    return distances





# =============================================================================
# For any coordinate index 'coord_idx', find the nearest 'num_neighbs' points.
# =============================================================================
def nearest_neighbors(coords, distances, coord_idx, num_neighbs):
    
    idxs = np.array( [idx for idx in range(len(coords)) ])
    distance_to_idx = distances[coord_idx,:]
    
    distances_indexed = np.stack( (distance_to_idx, idxs), 1)
    sorted_distances = distances_indexed[distances_indexed[:,0].argsort()] 

    return sorted_distances[:num_neighbs+1, 1].astype(int)





# =============================================================================
# Given an 'n' by 2 array of coordinate pairs, fits a line.
# =============================================================================
def linfit( coord_array ):
    
    np.seterr(all = 'ignore')
    N = coord_array.shape[0]
    x = coord_array[:,0]
    y = coord_array[:,1]
    
    num = np.dot(x, y) - x.sum()*y.sum()/N
    denom = np.dot(x, x) - ((x.sum())**2)/N
    
    m = num/denom

    b = ( y.sum() - m*x.sum() )/N
    
    error = (y - (m*x + b))/np.sqrt(N)
    
    return m, b, np.dot(error, error)





# =============================================================================
# For every point along an identified edge fit a line to its nearest neighbors. 
#   If that linfit meets a certain error criteria (set by "error_thresh") 
#   AND if the line has a slope beneath some max value ("slope_thresh") 
#   AND if the point is in the lower portion of the image (max y-value set by 
#   "cut_y_val"), then keep that point in play.
# =============================================================================
def keep_linear_only(coords, distances, num_neighbors,
                     error_thresh, slope_thresh, cut_y_val,
                    ):

    linear_regions = []
    
    y_thresh = cut_y_val
    
    for j, coord in enumerate(coords):

        neighbs = nearest_neighbors(coords, distances, j, num_neighbors)
        m, b, e2 = linfit(coords[neighbs])
        
        if ( (e2 < error_thresh) and ( abs(m) < slope_thresh ) 
                                 and (coord[1] > y_thresh) ):
            linear_regions.append(coord)
     
    linear_regions = np.array( linear_regions )
    
            
    return linear_regions







# =============================================================================
# Working under assumption that the floor-wall boundary can be estimated by a 
#   series of linear segments, cluster all known points along this boundary 
#   according to their relative proximity - will assume that any points
#   belonging to the same cluster are co-linear along the boundary.
# =============================================================================
def cluster_lin_segments(linear_coords):
    
    from sklearn.cluster import AffinityPropagation
    clusters = (AffinityPropagation(random_state = 42,
                                    damping = 0.8,
                                   )
                .fit_predict(linear_coords)
               )

    return clusters
    




# =============================================================================
# Each cluster of points along the boundary is fitted with a line then extended
#   in both directions until it either intersects the edge of the image or 
#   another line. This is done iteratively working from left-to-right.
# =============================================================================
def interpolate_clusters(linear_coords, clusters, w = 256, h = 192):

    lin_segs = {}
    
    # For each cluster, identify the coordinates that belong to that cluster,
    #   the minimum x-value of those coords, and the slope/y-int for the coords
    for cluster in np.unique(clusters):
        lin_segs[cluster] = {}
        
        mask = (clusters == cluster)
        lin_segs[cluster]['coords'] = linear_coords[mask,:]
        lin_segs[cluster]['min_x'] = linear_coords[mask,0].min()
        
        m, b, e2 = linfit(linear_coords[mask,:])
        lin_segs[cluster]['slope'] = m
        lin_segs[cluster]['y_int'] = b
    
    
    # Sort clusters on their min x-vals, orienting left-to-right
    min_x_vals = np.array([ (cluster, lin_segs[cluster]['min_x']) 
                            for cluster in lin_segs
                           ])
    min_x_vals_sorted = min_x_vals[min_x_vals[:,1].argsort()]
    
    
    # Work left-to-right. Extend line k to the right and line k+1 to the left
    #   until they intersect, use line k  as wall-floor boundary up to this 
    #   intersection, then repeat for lines k+1 and k+2, etc. 
    joined_coords = []
    x_left = 0
    for k, cluster in enumerate(min_x_vals_sorted[:,0]):
        
        if not k == len(lin_segs)-1:
            b_left  = lin_segs[min_x_vals_sorted[ k,0 ]]['y_int']
            b_right = lin_segs[min_x_vals_sorted[k+1,0]]['y_int']
            m_left  = lin_segs[min_x_vals_sorted[ k,0 ]]['slope']
            m_right = lin_segs[min_x_vals_sorted[k+1,0]]['slope']       
            
            x_intersect = (b_right - b_left)/(m_left - m_right)
        else:
            b_left  = lin_segs[min_x_vals_sorted[ k,0 ]]['y_int']
            m_left  = lin_segs[min_x_vals_sorted[ k,0 ]]['slope']       
            
            x_intersect = w
        
      
        
        # Deal with special case of lines that have intersections too-far left.
        if x_intersect <= x_left:
            x_intersect = x_left+1
        
        x_vals = np.arange(x_left, x_intersect, 0.25)
        
        
        # Append to list of coordinates all points on line k
        joined_coords += [ ( x, m_left*x + b_left ) for x in x_vals 
                           if (m_left*x+b_left < h) and (m_left*x+b_left > 0)
                         ]
        
        
        # Set starting point for next iteration
        x_left = x_intersect
        
    return np.array(joined_coords)
    




# =============================================================================
# Given a set of wall-floor boundary coordinates and an array containing the
#   room's depth informatuon, map the depth values on to the boundary. To get a
#   value of depth at any one point along the boundary average all points 
#   within a radius of 'avg_rad'. 
# =============================================================================
def map_depth_to_boundary(boundary_coords, depth_array, 
                          w = 256, h = 192, avg_rad = 10,
                         ):
    
    num_coords = boundary_coords.shape[0]
    floor_depth = np.zeros( [num_coords, 1] )
    
    for k in range( num_coords ) :
    
        x = round( boundary_coords[k,0] )
        y = round( boundary_coords[k,1] )
        
        # Deal with points that fall near the image edge
        if x == 0:
            x = 1
        elif x == w:
            x = w-1
        
        if y == 0:
            y = 1
        elif y == h:
            y = h-1
            
        rad_x = min(x, w-x, avg_rad)
        rad_y = min(y, h-y, avg_rad)
        
        floor_depth[k] = np.mean( depth_array[y-rad_y : y+rad_y, 
                                              x-rad_x : x+rad_x,
                                             ])
    return floor_depth
    




# =============================================================================
# Finite difference to find derivative
# =============================================================================
def diff(x,y):
    
    n = len(x)
    h = x[1]-x[0]
    dydx = np.zeros(n)
    
    for k in range(n):
        if k == 0:
            dydx[k] = ( y[k+1] - y[k] )/h
        if k == n-1:
            dydx[k] = ( y[k] - y[k-1] )/h
        else:
            dydx[k] = ( y[k+1] - y[k-1] )/(2*h)
            
    return dydx





# =============================================================================
# Fit the room boundary to a rectangle, output these corners to perform 
#   change in perspective. If the room is not at least mostly rectangular
#   this will do very poorly (check for angle between output lines)
# =============================================================================
def get_persp_corners(boundary_coords, im_w, im_h, offs = 10, num_clus=10, 
                      w = 256, h = 192):
    
    w_scale = im_w/w
    h_scale = im_h/h
    
    from sklearn.cluster import KMeans
    
    # Calculate derivative at every point and cluster by slope
    x = boundary_coords[:,0]
    y = boundary_coords[:,1]
    dydx = diff(x,y)
    
    clusters = (KMeans(n_clusters = num_clus,
                       random_state = 42,
                       )
                    .fit_predict( dydx.reshape(-1,1) )
                   )
    
    
    # For each cluster, compute the average x and y values in the cluster,
    #   if there are more than 10 boundary points that lie in that cluster.
    avg_x = np.zeros(num_clus)
    avg_y = np.zeros(num_clus)
    for clus in range(num_clus):
        pts = boundary_coords[ np.where(clusters == clus) ]
        if pts.shape[0] > 10:
            avg_x[clus] = pts[:,0].mean() 
            avg_y[clus] = pts[:,1].mean() 
        else:
            avg_x[clus] = np.nan
            avg_y[clus] = np.nan
    
    
    # Find the cluster indices that correspond to the lowest y average, as wll
    #   as highest/lowest x values (right and left, respectively).
    y_clus = np.nanargmin(avg_y)
    x_l_clus = np.nanargmin(avg_x)
    x_r_clus = np.nanargmax(avg_x)
    
    
    # Grab all points that lie in the afore-mentioned 3 clusters, fit each
    #   set of points to a line.
    neighbs_t = np.where( clusters == y_clus )
    neighbs_l = np.where( clusters == x_l_clus )
    neighbs_r = np.where( clusters == x_r_clus )
    
    m_l, b_l, e = linfit( boundary_coords[ neighbs_l ] )
    m_r, b_r, e = linfit( boundary_coords[ neighbs_r ] )
    m_t, b_t, e = linfit( boundary_coords[ neighbs_t ] )
    
    b_l, b_r, b_t = [b - offs for b in[b_l, b_r, b_t]]
    

    # Calculate boundaries of each segment: {}_l and {}_r bounds correspond to
    #   the point where the lines intersect the bottom of the image on the left
    #   and right, respectively. {}_ml and {}_mr refer to the points where the
    #   left and right lines intersect the middle line. 
    x_l_bound  = (h - b_l)/m_l
    x_r_bound  = (h - b_r)/m_r
    x_ml_bound = (b_t - b_l)/(m_l - m_t)
    x_mr_bound = (b_r - b_t)/(m_t - m_r)
    
    y_l_bound = x_l_bound*m_l + b_l
    y_r_bound = x_r_bound*m_r + b_r
    y_ml_bound = x_ml_bound*m_l + b_l
    y_mr_bound = x_mr_bound*m_r + b_r
    
    
    # Pack the points up as coordinate pairs for the four corner of the 
    #   rectangle that best captures the room boundary.
    tl = (x_ml_bound*w_scale, y_ml_bound*h_scale)
    tr = (x_mr_bound*w_scale, y_mr_bound*h_scale)
    br = (x_r_bound*w_scale, y_r_bound*h_scale)
    bl = (x_l_bound*w_scale, y_l_bound*h_scale)
    
    return np.array( [tl, tr, br, bl], np.float32 )





# =============================================================================
# Output for corners indicating destination image size after transform
# =============================================================================
def target_corners(h = 192, w = 256):
    
    tl = [0, 0]
    tr = [w, 0]
    br = [w, h]
    bl = [0, h]
    
    return np.array( [tl, tr, br, bl], np.float32 )





# =============================================================================
# 
# =============================================================================
def perspective_shift(img_to_warp, src_corners, trgt_corners):
   
    import cv2
    
    w = trgt_corners[1,0] - trgt_corners[0,0]
    h = trgt_corners[2,1] - trgt_corners[1,1]
    out_shape = (w, h)
    
    matrix = cv2.getPerspectiveTransform(src_corners, trgt_corners)
    result = cv2.warpPerspective(img_to_warp, matrix, out_shape)
    
    return result





# =============================================================================
# 
# =============================================================================
def boundary_to_img(boundary_coords, h = 192, w = 256):

    num_coords = len( boundary_coords )
    rough_bound = boundary_coords.astype(int)
    rough_coords = [ ( rough_bound[k,0], rough_bound[k,1] ) 
                     for k in range(num_coords)
                   ]
    
    bound_img = np.zeros( [int(h), int(w)] )
    
    for coord in rough_coords:
        x = int( coord[0] )
        y = int( coord[1] )
        
        if x == 0:
            x = 2
        elif x >= w:
            x = w-3
        
        if y == 0:
            y = 2
        elif y == h:
            y = h-3

        x = int(x)
        y = int(y)
       
        try:
            bound_img[y-2:y+3, x-2:x+3] = 1
        except:
            print('x coord is {}.\n y coord is {}'.format(x,y) )
        #bold_img = make_bound_bold(bound_img)
        
    return bound_img





# =============================================================================
# 
# =============================================================================
def make_bound_bold(bound_img):
    
    idxs = np.where( bound_img == 1 )
    x = idxs[0]
    y = idxs[1]
    

    num_zeros = len(x)
    
    bold_img = np.zeros_like(bound_img)
    
    print(x[:10])
    print(y[:10])
    
    for k in range(num_zeros):
        bold_img[x, y] = 1
        
        
    return bold_img
    
    
    

# =============================================================================
#     
# =============================================================================
def bounding_box(points):

    x_coordinates = points[:,0] 
    y_coordinates = points[:,1]

    minx = round( min(x_coordinates) )
    maxx = round( max(x_coordinates) )
    miny = round( min(y_coordinates) )
    maxy = round( max(y_coordinates) )

    return [minx, miny, maxx, maxy]




# =============================================================================
# 
# =============================================================================
def dominant_class(bbox, semantics ):
    
    minx, miny, maxx, maxy = bbox
    
    floor = semantics[miny:maxy, minx:maxx]

    unique, counts = np.unique(floor, return_counts=True)
    count_dict = dict(zip(unique, counts))

    floor_class = max(count_dict, key=count_dict.get)
    
    
    return floor_class


# =============================================================================
# 
# =============================================================================
def mask_semantics(semantics, floor_class):
    
    masked = (semantics == floor_class).astype(int)
    
    return masked



