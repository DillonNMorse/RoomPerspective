# -*- coding: utf-8 -*-
"""
Created on Wed Sep 30 11:06:02 2020

@author: Dillo
"""

import os
import util_funcs as f

import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import cv2
from urllib.request import urlopen

CNN_h = 192
CNN_w = 256

# =============================================================================
# im_path = './Images/1594AsterCt/'
# im_name = '924529_18_0.jpg'
# =============================================================================

# =============================================================================
# im_path = './Images/8100EUnionAve/'
# im_name = '03.jpg' #'09.jpg' 
# =============================================================================

# =============================================================================
# im_path = './Images/' 
# im_name = 'test_square_2.jpg' 
# =============================================================================

# =============================================================================
# im_path = './Images/2855RockCreekCir/' 
# im_name = '8955264_8_0.jpg' 
# =============================================================================

# =============================================================================
# im_path = './Images/Apt/' 
# im_name = '4.jpg' 
# =============================================================================


# =============================================================================
# save_path = './Processed/' + im_name[:-4] 
# 
# save_filepath = save_path + '/' + im_name[:-4] 
# img_filepath = im_path + im_name
# =============================================================================


img_url = 'https://ssl.cdn-redfin.com/photo/102/bigphoto/289/2290289_12_K.jpg'


def main(img_filepath, save_filepath):

    ckpt_file_loc = './Depth_Model/NYU_FCRN.ckpt'


    if not os.path.exists(save_filepath):
        os.mkdir(save_filepath)
    

    # =============================================================================
    # Load image
    # =============================================================================
    img = f.load_image(img_filepath,
                       rescale = False,
                      )
    im_w = img.shape[1]
    im_h = img.shape[0]
    
    w_scale = im_w/CNN_w
    h_scale = im_h/CNN_h
    
    # =============================================================================
    # Only pass through CNN's if there doesn't already exist CNN output info.
    # =============================================================================
    rebuild = True
    depth_exists  = os.path.exists(save_filepath + '_depth.npy')
    semant_exists = os.path.exists(save_filepath + '_segmentation.npy')
    
    if ( depth_exists & semant_exists ):
        rebuild = False
    
    if rebuild == True:
        f.build_depth(img_filepath,
                      save_filepath,
                      ckpt_file_loc,
                     )  
        f.build_semantics(img_filepath,
                          save_filepath,
                         )
    
    # =============================================================================
    # Load depth and semantic data
    # =============================================================================
    depth = f.load_CNN_results(save_filepath,
                               type = 'depth',
                               resize = True,
                              )
    semantics = f.load_CNN_results(save_filepath,
                                   type = 'semantics',
                                  )
    fig, (ax1, ax2) = plt.subplots(1,2)
    ax1.imshow(depth)
    ax2.imshow(semantics)
    plt.savefig(save_filepath + '_Depth_and_Semantics.jpg')
    
    edges = f.edges_from_semantics(semantics)
    fig, ax = plt.subplots(1,1)
    ax.imshow(edges)
    plt.savefig(save_filepath + '_Edges.jpg')
    
    # =============================================================================
    # Extract coordinates along all edges of the different segmentation classes. 
    #   Keep only those points that which are (i) in an approximately-linear
    #   boundary segment, (ii) have a slope less than some threshold value (want 
    #   to drop any vertical lines), and (iii) are below a critical y-value 
    #   (don't  want to capture the ceiling-wall interface).
    # =============================================================================
    edges = f.edges_from_semantics(semantics)
    edge_coords = f.extract_coords(edges, strip_size = 10)
    
    fig, ax = plt.subplots(1,1)
    ax.scatter( edge_coords[:,0], edge_coords[:,1] )
    plt.savefig(save_filepath + '_Edge_Coords_Original.jpg')
    
    distances = f.coord_distances(edge_coords)
    
    lin_segments =  f.keep_linear_only(edge_coords,
                                       distances,
                                       num_neighbors = 21,
                                       error_thresh = 0.8,
                                       slope_thresh = 6,
                                       cut_y_val = 110,
                                      )
    
    fig, ax = plt.subplots(1,1)
    ax.scatter( lin_segments[:,0], lin_segments[:,1] )
    ax.set_xlim([0,256])
    ax.set_ylim([0,192])
    ax.invert_yaxis()
    plt.savefig(save_filepath + '_Linear_Segments.jpg')
    
    # =============================================================================
    # Cluster remaining linear segments by relative proximity, fit each cluster
    #   to a line, and extrapolate those lines until they intersect. Finallly,
    #   map the depth data on to the newly-created boundary lines.
    # =============================================================================
    clusters = f.cluster_lin_segments(lin_segments)
    boundary = f.interpolate_clusters(lin_segments, clusters)
    depth_boundary = f.map_depth_to_boundary(boundary, depth)
    
    fig, ax = plt.subplots(1,1)
    ax.scatter(boundary[:,0], boundary[:,1] )
    ax.set_xlim([0,256])
    ax.set_ylim([0,192])
    ax.invert_yaxis()
    plt.savefig(save_filepath + '_Boundary.jpg')
    
    # =============================================================================
    # Make plot
    # =============================================================================
    m_to_ft = 3.28084
    fig, ax = plt.subplots(1,1)
    depth_plot = ax.scatter( boundary[:,0]*(w_scale),
                             boundary[:,1]*(h_scale),
                             c = depth_boundary*m_to_ft,
                             cmap = 'YlGnBu_r',
                             alpha = 0.5
                           )
    ax.imshow(img)
    
    plt.colorbar(depth_plot)
    ax.axis('tight')
    ax.axis('off')
    
    ax.set_title('Floor-Wall Boundary\n(depth measured in feet)')
    
    plt.savefig(save_filepath + '_depth.jpg')
    
    
    
    
    # =============================================================================
    # Find perspective transformation for approx-rectangular rooms.
    # =============================================================================
    
    
    
    corners = f.get_persp_corners(boundary, im_w, im_h)
    
    tl = corners[0,:]
    tr = corners[1,:]
    br = corners[2,:]
    bl = corners[3,:]
    
    
    top_x = [tl[0], tr[0]]
    left_x = [bl[0], tl[0]]
    right_x = [br[0], tr[0]]
    
    top_y = [tl[1], tr[1]]
    left_y = [bl[1], tl[1]]
    right_y = [br[1], tr[1]]
    
    
    fig3, ax3 = plt.subplots(1,1)
    
    ax3.plot(top_x, top_y )
    ax3.plot(left_x, left_y )
    ax3.plot(right_x, right_y )
    
    plt.imshow(img)
    plt.savefig(save_filepath + '_persp_outline.jpg')

    
    dst_pts = f.target_corners(h = 19*50, w = 12*50)
    
    
    result1 = f.perspective_shift(img, corners, dst_pts) 
    fig4, (ax4) = plt.subplots(1,1)
    fig.tight_layout()
    ax4.imshow(result1)
    
    
    
    boundary_scaled = np.array([(boundary[k,0]*w_scale, boundary[k,1]*h_scale) for k in range(  boundary.shape[0] )] )
    
    boundary_img = f.boundary_to_img(boundary_scaled, h = 192*h_scale, w = 256*w_scale)
    
    result2 = f.perspective_shift(boundary_img, corners, dst_pts) 
    
    
    fig, ax = plt.subplots(1,1)
    ax.imshow(result2, cmap = 'gray_r')
    #plt.axis('off')
    plt.savefig(save_filepath + '_map.jpg')



    return (depth_boundary.max() - 0*depth_boundary.min())*m_to_ft


if __name__ == '__main__':

    resp = urlopen(img_url)
    image = np.asarray(bytearray(resp.read()), dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    
    
    # =============================================================================
    # Create a directory to store image data in
    # =============================================================================
    def create_filepath(im_name, append = '', type = 'folder' ):
        
        save_path = 'F:/RoomRotate/Processed/' 
        save_filepath = save_path + im_name + '/'
        
        if type == 'file':
            save_filepath = save_filepath  + '_{}.jpg'.format( append )
        
        return save_filepath
    
    
    
    im_name = img_url.split('/')[-1].split('.')[0]
    
    if not os.path.exists(  create_filepath( im_name )  ):
        os.mkdir(  create_filepath( im_name )  )
    
    cv2.imwrite( create_filepath(im_name, 'original', 'file'), image)
    
    
    img_filepath = create_filepath(im_name, 'original', 'file')
    save_filepath = save_filepath = create_filepath(im_name)
    
    
    main(img_filepath, save_filepath)
