# -*- coding: utf-8 -*-
"""
Created on Tue Oct  6 19:56:25 2020

@author: Dillo
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from main import main
import os




def process_validation_data():
    depth_path = 'F:/Image_Perspective/data/nyu_datasets_changed/target_depths/'
    img_path = 'F:/Image_Perspective/data/nyu_datasets_changed/input/'
    label_path = 'F:/Image_Perspective/data/nyu_datasets_changed/labels_38/'
    
    #im_num = 910
    min_im_num = 294
    max_im_num = 300
    
    for im_num in range(min_im_num, max_im_num):
        
        im_name = (5 - len(str(im_num)))*'0' +  str(im_num) 
        
        depth = cv2.imread(depth_path + im_name + '.png')
        img = cv2.imread(img_path + im_name + '.jpg')
        labels = cv2.imread(label_path + im_name + '.png')[:,:,0]
        
        floor_label = 2
        floor = (labels == floor_label).astype(int)
        
        
        num_floor_pixels = floor.sum()
        tot_pixels = labels.shape[0]*labels.shape[1]
        pct_floor = num_floor_pixels/tot_pixels*100
        
        
        save_filepath = 'F:/Insight/RoomPerspective/Validation/'
        
        
        floor_thresh = 20
        
        
        if pct_floor > floor_thresh:
              
            
            save_labeled = save_filepath + 'labeled_images/' + im_name + '.npy'
            save_processed = save_filepath + 'processed/' + im_name + '/'
            save_img = save_filepath + 'rgb_images/' + im_name + '.npy'
            save_target_depth = save_filepath + 'target_depths/' + im_name + '.npy'
            save_target_labels = save_filepath + 'target_labels/' + im_name + '.npy'
        
        
            try:
                main(img_path + im_name + '.jpg', save_processed)
            except:
                None
            np.save(save_labeled, labels )
            np.save(save_img, img )
            np.save(save_target_depth, depth )
            np.save(save_target_labels, floor )
            
        
            fig, ( (ax1, ax2), (ax3, ax4) ) = plt.subplots( 2,2)
        
            ax1.imshow(img)
            ax1.axis('off')
            ax1.set_title('original')
            
            ax2.imshow(labels)
            ax2.axis('off')
            ax2.set_title('labeled')
            
            ax3.imshow(depth)
            ax3.axis('off')
            ax3.set_title('depth')
            
            ax4.imshow(floor)
            ax4.axis('off')
            ax4.set_title('floor')
            
            fig.suptitle( 'Image ' + im_name + '\n({:.0f}% floor)'.format(pct_floor) )
            
            
            plt.savefig(save_processed + '_targets.jpg', dpi = 199)

    return None





def get_depth_error(im_name, norm = 41):
    
    m_to_ft = 3.28084
    
    save_filepath = 'F:/Insight/RoomPerspective/Validation/'    
    target_depth_path = save_filepath + 'target_depths/' + im_name + '.npy'
    processed_path = save_filepath + 'processed/' + im_name + '/_depth.npy'
    
    
    target = np.load( target_depth_path )[:,:,0]/norm
    estimate = np.load( processed_path )[0,:,:,0]
    
        
    target_size = (target.shape[1], target.shape[0])
    estimate_size =  (estimate.shape[1],estimate.shape[0])
    
    estimate = cv2.resize(estimate, target_size)
    #target = cv2.resize(target, estimate_size)
    
    
    error = (target - estimate)*m_to_ft
    T = target.shape[0]*target.shape[1]
    
    rmse = np.sqrt( np.multiply(error, error).sum()/T )
    
    target_cent = target[50:-50, 50:-50]
    estim_cent = estimate[50:-50, 50:-50]
    T2 = target_cent.shape[0]*target_cent.shape[1]
    
    abs_rel_error = np.divide( np.abs( target_cent - estim_cent), target_cent).sum()/T2
    
    med_abs_error = np.median(error)
    
    fig, (ax1, ax2) = plt.subplots(1,2)
    
    ax1.imshow( (target - target.min())/( target.max() - target.min() )   )
    ax1.axis('off')
    
    ax2.imshow( (estimate - estimate.min())/( estimate.max() - estimate.min() )  )
    ax2.axis('off')

    
    plt.savefig(save_filepath + im_name + '_depth_diff.jpg', dpi = 199)
    
    return rmse, abs_rel_error, med_abs_error


rmse, abs_rel_error, med_abs_error = get_depth_error( '00292' )





im_name = '00292'

save_filepath = 'F:/Insight/RoomPerspective/Validation/'
target_label_path = save_filepath + 'target_labels/' + im_name + '.npy'


target_labels = np.load(target_label_path)



fig, ax = plt.subplots(1,1)

ax.imshow(target_labels, cmap = 'binary')
ax.axis('off')
plt.savefig(save_filepath + im_name + '_target_label.jpg', dpi = 199 )
