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
    min_im_num = 295
    max_im_num = 1448
    
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
        
        
        save_filepath = 'F:/Insight/SpaceAce/Validation/'
        
        
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
    
    # =============================================================================
    #   Load, normalize, and resize
    # =============================================================================
    
    save_filepath = 'F:/Insight/SpaceAce/Validation/'    
    target_depth_path = save_filepath + 'target_depths/' + im_name + '.npy'
    processed_path = save_filepath + 'processed/' + im_name + '/_depth.npy'
    
    
    target = np.load( target_depth_path )[:,:,0]/norm
    estimate = np.load( processed_path )[0,:,:,0]  - 1.7
    
        
    target_size = (target.shape[1], target.shape[0])
    #estimate_size =  (estimate.shape[1],estimate.shape[0])
    
    estimate = cv2.resize(estimate, target_size)
    #target = cv2.resize(target, estimate_size)
    
    # =============================================================================
    #     Compute error, stripping off edges of photos
    # =============================================================================
    
    target_cent = target[40:-40, 40:-40]
    estim_cent = estimate[40:-40, 40:-40]
    
    error = (target_cent - estim_cent)*m_to_ft
    T = target_cent.shape[0]*target_cent.shape[1] # num pixels in target
    
    rmse = np.sqrt( np.multiply(error, error).sum()/T )
    abs_rel_error = np.divide( np.abs( error ), target_cent).sum()/T
    med_abs_error = np.median( abs(error) )
    
    depth_error_corr = np.corrcoef(error.flatten(), target_cent.flatten()  )
    # =============================================================================
    #     Make sure both have same min/max values for correct scaling.
    # =============================================================================
    
    maxes = [ target.max(), estimate.max() ]
    mins  = [ target.min(), estimate.min() ]
    
    target[0,0] = max( maxes )
    estimate[0,0] = max( maxes )
    
    target[0,1] = min( mins )
    estimate[0,2] = max( mins )
    
    # =============================================================================
    #   Plot target and estimate on same scale
    # =============================================================================
    
    fig, (ax1, ax2) = plt.subplots(2,1)
    
    ax1.imshow(target*m_to_ft)
    im = ax2.imshow(estimate*m_to_ft)
    
    fig.subplots_adjust(right=0.8)
    #cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    fig.colorbar(im, ax = [ax1,ax2] )
    
    ax1.axis('off')
    ax2.axis('off')
    plt.savefig(save_filepath + im_name + '_depth_both.jpg', dpi = 199, transparent = True)
    plt.show()
    
    
    
    # =============================================================================
    #   Plot target - estimate  
    # =============================================================================
    error_fig, error_ax = plt.subplots(1,1)
    
    im = error_ax.imshow( target_cent - estim_cent )
    error_fig.colorbar(im)
    error_ax.axis('off')
    plt.savefig(save_filepath + im_name + '_depth_diff.jpg', dpi = 199)
    
    
    
    
    # =============================================================================
    #     Plot error vs. depth
    # =============================================================================
    num_pix = len( target_cent.flatten() )
    
    fig2, ax2 = plt.subplots(1,1)
    ax2.scatter( target_cent.flatten(), ( error.flatten() ), s=1)
    ax2.plot(target_cent.flatten(), num_pix*[error.mean()], c= 'red', lw = 2 )
    ax2.plot(target_cent.flatten(), num_pix*[error.mean() + error.std()], c= 'red', ls = '--', lw = 1 )
    ax2.plot(target_cent.flatten(), num_pix*[error.mean() - error.std()], c= 'red', ls = '--', lw = 1 )
    #plt.yscale('log')
    ax2.set_xlabel('Depth (ft)')
    ax2.set_ylabel('Error (ft)')
    plt.savefig(save_filepath + im_name + '_error_vs_depth.jpg', dpi = 199 )
    
    return rmse, abs_rel_error, med_abs_error, depth_error_corr, target, estimate



def resize_labels(target, estimate):
    
    target_size = (target.shape[1], target.shape[0])
    estimate_size =  (estimate.shape[1],estimate.shape[0])
    
    h_scale = target_size[0]/estimate_size[0]
    w_scale = target_size[1]/estimate_size[1]
    
    new_target = np.zeros_like( estimate )
    for x in range( estimate_size[1] ):
        for y in range( estimate_size[0] ):
            est_x = round( x*w_scale )
            est_y =round( y*h_scale )
            
            if est_x >= estimate_size[1]:
                est_x -= 1
            if est_y >= estimate_size[0]:
                est_y -= 1            
            
            new_target[x,y] = target[est_x, est_y]
    
    return new_target




def get_segmentation_error(im_number):
    
    
    save_filepath = 'F:/Insight/SpaceAce/Validation/'    
    target_label_path = save_filepath + 'target_labels/' + im_name + '.npy'
    processed_path = save_filepath + 'processed/' + im_name + '/_segmentation.npy'

    target = np.load( target_label_path )
    estimate = np.load( processed_path )
    
    label = estimate[175, 100]
    
    
    target_size = (target.shape[1], target.shape[0])
    estimate_size =  (estimate.shape[1],estimate.shape[0])
    
    #estimate = cv2.resize( estimate.astype('uint8')  , target_size)
    #target = cv2.resize(target, estimate_size)
    
    new_target = resize_labels(target, estimate)
    
    estimate_masked = (estimate == label).astype(int)

    fig, (ax1, ax2) = plt.subplots(1,2)
    
    ax1.imshow(new_target, cmap = 'Greys')
    #ax1.axis('off')
    ax2.imshow(estimate_masked, cmap = 'Greys' )
    #ax2.axis('off')

    plt.savefig(save_filepath + im_name + '_labels.jpg', dpi = 199)
    
    from matplotlib.colors import LinearSegmentedColormap
    cmap_pred = LinearSegmentedColormap.from_list('mycmap', ['white', 'red'])
    cmap_targ = LinearSegmentedColormap.from_list('mycmap', ['white', 'blue'])
    fig2, ax2 = plt.subplots(1,1)
    ax2.axis('off')
    ax2.imshow(new_target, cmap = cmap_targ, alpha = 0.4, label = 'Ground Truth'  )
    ax2.imshow(estimate_masked, cmap = cmap_pred, alpha = 0.4, label = 'Predicted'  )
    plt.savefig(save_filepath + im_name + '_overlayed_labels.jpg', dpi = 199,bbox_inches='tight')
    
    
    
    from sklearn.metrics import confusion_matrix
    
    cm = confusion_matrix( new_target.ravel(), estimate_masked.ravel() )

    return target, estimate, cm



im_name = '00424'

rmse, abs_rel_error, med_abs_error, depth_error_corr, target, estimate = get_depth_error( im_name )

label_target, label_estimate, cm = get_segmentation_error(im_name)


save_filepath = 'F:/Insight/SpaceAce/Validation/'
target_label_path = save_filepath + 'target_labels/' + im_name + '.npy'


target_labels = np.load(target_label_path)



fig, ax = plt.subplots(1,1)

ax.imshow(target_labels, cmap = 'binary')
ax.axis('off')
plt.savefig(save_filepath + im_name + '_target_label.jpg', dpi = 199 )









# =============================================================================
# Display actual photo
# =============================================================================
img = cv2.imread(  'F:/Image_Perspective/data/nyu_datasets_changed/input/' + im_name + '.jpg'  )
im_fig, im_ax = plt.subplots(1,1)
im_ax.imshow(img)
im_ax.axis('off')
plt.savefig(save_filepath + im_name + '_original.jpg', dpi = 199 )