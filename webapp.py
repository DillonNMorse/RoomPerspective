# -*- coding: utf-8 -*-
"""
Created on Thu Sep 24 23:00:37 2020

@author: Dillo
"""

import streamlit as st
import random
import time
import cv2
import matplotlib.pyplot as plt
from urllib.request import urlopen
import numpy as np
import os

from main import main 



# Add a title
st.title('SpaceAce')



# =============================================================================
# Load an image from url
# =============================================================================
img_url = st.text_input('Enter the url of an image you\'d like to analyze.',
                        'https://ssl.cdn-redfin.com/system_files/media/463635_JPG/item_14.jpg'
                        )

resp = urlopen(img_url)
image = np.asarray(bytearray(resp.read()), dtype="uint8")
image = cv2.imdecode(image, cv2.IMREAD_COLOR)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)



# =============================================================================
# Create a directory to store image data in
# =============================================================================
def create_filepath(im_name, append = '', type = 'folder' ):
    
    save_path = './Processed/' 
    save_filepath = save_path + im_name + '/'
    
    if type == 'file':
        save_filepath = save_filepath  + '_{}.jpg'.format( append )
    
    return save_filepath



im_name = img_url.split('/')[-1].split('.')[0]

if not os.path.exists(  create_filepath( im_name )  ):
    os.mkdir(  create_filepath( im_name )  )

cv2.imwrite( create_filepath(im_name, 'original', 'file'), image)


# =============================================================================
# Resize then display image
# =============================================================================
h, w = (image.shape[0], image.shape[1])
new_height = 400
new_width = round( new_height/h*w )
image = cv2.resize( image, (new_width,new_height))

st.image(image)


# =============================================================================
# img_filepath = im_path + im_name
# save_filepath = save_path + '/' + im_name[:-4] 
# 
# 
# d2 = cv2.imread(img_filepath)
# 
# h, w = (d2.shape[0], d2.shape[1])
# new_height = 400
# new_width = round( new_height/h*w )
# d2 = cv2.resize( d2, (new_width,new_height))
# 
# st.image(d2)
# =============================================================================



# =============================================================================
# Process image
# =============================================================================
try:
    d = main( img_filepath  = create_filepath(im_name, 'original', type = 'file'  ),
              save_filepath = create_filepath(im_name),
             )
except:
    'Sorry, an unexpected error occured.'
    overlay_image = cv2.imread( create_filepath(im_name, 'depth_overlay', type = 'file') )
    overlay_image = cv2.resize( overlay_image, (round(new_width*1.2),round(new_height*1.2)))
    st.image(overlay_image)
# =============================================================================
# Load processed image
# =============================================================================
image_depth = cv2.imread( create_filepath(im_name, 'depthimage2', type = 'file') )
#image_depth = cv2.resize( image_depth, (new_width,new_height))
#overlay_image = cv2.imread( create_filepath(im_name, 'depth_overlay', type = 'file') )
#overlay_image = cv2.resize( overlay_image, (round(new_width*1.2),round(new_height*1.2)))



# =============================================================================
# Display the images on the website
# =============================================================================

st.image(image_depth)
#st.image(overlay_image)


'Top-down map view of room\n(works best on nearly-rectangular rooms).'

d4 = cv2.imread(  create_filepath(im_name, 'map2', type = 'file')  )
d4 = cv2.resize( d4, (new_width,new_height))
st.image(d4)

















