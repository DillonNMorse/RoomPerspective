# SpaceAceSpaceAce is a webapp that allows a user to get a feel for room sizes at a glance. When provided an image of a (mostly empty) room, trace the outline of the floor and map depth (distance from camera in the direction perpendicular to the plane of the lens) on to this outline. For some nearly-rectangular rooms, determine appropriate horizon lines to apply transformation on to this outline in order to estimate a top-down map of the room.## InstallationRefer to packages.txt for a list of required packages with some required package versions.Depth_Model/ contains the CNN used for determining depth, see the [GitHub repo](https://github.com/iro-cp/FCRN-DepthPrediction) based on [this paper](https://arxiv.org/abs/1606.00373). PlanarReconstruction/ contains the CNN used for surface segmentation within images, see the [GitHub repo](https://github.com/svip-lab/PlanarReconstruction) based on [this paper](https://arxiv.org/pdf/1902.09777.pdf).To complete the installation:* Create the folder PlanarReconstruction/ckpt/ and add the corresponding file pretrained.pt from the PlanarReconstruction GitHub repo.*  Inside the folder Depth_Model/ add the weight pre-trained from NYU Depth v2, links can be found in the DepthPrediction GitHub repo ReadMe under the heading Models.## Usage* To run the web-app locally execute the terminal command:`streamlit run webapp.py`* To execute the app from main.py, alter the url variable at the top of the file before execution. A number of new folders will be created within the repository on your local machine, containing a variety of output images.