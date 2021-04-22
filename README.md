## Code used for the manuscript Physics-Informed Deep Learning Characterizes Morphodynamics of Asian Soybean Rust Disease
Please do get in touch if needed (henry.cavanagh14@imperial.ac.uk)


Please note:
- All figures can be reproduced by running the scripts within morphodynamics/scripts/
- Plots generated will be saved to morphodynamics/outputs/ (where the expected outputs can also be found in folders named via paper sections, as below)
- In the following, replace [compound] with the relevant compound out of [DMSO, compound_A, compound_B, compound_C_0_041, compound_C_10, compound_X]
- Neural networks were trained with a Quadro RTX 6000 GPU. The scripts below will run on a CPU, mostly within < 30s, however for some a GPU is required for reasonable completion times (training networks from scratch & tip model inference).
- Example image data (75 images for each time point and compound) is in morphodynamics/scripts/images/
- To run on the full datasets (available from r.endres@imperial.ac.uk), simply change the load path in morphodynamics/morphospace/dataset.py




# Setup

- Tested on: Mac OS Catalina 10.15.3 & CentOS Linux release 7.7.1908
- Typical install time: <10s

- To run the setup so imports of internal modules work:    
*python setup.py develop*

- Requirements (can be installed via e.g. *python -m pip install torch==1.6.0* once the desired python environment has been loaded)   
python 3.7.3      
torch 1.6.0    
torchvision 0.2.1    
tensorflow 2.3.1  
numpy 1.20.2  
opencv-python 4.4.0.46  
pyabc 0.10.3  
scipy 1.6.2  
matplotlib 3.4.1  
mayavi 4.7.2  






# Section 1: Morphospace

- Fig.2a&b:   
*python run_morphospace.py load*    
Note: to train autoencoder from scratch, replace 'load' with 'train' and new weights will be saved to morphodynamics/outputs/. Please note however that only a subset of images are in the /data/images/ folder



# Section 2: Landscape Model


- Fig.3d&S1a:  
*python run_landscape_visualizations.py [compound] landscape*   
Note: can be viewed in interactive mode by uncommenting line 191, 'mlab.show()'; please note a window will appear for ~15s as the high resolution output is rendered, though this can be adjusted at the mlab.savefig line

- Fig.S2a-f:  
*python run_landscape_visualizations.py [compound] errors*

- Fig. S1b&c:  
*python MSDs.py*   
Note: please note: fewer trajectories are used in this code than for the paper figure for memory considerations, though the results are near-identical

- To train the PINN from scratch:  
*python run_landscape_model.py [compound] train [number of hours to train for] [number of times to save weights during training]*   
Note: new weights will be saved to morphodynamics/outputs/.

- To run inference to get the landscape in array form and do eq. 1 simulations over this from scratch:    
*python run_landscape_model.py [compound] load 0 0*




# Section 3: Tip Growth Model

Note: options for model index (idx_model) are [0, 1, 2]; in the manuscript these are called models 1, 2 & 3 respectively.

- Fig.4b&d Fig.S4a (comparison of MAP simulations with data for lengthening model & probability distributions associated with the MAP values):  
*python L_ABC.py [compound] MAP_simulations*    
Note: to run the full inference process: *python L_ABC.py [compound] full_inference*

- Fig.4c, Fig. S3d & Fig.S4b (comparison of MAP simulations with data for bending models):  
*python theta_ABC.py MAP_vis [compound] [idx_model]*    
Note: inference was run with all three models for compound_A, and model 2 only for all other compounds. To run full inference: *python theta_ABC.py full_inference [compound] 2*. This prints parameters and weights which can then be swapped in to morphodynamics/tip_model/theta/accepted_params_kappa.py for plotting

- Fig.4e (Posterior distribution for the two-parameter optimal bending model):  
*python theta_ABC.py M2_posterior [compound] 2*

- Fig.S3b (comparison of global theta dynamics for all three models, using MAP values):  
*python theta_plot_compare_models.py [idx_model]*

- Fig.S3c (model probabilities):  
*python theta_ABC.py model_probabilities compound_A -1*   
Note: the -1 means all models are being used. o run full model selection: *python theta_ABC.py full_inference compound_A -1*
