# IJCAI-TULVAE
## Desccription of TULVAE
source_code for TULVAE (implement of TULVAE model).<br>
--Learning Hierarchical Structures and Latent Representation
of Human Mobility for Trajectory-User Linking
## Environment
Tensorflow 1.0 or++<br> 
python 2.7<br>
numpy
## Trajectory Splitting and Embbeding
We have show the details in our paper. For embbedding, you can choose the wordv2vec toolkit.
## Usage
To run TULVAE, run the following commands:<br>
'python TULVAE_1122.py'<br>
In each data, for example (Gowalla):<br>
0 480992 49904 420315 73407  (every line in our dataset)
<br>The first one is the user id, others are POIs, which are generated in 6 hours.
## Dataset
We use four different Location-based Social Network data as follows. 
* Gowalla: http://snap.stanford.edu/data/loc-gowalla.html
* Brightkite: http://snap.stanford.edu/data/loc-brightkite.html
* Foursquare(New York): https://sites.google.com/site/yangdingqi/home/foursquare-dataset
* (remark) Please do not use these datasets for commercial purpose. For academic uses, please cite the paper.
## Reference
If the code help to you, please cite the reference...
<br>Any comments and feedback are appreciated.
