# MSAAFNet
# Requirements
python 3.7 + pytorch 1.9.0 + imageio 2.22.2
# Saliency maps
We provide [saliency maps](https://pan.baidu.com/s/1eUWvdXUZpbBFkbDJDa-Utw?pwd=lbc0) (code:lbc0) of MSAAFNet_V2 on ORSSD, EORSSD and ORSI-4199 datasets.  
We also provide [saliency maps](https://pan.baidu.com/s/1krIXhl7KF8_fqjIF7P6yyQ?pwd=lbc0) (code:lbc0) of MSAAFNet_V2 on DUT-O, DUTS-TE, HKU-IS, ECSSD and PASCAL-S datasets.  
We provide [saliency maps](https://pan.baidu.com/s/1Ey04P98O2pOC9O3OXaq4RA) (code:lbc0) of our MSAAFNet_V3 on ORSSD, EORSSD and ORSI-4199 datasets.  
We also provide [saliency maps](https://pan.baidu.com/s/1stY_QeeDg2XQgy0pCyZHiw) (code:lbc0) of MSAAFNet_V3 on DUT-O, DUTS-TE, HKU-IS, ECSSD and PASCAL-S datasets.
# Training
Run train_MSAAFNet.py. We use the Canny.py to generate the edge label for training.
# Pre-trained model and testing
Download the following pre-trained model and put them in ./models/MSAAFNet/, then run test_MSAAFNet.py.  
[MSAAFNet_V2_EORSSD](https://pan.baidu.com/s/1dJDRkUC5Hc6Q28XucuISUw?pwd=lbc0) (code:lbc0)  
[MSAAFNet_V2_ORSSD](https://pan.baidu.com/s/1IXQvmE2DCazyKGcs3TCVQQ?pwd=lbc0) (code:lbc0)  
[MSAAFNet_V2_ORSI-4199](https://pan.baidu.com/s/1wGtUsyGd_J2OMgUwZHbZdw?pwd=lbc0) (code:lbc0)  
[MSAAFNet_V2_DUTS-TR](https://pan.baidu.com/s/1XpBx9gPWCaxM1YnQsgELDw?pwd=lbc0) (code:lbc0)  
[MSAAFNet_V3_EORSSD](https://pan.baidu.com/s/1p8NpIq-9Bgny57W9OUg4nQ) (code:lbc0)  
[MSAAFNet_V3_ORSSD](https://pan.baidu.com/s/18KIMvQl3BwRl_eGN653ehw) (code:lbc0)  
[MSAAFNet_V3_ORSI-4199](https://pan.baidu.com/s/10AE4MpdLsIiH6NhJ38dJwg) (code:lbc0)  
[MSAAFNet_V3_DUTS-TR](https://pan.baidu.com/s/1YIq1E2h15Ik24wYQlVn-VQ) (code:lbc0)
# Evaluation Tool
You can use the [evaluation tool (MATLAB version)](https://github.com/MathLee/MatlabEvaluationTools) to evaluate the above saliency maps.
# Acknowledgement
We would like to thank the contributors to the [MCCNet](https://github.com/MathLee/MCCNet).
