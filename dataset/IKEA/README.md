Disclaimer: This is a processed version of the [IKEA dataset](http://ikea.csail.mit.edu/) that I've added here for easy reproduction of our results. Please also cite the original authors of IKEA dataset if you use this data.

1. `Lists/Images.txt` is the list of 937 test images we used, which are in `Images/` directory. This have been cropped from the provided IKEA images.
2. `Lists/ModelID.txt` is the corresponding 3D model from the IKEA dataset that is labeled as the corresponding model for that image. The ID indexes into the file `Lists/Model.txt`.
3. `Voxels/` contains the voxel representation for each of the 225 from the `Lists/ModelID.txt` list. These have been voxelized from the provided CAD models.

The evaluation is done using Average Precision (AP) metric. 
I used a [python function](http://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html),
which takes as input the output of the network (real valued), and the
ground truth (binary), and returns the AP score, by automatically choosing
the threshold points required to construct the P-R curve.
As opposed to a more traditional IoU metric, this allows us to avoid hand-coding thresholds for getting the final output voxel representation.
