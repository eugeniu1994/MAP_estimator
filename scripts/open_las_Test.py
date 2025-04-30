
import numpy as np
import laspy

input_path = "/home/eugeniu/vux-georeferenced/merged/segment_0.las"

pcloud = laspy.read(input_path)
print('pcloud:', np.shape(pcloud.x))

