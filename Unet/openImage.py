
from PIL import Image

import numpy as np
name='test30y'
img_nameX = r"D:\YANG Luoxiao\Data\WPC\Generate\SecondDEOnly\%s.jpg"%(name)
imageX = Image.open(img_nameX).convert("L")

x=np.array(imageX)
print(1)