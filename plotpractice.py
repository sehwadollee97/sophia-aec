import matplotlib.pyplot as plt
import matplotlib.patches as patches

from PIL import Image

img=Image.open('Forests.jpg')

plt.imshow(img)

ax=plt.gca()


rect = patches.Rectangle((80,10),
                 70,
                 100,
                 linewidth=2,
                 edgecolor='cyan',
                 fill = False)

ax.add_patch(rect)

plt.show()