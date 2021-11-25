import numpy as np
from pycocotools.coco import COCO
from PIL import Image
import os
from matplotlib import pyplot as plt

coco = COCO('./annotations/instances_default.json')
img_dir = './dataset/'


def loadImagesAndMasks():
    result = []
    for id in coco.imgs:
        one = coco.imgs[id]
        try:
            image = np.array(Image.open(os.path.join(img_dir + 'furcullaria/', one['file_name'])))
        except Exception:
            pass
        try:
            image = np.array(Image.open(os.path.join(img_dir + 'zostera/', one['file_name'])))
        except Exception:
            pass

        cat_ids = coco.getCatIds()
        anns_ids = coco.getAnnIds(imgIds=one['id'], catIds=cat_ids, iscrowd=None)

        anns = coco.loadAnns(anns_ids)

        mask = coco.annToMask(anns[0])
        for i in range(len(anns)):
            mask += coco.annToMask(anns[i])

        result.append((image, mask))
        
    return result

imagesWithMasks = loadImagesAndMasks()

for data in imagesWithMasks:
    for img in data:
        plt.imshow(img)
        plt.show()

# for x in imagesWithMasks[11]:
#     plt.imshow(x)    
#     plt.show()



