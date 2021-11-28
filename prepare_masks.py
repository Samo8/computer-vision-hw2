import numpy as np
from pycocotools.coco import COCO
from PIL import Image
import os
from matplotlib import pyplot as plt

coco = COCO('./annotations/instances_default.json')

def loadImagesAndMasks(img_dir):
    result = []
    for id in coco.imgs:
        one = coco.imgs[id]
        image = np.array([])
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

        result.append((image, mask, one['file_name']))
        
    return result

# imagesWithMasks = loadImagesAndMasks()

# for data in imagesWithMasks:
#     plt.imsave(f'./masks/{data[2]}', data[1])

    # for img in data:
    #     plt.imshow(data[1])
    #     plt.show()

# for x in imagesWithMasks[11]:
#     plt.imshow(x)    
#     plt.show()



