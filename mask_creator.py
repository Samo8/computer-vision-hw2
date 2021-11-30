from pycocotools.coco import COCO
import numpy as np

class MaskCreator():
    def __init__(self):
        self.ann_file='./annotations/annotations.json'
        self.coco = COCO(self.ann_file)
        categories_ids = self.coco.getCatIds()
        self.categories = self.coco.loadCats(categories_ids)

    def __findImageCocoObj__(self, className, imgName):
        catIds  = self.coco.getCatIds(catNms=className)
        imgIds  = self.coco.getImgIds(catIds=catIds)
        images  = []
        images += self.coco.loadImgs(imgIds)

        relevantImg = None
        for i in range(len(images)):
            # print("images", images[i])
            if images[i]["file_name"] == imgName:
                relevantImg = images[i]
                break
        return relevantImg

    def __createMaskInner__(self, class_name, img):
        cat_ids = self.coco.getCatIds(catNms=class_name)
        ann_ids = self.coco.getAnnIds(imgIds=img['id'], catIds=cat_ids, iscrowd=None)

        anns = self.coco.loadAnns(ann_ids)
        mask = np.zeros((img['height'], img['width']))
        for m in range(len(anns)):
            mask = np.maximum(self.coco.annToMask(anns[m]), mask)
       
        return mask

    # returns mask and className
    def __createMask__(self, imageName):
        imgObj = None
        className = None

        # find imageCocoObj
        for i in range (len(self.categories)):
            imgObj = self.__findImageCocoObj__(self.categories[i]["name"], imageName)
            if imgObj != None:
                className = self.categories[i]["name"]
                break
        
        return self.__createMaskInner__(className, imgObj), className