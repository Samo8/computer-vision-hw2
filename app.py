import cv2 as cv
from matplotlib import pyplot as pltimg

def match_image(img, template):
	# image_gray = img.copy()
	# methods = [cv.TM_CCOEFF, cv.TM_CCOEFF_NORMED, cv.TM_CCORR, cv.TM_CCORR_NORMED,
	# 	cv.TM_SQDIFF, cv.TM_SQDIFF_NORMED]
	# result_list = []
	w, h = template.shape[::-1]
	match_method = cv.TM_CCOEFF_NORMED
	res = cv.matchTemplate(img, template, match_method)
	minval, maxval, minloc, maxloc = cv.minMaxLoc(res)
	# if match_method in [cv.TM_SQDIFF, cv.TM_SQDIFF_NORMED]:
	# 	topleft = minloc
	# else:
	# topleft = maxloc
	# btm_right = (topleft[0] + w, topleft[1] + h)
	return maxval
	# cv.rectangle(img, topleft, btm_right, 255, 2)
	# pltimg.subplot(121),pltimg.imshow(res,cmap = 'gray')
	# pltimg.title('Result that matches'), pltimg.xticks([]), pltimg.yticks([])
	# pltimg.subplot(122),pltimg.imshow(img,cmap = 'gray')
	# pltimg.title('Detection Point of image'), pltimg.xticks([]), pltimg.yticks([])
	# pltimg.suptitle(match_method)
	# pltimg.show()
	
	# for match_method in methods:
		# image = image_gray.copy()
		# res = cv.matchTemplate(img, template, match_method)
		# minval, maxval, minloc, maxloc = cv.minMaxLoc(res)
		# if match_method in [cv.TM_SQDIFF, cv.TM_SQDIFF_NORMED]:
		# 	topleft = minloc
		# else:
		# 	topleft = maxloc
		# btm_right = (topleft[0] + w, topleft[1] + h)
		# cv.rectangle(img, topleft, btm_right, 255, 2)
		# pltimg.subplot(121),pltimg.imshow(res,cmap = 'gray')
		# pltimg.title('Result that matches'), pltimg.xticks([]), pltimg.yticks([])
		# pltimg.subplot(122),pltimg.imshow(img,cmap = 'gray')
		# pltimg.title('Detection Point of image'), pltimg.xticks([]), pltimg.yticks([])
		# pltimg.suptitle(match_method)
		# pltimg.show()
		# result_list.append([minval, maxval, minloc, maxloc])
	# return result_list
	
from pathlib import Path

def readImages(dir: str):
	images = []
	for p in Path(dir).iterdir() :
		if p.is_file() and str(p).split(".")[-1].lower() in {"jpeg", "jpg", "png"}:
			with p.open() as f :
				images.append(f.name)
	return images

def evaluateEtalonOnImages(etalons, images):
	result_sum = 0
	for image in images:
		sum = 0
		for etalon in etalons:
			sum += match_image(cv.imread(image, 0), cv.imread(etalon, 0))
		
		average = sum / len(etalons)
		print(f"{image.split('/')[-1]}: {average}")
		result_sum += average

	print(f"Result -> {result_sum / len(images)}")

# for etalon in etalons:
# 	sum = 0
# 	for furNieco in furcullariaImages:
# 		sum += match_image(cv.imread(furNieco, 0), cv.imread(etalon, 0))
# 	average = sum / len(furcullariaImages)
# 	print(f"{etalon.split('/')[-1]}: {average}")


current_path = Path().absolute()
etalons = readImages(f"{current_path}/dataset/etalons/zostera")
furcullariaImages = readImages(f"{current_path}/dataset/zostera")


sift = cv.xfeatures2d.SIFT_create()
surf = cv.xfeatures2d.SURF_create()
orb = cv.ORB_create(nfeatures=1000)

keypoints_sift, descriptors = sift.detectAndCompute(etalons[0], None)
keypoints_surf, descriptors = surf.detectAndCompute(etalons[0], None)
keypoints_orb, descriptors = orb.detectAndCompute(etalons[0], None)

img = cv.drawKeypoints(etalons[0], keypoints_sift, None)

cv.imshow("Image", img)
cv.waitKey(0)
cv.destroyAllWindows()

