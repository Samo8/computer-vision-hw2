import cv2 as cv
# from matplotlib import pyplot as pltimg
import matplotlib.pyplot as plt
from numpy import mat


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


sift = cv.SIFT_create()
# surf = cv.SURF_create()
# orb = cv.ORB_create(nfeatures=1000)
etalon = cv.imread('/Users/samueldubovec/taltech/computer_vision/its8030-2021-hw2/dataset/etalons/furcullaria/f_etallon8.jpeg', 0)
image = cv.imread('/Users/samueldubovec/taltech/computer_vision/its8030-2021-hw2/dataset/furcullaria/furcularia11.jpeg', 0)

keypoints_sift_e, descriptors_e = sift.detectAndCompute(etalon, None)
keypoints_sift_img, descriptors_img = sift.detectAndCompute(image, None)
# keypoints_surf, descriptors = surf.detectAndCompute(etalons[0], None)
# keypoints_orb, descriptors = orb.detectAndCompute(etalons[0], None)

# img = cv.drawKeypoints(image, keypoints_sift_img, None)

bf = cv.BFMatcher(cv.NORM_L1, crossCheck=True)
matches = bf.match(descriptors_e, descriptors_img)
matches = sorted(matches, key = lambda x:x.distance)

print(matches)
# cv.imshow("Image", img)
matching_result = cv.drawMatches(etalon,keypoints_sift_e,image,keypoints_sift_img,matches[:10],None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
# matching_result = cv.drawMatches(etalon, keypoints_sift_e, image, keypoints_sift_img, matches[:50], None, flags=2)
# print(matching_result)
plt.imshow(matching_result),plt.show()
# cv.imshow("AAA", matching_result)
# cv.waitKey(0)
# cv.destroyAllWindows()
