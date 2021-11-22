import cv2
# import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def match_image(img, template):
	# image_gray = img.copy()
	# methods = [cv2.TM_CCOEFF, cv2.TM_CCOEFF_NORMED, cv2.TM_CCORR, cv2.TM_CCORR_NORMED,
	# 	cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]
	# result_list = []
	w, h = template.shape[::-1]
	match_method = cv2.TM_CCOEFF_NORMED
	res = cv2.matchTemplate(img, template, match_method)
	minval, maxval, minloc, maxloc = cv2.minMaxLoc(res)
	return maxval
	

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
			sum += match_image(cv2.imread(image, 0), cv2.imread(etalon, 0))
		
		average = sum / len(etalons)
		print(f"{image.split('/')[-1]}: {average}")
		result_sum += average

	print(f"Result -> {result_sum / len(images)}")

def getKeypointsAndDescriptors(imagePaths, detectionAlghoritm):
	if (detectionAlghoritm == 'SIFT'):
		alghoritm = cv2.SIFT_create()
	elif (detectionAlghoritm == 'FAST'):
		alghoritm = cv2.FastFeatureDetector_create()
	elif (detectionAlghoritm == 'ORB'):
		alghoritm = cv2.ORB_create(nfeatures=1000)
	else:
		raise "detection alghoritm not implemented"
	keypoints = []
	descriptors = []
	for path in imagePaths:
		if (detectionAlghoritm != 'FAST'):
			print ("aaaaaa")
			keypoint, descriptor = alghoritm.detectAndCompute(cv2.imread(path), None)
		else:
			print ("TU")
			img = (cv2.imread(path))

			
			star = cv2.xfeatures2d.StarDetector_create()
			brief = cv2.xfeatures2d.BriefDescriptorExtractor_create()
			kp = star.detect(img,None)
			keypoint, descriptor = brief.compute(img, kp)
			
			# # Initiate FAST detector
			# star = cv2.xfeatures2d.StarDetector_create()
			# # Initiate BRIEF extractor
			# brief = cv2.xfeatures2d.BriefDescriptorExtractor_create()
			# # find the keypoints with STAR
			# kp = star.detect(img,None)
			# # compute the descriptors with BRIEF
			# keypoint, descriptor = brief.compute(img, kp)

		keypoints.append(keypoint)
		descriptors.append(descriptor)
	return keypoints, descriptors

def flattenList(list):
	flatten_list = []
	for subl in list:
		for item in subl:
			flatten_list.append(item)
	return flatten_list

# for etalon in etalons:
# 	sum = 0
# 	for furNieco in furcullariaImages:
# 		sum += match_image(cv2.imread(furNieco, 0), cv2.imread(etalon, 0))
# 	average = sum / len(furcullariaImages)
# 	print(f"{etalon.split('/')[-1]}: {average}")


current_path = Path().absolute()
etalons = readImages(f"{current_path}/dataset/etalons/zostera")
dataset = readImages(f"{current_path}/dataset/zostera")

# sift = cv2.SIFT_create()
# surf = cv2.SURF_create()
# orb = cv2.ORB_create(nfeatures=1000)
# etalon = cv2.imread('./dataset/etalons/furcullaria/f_etallon8.jpeg', 0)
# image = cv2.imread('./dataset/furcullaria/furcellaria3.jpeg', 0)


alghoritm_type = 'FAST'

keypoints_e, descriptors_e = getKeypointsAndDescriptors(etalons, alghoritm_type)
descriptors_e = flattenList(descriptors_e)
keypoints_e = flattenList(keypoints_e)
# print(np.array(descriptors_e).shape())

keypoints_sift_img, descriptors_img = getKeypointsAndDescriptors(dataset, alghoritm_type)

# print(descriptors_e.shape)
# keypoints_sift_e, descriptors_e = sift.detectAndCompute(etalon, None)
# keypoints_sift_img, descriptors_img = sift.detectAndCompute(image, None)
# keypoints_surf, descriptors = surf.detectAndCompute(etalons[0], None)
# keypoints_orb, descriptors = orb.detectAndCompute(etalons[0], None)

# img = cv2.drawKeypoints(image, keypoints_sift_img, None)


def calculateMatches(descriptors_image, descriptors_etalon):
	matcher = cv2.FlannBasedMatcher(dict(algorithm = 0), dict(checks = 200))
	# bf = cv2.BFMatcher()
	for ind, desc in enumerate(descriptors_image):
		matches = matcher.knnMatch(np.array(descriptors_etalon), np.array(desc), 2)
		m = np.array(matches)
		print(m.shape)
		good = []
		for m,n in matches:
			if m.distance < 0.75 * n.distance:
				good.append([m])
		print(f"{ind} -> {len(good)}")

calculateMatches(descriptors_img, descriptors_e)


# matching_result = cv2.drawMatches(etalon, keypoints_sift_e, image, keypoints_sift_img, good, None, flags=2)
# print(matching_result)
# matches = bf.match(descriptors_e, descriptors_img)
# matches = sorted(matches, key = lambda x:x.distance)

# cv22.drawMatchesKnn expects list of lists as matches.
# img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,good,flags=2)

# print(matches)
# cv2.imshow("Image", img)
# cv2.imshow("AAA", matching_result)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
