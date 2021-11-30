import cv2
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import os

def match_image(img, template):
	w, h = template.shape[::-1]
	match_method = cv2.TM_CCOEFF_NORMED
	res = cv2.matchTemplate(img, template, match_method)
	_, maxval, _, _ = cv2.minMaxLoc(res)
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

def getKeypointsAndDescriptorsSIFT(imagePaths):	
	alghoritm = cv2.SIFT_create()
	keypoints = []
	descriptors = []
	for path in imagePaths:
		keypoint, descriptor = alghoritm.detectAndCompute(cv2.imread(path), None)

		keypoints.append(keypoint)
		descriptors.append(descriptor)
	return keypoints, descriptors

def flattenList(list):
	flatten_list = []
	for subl in list:
		for item in subl:
			flatten_list.append(item)
	return flatten_list


current_path = Path().absolute()
etalons = readImages(f"{current_path}/dataset/etalons/furcullaria")
dataset = readImages(f"{current_path}/dataset/furcullaria")

# etalons = readImages(f"{current_path}/dataset/etalons/zostera")
# dataset = readImages(f"{current_path}/dataset/zostera")

keypoints_e, descriptors_e = getKeypointsAndDescriptorsSIFT(etalons)
descriptors_e = flattenList(descriptors_e)
keypoints_e = flattenList(keypoints_e)

keypoints_sift_img, descriptors_img = getKeypointsAndDescriptorsSIFT(dataset)


def calculateMatches(descriptors_image, descriptors_etalon):
	matcher = cv2.FlannBasedMatcher(dict(algorithm = 0), dict(checks = 200))
	for ind, desc in enumerate(descriptors_image):
		matches = matcher.knnMatch(np.array(descriptors_etalon), np.array(desc), 2)
		m = np.array(matches)
		# print(m.shape)
		good = []
		for m,n in matches:
			if m.distance < 0.75 * n.distance:
				good.append([m])
		print(f"{ind} -> {len(good)}")

calculateMatches(descriptors_img, descriptors_e)

def orb_implementation():
	img1 = cv2.imread('dataset/furcullaria/furcularia11.jpeg', 0)
	img2 = cv2.imread('dataset/etalons/furcullaria/f_etallon11.jpeg', 0)

	orb = cv2.ORB_create(nfeatures=1000)
	kp1, des1 = orb.detectAndCompute(img1, None)
	kp2, des2 = orb.detectAndCompute(img2, None)

	if (not (des1 is None and des2 is None)):
		bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
		matches = bf.match(des1, des2)
		matches = sorted(matches, key=lambda x: x.distance)

		match_img = cv2.drawMatches(img1, kp1, img2, kp2, matches[:30], None)
		cv2.imwrite(os.path.join('task3b.png'), match_img)
		plt.imshow(match_img),
		plt.show() 

# orb_implementation()