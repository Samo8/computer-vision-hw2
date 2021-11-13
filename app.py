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
	if match_method in [cv.TM_SQDIFF, cv.TM_SQDIFF_NORMED]:
		topleft = minloc
	else:
		topleft = maxloc
	btm_right = (topleft[0] + w, topleft[1] + h)
	print(maxval)
	cv.rectangle(img, topleft, btm_right, 255, 2)
	pltimg.subplot(121),pltimg.imshow(res,cmap = 'gray')
	pltimg.title('Result that matches'), pltimg.xticks([]), pltimg.yticks([])
	pltimg.subplot(122),pltimg.imshow(img,cmap = 'gray')
	pltimg.title('Detection Point of image'), pltimg.xticks([]), pltimg.yticks([])
	pltimg.suptitle(match_method)
	pltimg.show()
	
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
	
# img = cv.imread('furcularia9.jpeg', 0)
# img = cv.imread('furcularia3.jpeg', 0)
# img_teplate= cv.imread('furcularia10_2.jpeg', 0)

from pathlib import Path

def readImages(dir: str):
	images = []
	for p in Path(dir).iterdir() :
		if p.is_file() and str(p).split(".")[-1].lower() in {"jpeg", "jpg", "png"}:
			with p.open() as f :
				images.append(f.name)
	return images


	# files = os.listdir(dir_path)
	# for file in files:
	# 	if os.path.isfile(os.path.join(dir_path, file)):
	# 		f = open(os.path.join(dir_path, file),'r')
	# 		print(f)
	# 		f.close()
current_path = Path().absolute()
etalons = readImages(f"{current_path}/dataset/etalons/furcullaria")
print(etalons)
furcullariaImages = readImages(f"{current_path}/dataset/furcullaria")
print(furcullariaImages)


# match_image(img, img_teplate)