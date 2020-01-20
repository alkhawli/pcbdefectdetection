import numpy as np
import cv2
from skimage.measure import compare_ssim
import imutils
from skimage.feature import register_translation
import scipy.ndimage as ndi
import matplotlib.pyplot as plt

############### Functions ###############
def showfigure(winTitle, img):
    cv2.namedWindow(winTitle, cv2.WINDOW_NORMAL)        
    while(1):
        cv2.imshow(winTitle, img)   
        k = cv2.waitKey(1)
        if k==27:    # Esc key to stop
            cv2.imwrite("./output/"+winTitle+".png", img)
            break
        else:
            continue
        
def match_images(img1, img2,img2color, ratio = 2, withvisualization=1):
    """Given two images, returns the matches"""
    detector = cv2.xfeatures2d.SIFT_create()
    matcher =  cv2.BFMatcher(cv2.NORM_L2)
    kp1, desc1 = detector.detectAndCompute(img1, None)
    kp2, desc2 = detector.detectAndCompute(img2, None)
    raw_matches = matcher.knnMatch(desc1, trainDescriptors = desc2, k = 2) 
    mkp1, mkp2 = [], []
    for m in raw_matches:
        if len(m) == 2 and m[0].distance < m[1].distance * ratio:
            m = m[0]
            mkp1.append( kp1[m.queryIdx] )
            mkp2.append( kp2[m.trainIdx] )
    kp_pairs = list(zip(mkp1, mkp2))
    if kp_pairs:
        mkp1, mkp2 = zip(*kp_pairs)
        p1 = np.float32([kp.pt for kp in mkp1])
        p2 = np.float32([kp.pt for kp in mkp2])
        if len(kp_pairs) >= 4:
            H, status = cv2.findHomography(p1, p2, cv2.RANSAC, 50.0)
        else:
            H, status = None, None
        if withvisualization:
            visualizeMatching(img1, img2, kp_pairs, status, H)
        h,w = img1.shape[:2]
        pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
        dst = cv2.perspectiveTransform(pts,H)
        perspectiveM = cv2.getPerspectiveTransform(np.float32(dst),pts)
        found = cv2.warpPerspective(img2,perspectiveM,(w,h))
        return found
    else:
        print ("No matches found")

def visualizeMatching(img1, img2, kp_pairs, status = None, H = None):
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    vis = np.zeros((max(h1, h2), w1+w2), np.uint8)
    vis[:h1, :w1] = img1
    vis[:h2, w1:w1+w2] = img2
    vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)
    if H is not None:
        corners = np.float32([[0, 0], [w1, 0], [w1, h1], [0, h1]])
        corners = np.int32( cv2.perspectiveTransform(corners.reshape(1, -1, 2), H).reshape(-1, 2) + (w1, 0) )
        cv2.polylines(vis, [corners], True, (255, 255, 255))
    if status is None:
        status = np.ones(len(kp_pairs), np.bool_)
    p1 = np.int32([kpp[0].pt for kpp in kp_pairs])
    p2 = np.int32([kpp[1].pt for kpp in kp_pairs]) + (w1, 0)
    green = (0, 255, 0)
    red = (0, 0, 255)
    for (x1, y1), (x2, y2), inlier in zip(p1, p2, status):
        if inlier:
            col = green
            cv2.circle(vis, (x1, y1), 2, col, -1)
            cv2.circle(vis, (x2, y2), 2, col, -1)
        else:
            col = red
            r = 2
            thickness = 3
            cv2.line(vis, (x1-r, y1-r), (x1+r, y1+r), col, thickness)
            cv2.line(vis, (x1-r, y1+r), (x1+r, y1-r), col, thickness)
            cv2.line(vis, (x2-r, y2-r), (x2+r, y2+r), col, thickness)
            cv2.line(vis, (x2-r, y2+r), (x2+r, y2-r), col, thickness)
    for (x1, y1), (x2, y2), inlier in list(zip(p1, p2, status)):
        if inlier:
            cv2.line(vis, (x1, y1), (x2, y2), green)
    showfigure("usecase_1_matching",vis)

def autocorrelateandshift(image,offset_image,withplot=0):
    shift, error, diffphase = register_translation(image, offset_image,upsample_factor=10)
    shifted = ndi.shift(offset_image, shift)
    if withplot:
        fig = plt.figure(figsize=(8, 3))
        ax1 = plt.subplot(1, 3, 1)
        ax2 = plt.subplot(1, 3, 2, sharex=ax1, sharey=ax1)
        ax3 = plt.subplot(1, 3, 3)
        ax1.imshow(img1, cmap='gray')
        ax1.set_axis_off()
        ax1.set_title('Reference image')
        ax2.imshow(offset_image.real, cmap='gray')
        ax2.set_axis_off()
        ax2.set_title('Offset image')
        # Show the output of a cross-correlation to show what the algorithm is
        # doing behind the scenes
        image_product = np.fft.fft2(image) * np.fft.fft2(offset_image).conj()
        cc_image = np.fft.fftshift(np.fft.ifft2(image_product))
        ax3.imshow(cc_image.real)
        ax3.set_axis_off()
        ax3.set_title("Cross-correlation")
        #plt.show()
        print(f"Detected pixel offset (y, x): {shift}")
        cv2.imwrite("3.jpg", shifted)
    return shifted



def createMorphology(image,shifted,withplot=0):
    thresh1=cv2.adaptiveThreshold(image,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,2)
    thresh2=cv2.adaptiveThreshold(shifted,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,2)
    img_bwx = cv2.bitwise_xor(thresh1,thresh2)
    if withplot:
        showfigure("usecase_1_Difference",img_bwx)
    img_median = cv2.medianBlur(img_bwx, 11) # Add median filter to image
    return cv2.morphologyEx(img_median, cv2.MORPH_OPEN, np.ones((2,2), np.uint8))  # filtre


def findContours(morphology,imageB,withplot=1):
    cnts = cv2.findContours(morphology.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    hh,ww = imageB.shape[:2]
    for c in cnts:
        (x, y, w, h) = cv2.boundingRect(c)
        if x>0.1*ww and x <0.9*ww and y>0.1*hh and y <0.9*hh and len(c)>10:
            cv2.rectangle(imageB, (x-50, y-50), (x + 50, y + 50), (255, 255, 255), 10)
    if withplot:
        showfigure("usecase_1_Outcomes",imageB)
    return 1

def StructuralSimilarityIndex(image,shifted,withplot=1):
    (score, diff) = compare_ssim(image, shifted, full=True)
    diff = (diff * 255).astype("uint8")
    print("SSIM: {}".format(score))
    return score
        
############### Main ###############

originalpicture='./data/01.jpg'
snappedpicture='./data/01_missing_hole_01_r.jpg'

img_original = cv2.imread(originalpicture, 0)
image_snapped_gray = cv2.imread(snappedpicture, 0)
image_snapped_colored =cv2.imread(snappedpicture, 1)

findmatch=match_images(img_original,image_snapped_gray,image_snapped_colored)       
showfigure("usecase_1_machingoutcome",findmatch)
shiftedandcorrelated=autocorrelateandshift(img_original,findmatch)
StructuralSimilarityIndex(img_original,shiftedandcorrelated)
morphology=createMorphology(img_original,shiftedandcorrelated)
showfigure("usecase_1_Morphology",morphology)
findContours(morphology,shiftedandcorrelated,withplot=1)





