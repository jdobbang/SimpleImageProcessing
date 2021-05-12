import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

plot_grid_size = (3, 6)

def plot_img(index, img, title):
    plt.subplot(plot_grid_size[0],plot_grid_size[1],index)
    plt.imshow(img[...,::-1])
    plt.axis('off'), plt.title(title) 
    
def display_untilKey(Pimgs, Titles, file_out = False):
    for img, title in zip(Pimgs, Titles):
        cv.imshow(title, img)
        if file_out == True:
            cv.imwrite(title + ".jpg", img)
    cv.waitKey(0)

def ellipseComposite(hand,eye):
    
    mask = np.zeros_like(eye)
    rows, cols,_ = mask.shape
    
    eye_mask=cv.ellipse(mask, ((rows/2, cols/2+cols/7), (rows/2,cols/4), 0), (255,255,255), -1)
    
    hand_mask = cv.bitwise_not(mask) # OR
    
    eyeRegion = cv.bitwise_and(eye, eye_mask)
    handRegion = cv.bitwise_and(hand, hand_mask)
    composite = eyeRegion+handRegion
    
    return composite

def __pyrUp(img, size = None):
    nt = tuple([x*2 for x in img.shape[:2]])
    if size == None:
        size = nt
    # bug?!
    if nt != size:
        upscale_img = cv.pyrUp(img, None, size)
    else:
        upscale_img = cv.pyrUp(img)
    return upscale_img

def generate_gaussian_pyramid(img, levels):
    GP = [img]
    for i in range(1, levels): # 0 to levels - 1 same as range(0, levels, 1)
        img = cv.pyrDown(img)
        GP.append(img)
    return GP

def generate_laplacian_pyramid(GP):
    levels = len(GP)
    LP = [] 
    for i in range(levels - 1, 0, -1):
        
        upsample_img = __pyrUp(GP[i], GP[i-1].shape[:2])
        
        laplacian_img = cv.subtract(GP[i-1], upsample_img)
        LP.append(laplacian_img)
    LP.reverse()
    return LP

def generate_pyramid_composition_image(Pimgs):
    levels = len(Pimgs)
    #print(levels)
    rows, cols = Pimgs[0].shape[:2] 
    composite_image = np.zeros((rows, cols + int(cols / 2 + 0.5), 3), dtype=Pimgs[0].dtype)
    composite_image[:rows, :cols, :] = Pimgs[0]
    i_row = 0
    for p in Pimgs[1:]:
        n_rows, n_cols = p.shape[:2]
        composite_image[i_row:i_row + n_rows, cols:cols + n_cols] = p
        i_row += n_rows
    return composite_image

def stitch_LandR(P_hand, P_eye):
    P_stitch = []
    for la,lb in zip(P_hand, P_eye):
        
        lpimg_stitch = ellipseComposite(la,lb)
        
        P_stitch.append(lpimg_stitch)
    return P_stitch

eye = cv.imread('eye.jpg')
hand = cv.imread('hand.jpg')

eye = cv.resize(eye,(300,300))
hand = cv.resize(hand,(300,300))

# sampling level 지정
GP_hand = generate_gaussian_pyramid(hand, 6)
GP_eye = generate_gaussian_pyramid(eye, 6)
LP_hand = generate_laplacian_pyramid(GP_hand)
LP_eye = generate_laplacian_pyramid(GP_eye)


# 피라미드 확인
display_untilKey([generate_pyramid_composition_image(GP_hand), 
                  generate_pyramid_composition_image(GP_eye),
                  generate_pyramid_composition_image(LP_hand),
                  generate_pyramid_composition_image(LP_eye)], 
                 ["GP_eye", "GP_hand", "LP_eye", "LP_hand"])
    

LP_stitch = stitch_LandR(LP_hand, LP_eye) 
GP_stitch = stitch_LandR(GP_hand, GP_eye)

display_untilKey([generate_pyramid_composition_image(GP_stitch),
                  generate_pyramid_composition_image(LP_stitch)], 
                 ["composite GP imgs", "composite LP imgs"])

recon_img = GP_stitch[-1] 
lp_maxlev = len(LP_stitch) - 1
plot_img(6, recon_img.copy(), "level: " + str(6))
print(lp_maxlev)
for i in range(lp_maxlev, -1, -1):
    recon_img = __pyrUp(recon_img, LP_stitch[i].shape[:2])
    plot_img(i + 1 + 12, recon_img.copy(), "level: " + str(i))
    recon_img = cv.add(recon_img, LP_stitch[i])
    plot_img(i + 1, recon_img.copy(), "level: " + str(i))
    plot_img(i + 1 + 6, LP_stitch[i].copy(), "level: " + str(i))
    
cols = hand.shape[1]
naive_mix = ellipseComposite(hand,eye)

display_untilKey([recon_img, naive_mix], ["blending", "direct connecting"])
plt.show()


