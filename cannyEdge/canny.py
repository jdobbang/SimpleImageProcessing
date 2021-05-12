import numpy as np
import cv2

#컨볼루션 함
def convolution(image, kernel, average=False):
    # 컬러이미지를 input 하였을때
    if len(image.shape) == 3:        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #이미지, 커널 shape 
    cv2.imshow("gray image",image)
    image_row, image_col = image.shape
    kernel_row, kernel_col = kernel.shape
    #출력이미지 
    output = np.zeros(image.shape)
    #패딩을 하기 위해 추가할 값 : 커널의 크기에 따라
    pad_height = int((kernel_row - 1) / 2) 
    pad_width = int((kernel_col - 1) / 2) 
    #패딩
    padded_image = np.zeros((image_row + (2 * pad_height), image_col + (2 * pad_width)))
    padded_image[pad_height:padded_image.shape[0] - pad_height, pad_width:padded_image.shape[1] - pad_width] = image
    #행과 열에 따라 커널을 패딩한 이미지에 컨볼루션 
    for row in range(image_row):
        for col in range(image_col):
            output[row, col] = np.sum(kernel * padded_image[row:row + kernel_row, col:col + kernel_col])
            if average:
                output[row, col] /= kernel.shape[0] * kernel.shape[1]
    return output

#소벨 엣지 함
def sobel_edge_detection(image):
    #필터 선언
    filter = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    #x,y 방향에 따른 image를 filter에 컨볼루
    new_image_x = convolution(image, filter)
    new_image_y = convolution(image, np.flip(filter.T, axis=0))
    #그래디언트 magnitude 계산 
    gradient_magnitude = np.sqrt(np.square(new_image_x) + np.square(new_image_y))
    gradient_magnitude *= 255.0 / gradient_magnitude.max()
    #그래디언트 방향을 radian에서 degree로 변환
    gradient_direction = np.arctan2(new_image_y, new_image_x)
    gradient_direction = np.rad2deg(gradient_direction)
    gradient_direction += 180
     
    return gradient_magnitude, gradient_direction

def non_max_suppression(gradient_magnitude, gradient_direction):
    #gradient magnitude 모양
    image_row, image_col = gradient_magnitude.shape
    #출력 이미지
    output = np.zeros(gradient_magnitude.shape)
    gradient_magnitude = cv2.copyMakeBorder( gradient_magnitude, 4, 4, 4, 4, cv2.BORDER_CONSTANT)
    PI = 180
    # 그래디언트 각 방향에 따라 인접 8픽셀들 픽셀과의 비교하
    for row in range(1, image_row - 1):
        for col in range(1, image_col - 1):
            direction = gradient_direction[row, col]
            neighbor = []
            if (0 <= direction < PI / 8) or (15 * PI / 8 <= direction <= 2 * PI):
                 for i in range (1,5):
                     before_pixel = gradient_magnitude[row, col - i]
                     after_pixel = gradient_magnitude[row, col + i]
                     neighbor.append(before_pixel)
                     neighbor.append(after_pixel)
 
            elif (PI / 8 <= direction < 3 * PI / 8) or (9 * PI / 8 <= direction < 11 * PI / 8):
                 for i in range (1,5):
                     before_pixel = gradient_magnitude[row+i, col - i]
                     after_pixel = gradient_magnitude[row-i, col + i]
                     neighbor.append(before_pixel)
                     neighbor.append(after_pixel)                
                
            elif (3 * PI / 8 <= direction < 5 * PI / 8) or (11 * PI / 8 <= direction < 13 * PI / 8):
                 for i in range (1,5):
                     before_pixel = gradient_magnitude[row-i, col]
                     after_pixel = gradient_magnitude[row+i, col ]
                     neighbor.append(before_pixel)
                     neighbor.append(after_pixel)   
 
            else:
                 for i in range (1,5):
                     before_pixel = gradient_magnitude[row-i, col - i]
                     after_pixel = gradient_magnitude[row+i, col + i]
                     neighbor.append(before_pixel)
                     neighbor.append(after_pixel)   
            # 전,후 픽셀들과 비교하여 해당 gradient magnitude가 크다면 그대로 output
            neighbor = np.array(neighbor)
            if np.all(neighbor<gradient_magnitude[row,col]):
                output[row, col] = gradient_magnitude[row, col]

    return output
 
# thresdholding
def threshold(image,weak):
    output = np.zeros(image.shape)
    #strongResult = np.zeros(image.shape)
    #weakResult = np.zeros(image.shape)
    
    strong = 255
    low = 5
    high = 20
    
    strong_row, strong_col = np.where(image >= high)
    weak_row, weak_col = np.where((image <= high) & (image >= low))
    
    output[strong_row, strong_col] = strong
    output[weak_row, weak_col] = weak
    
    return output ,strong_row , strong_col , weak_row, weak_col
 
 
def hysteresis(image, weak):
    image_row, image_col = image.shape
 
    top_to_bottom = image.copy()
 
    for row in range(1, image_row):
        for col in range(1, image_col):
            if top_to_bottom[row, col] == weak:
                if top_to_bottom[row, col + 1] == 255 or top_to_bottom[row, col - 1] == 255 or top_to_bottom[row - 1, col] == 255 or top_to_bottom[
                    row + 1, col] == 255 or top_to_bottom[
                    row - 1, col - 1] == 255 or top_to_bottom[row + 1, col - 1] == 255 or top_to_bottom[row - 1, col + 1] == 255 or top_to_bottom[
                    row + 1, col + 1] == 255:
                    top_to_bottom[row, col] = 255
                else:
                    top_to_bottom[row, col] = 0
 
    bottom_to_top = image.copy()
 
    for row in range(image_row - 1, 0, -1):
        for col in range(image_col - 1, 0, -1):
            if bottom_to_top[row, col] == weak:
                if bottom_to_top[row, col + 1] == 255 or bottom_to_top[row, col - 1] == 255 or bottom_to_top[row - 1, col] == 255 or bottom_to_top[
                    row + 1, col] == 255 or bottom_to_top[
                    row - 1, col - 1] == 255 or bottom_to_top[row + 1, col - 1] == 255 or bottom_to_top[row - 1, col + 1] == 255 or bottom_to_top[
                    row + 1, col + 1] == 255:
                    bottom_to_top[row, col] = 255
                else:
                    bottom_to_top[row, col] = 0
 
    right_to_left = image.copy()
 
    for row in range(1, image_row):
        for col in range(image_col - 1, 0, -1):
            if right_to_left[row, col] == weak:
                if right_to_left[row, col + 1] == 255 or right_to_left[row, col - 1] == 255 or right_to_left[row - 1, col] == 255 or right_to_left[
                    row + 1, col] == 255 or right_to_left[
                    row - 1, col - 1] == 255 or right_to_left[row + 1, col - 1] == 255 or right_to_left[row - 1, col + 1] == 255 or right_to_left[
                    row + 1, col + 1] == 255:
                    right_to_left[row, col] = 255
                else:
                    right_to_left[row, col] = 0
 
    left_to_right = image.copy()
 
    for row in range(image_row - 1, 0, -1):
        for col in range(1, image_col):
            if left_to_right[row, col] == weak:
                if left_to_right[row, col + 1] == 255 or left_to_right[row, col - 1] == 255 or left_to_right[row - 1, col] == 255 or left_to_right[
                    row + 1, col] == 255 or left_to_right[
                    row - 1, col - 1] == 255 or left_to_right[row + 1, col - 1] == 255 or left_to_right[row - 1, col + 1] == 255 or left_to_right[
                    row + 1, col + 1] == 255:
                    left_to_right[row, col] = 255
                else:
                    left_to_right[row, col] = 0
 
    final_image = top_to_bottom + bottom_to_top + right_to_left + left_to_right
 
    final_image[final_image > 255] = 255
    final_row,final_col = np.where(final_image >= 255)
    
    return final_row, final_col
 
#이미지 입력
image = cv2.imread("Lenna.png")

image_str = np.zeros(np.shape(image),np.uint8)

image_wk = np.zeros(np.shape(image),np.uint8)

image_true = np.zeros(np.shape(image),np.uint8)

kernel = np.array([[2,4,5,4,2],
          [4,9,12,9,4],
          [5,12,15,12,5],
          [4,9,12,9,4],
          [2,4,5,4,2]],np.uint8)/159
 
blurred_image = cv2.filter2D(image,-1,kernel)
 
gradient_magnitude, gradient_direction = sobel_edge_detection(blurred_image)
 
new_image = non_max_suppression(gradient_magnitude, gradient_direction)

weak = 100
new_image,strong_row , strong_col , weak_row, weak_col = threshold(new_image, weak==weak)
cv2.imshow("new_image",new_image)
true_row, true_col = hysteresis(new_image, weak)

image_str[strong_row,strong_col] = (255,0,0)
image_wk [weak_row, weak_col] = (0,255,255)
image_true[true_row, true_col] = (0,0,255)

result3 = image_wk + image_true + image_str

cv2.imshow("result3",result3 )
cv2.imshow("high",image_true)
cv2.imshow("st",image_str)
cv2.imshow("wk",image_wk)
cv2.waitKey(0)
