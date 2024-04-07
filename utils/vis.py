import numpy as np
import cv2
import matplotlib.pyplot as plt

def gray_enhancing_tvg(bs, range, lambd, alpha):    
    n_sample = bs.shape[1]
    pix_index = np.arange(1, n_sample//2)
    range_index = range/n_sample * pix_index
    tvg = lambd*np.log10(range_index) + 2*alpha*range_index
    tvg = tvg.tolist()
    tvg.insert(0,0)
    tvg = np.array(tvg)
    tvg = np.maximum(tvg,0)
    tvg_dual = tvg.tolist()[::-1]+tvg.tolist()
    bs_correct = np.zeros_like(bs)
    bs_correct = tvg_dual * bs
    return bs_correct
    
def npy2img(bs_tvg):
    '''
    input:
        bs_tvg: tvg_corrected bs, float
    return:
        img_clip: uint8

    '''
    img_clip = np.clip(bs_tvg, 0,255) #clip
    img_clip = np.round(img_clip).astype(np.uint8) #to uint8
    return img_clip

def hist_calc(img, ratio):
    '''refer https://zhaoxuhui.top/blog/2018/06/12/GrayScaleStretch.html'''
    bins = np.arange(img.max())
    # 调用Numpy实现灰度统计
    hist, bins = np.histogram(img, bins)
    total_pixels = img.shape[0] * img.shape[1]
    # 计算获得ratio%所对应的位置，
    # 这里ratio为0.02即为2%线性化，0.05即为5%线性化
    min_index = int((ratio) * total_pixels)
    max_index = int((1 - ratio) * total_pixels)
    min_gray = 0
    max_gray = 0
    # 统计最小灰度值(A)
    sum = 0
    for i in range(hist.__len__()):
        sum = sum + hist[i]
        if sum > min_index:
            min_gray = i
            break
    # 统计最大灰度值(B)
    sum = 0
    for i in range(hist.__len__()):
        sum = sum + hist[i]
        if sum > max_index:
            max_gray = i
            break
    return min_gray, max_gray


def linearStretch(img, new_min, new_max, ratio):
    '''from https://zhaoxuhui.top/blog/2018/06/12/GrayScaleStretch.html'''
    # 获取原图除去2%像素后的最小、最大灰度值(A、B)
    old_min, old_max = hist_calc(img, ratio)
    # 对原图中所有小于或大于A、B的像素都赋为A、B
    img1 = np.where(img < old_min, old_min, img)
    img2 = np.where(img1 > old_max, old_max, img1)
    # print('old min = %d,old max = %d' % (old_min, old_max))
    # print('new min = %d,new max = %d' % (new_min, new_max))
    # 按照线性拉伸公式计算新的灰度值
    img3 = (new_max - new_min) / (old_max - old_min) * (img2 - old_min) + new_min
    img3 = np.uint8(np.minimum(np.maximum(img3, 1), 255))
    return img3

def img_process(img, new_min, new_max, ratio):
    '''refer https://zhaoxuhui.top/blog/2018/06/12/GrayScaleStretch.html'''
    # 采用2%灰度直方图分段线性拉伸
    old_min, old_max = hist_calc(img, ratio)
    old_min = 1
    img_max = np.max(img)
    img_out = np.zeros([img.shape[0], img.shape[1]], np.uint8)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if img[i, j] <= 1:
                img_out[i, j] = old_min
            elif img[i, j] > old_max:
                img_out[i, j] = ((new_max - 100) / (img_max - old_max)) * (
                        img[i, j] - old_max) + 100
            else:
                img_out[i, j] = ((100 - new_min) / (old_max - old_min)) * (
                        img[i, j] - old_min) + new_min
    return img_out

def logTransform(img, v=200, c=256):
    '''refer https://zhaoxuhui.top/blog/2018/06/12/GrayScaleStretch.html'''
    # # 获取原图除去2%像素后的最小、最大灰度值(A、B)
    # old_min, old_max = hist_calc(img, 0.01)
    # # 对原图中所有小于或大于A、B的像素都赋为A、B
    img1 = np.where(img < 1, 1, img)
    img2 = np.where(img1 > c, c, img1)
    img_normalized = img2 * 1.0 / np.max(img2)
    log_res = c * (np.log(1 + v * img_normalized) / np.log(v + 1))
    img_new = np.uint8(np.minimum(np.maximum(log_res, 0), 255))
    return img_new

def logTransform_gain(img, v=200, c=256): #过曝，需针对每张图像调节c
    '''refer https://zhaoxuhui.top/blog/2018/06/12/GrayScaleStretch.html'''
    img = img*10.
    old_min, old_max = hist_calc(img, 0.01)
    img1 = np.where(img < old_min, old_min, img)
    img2 = np.where(img1 > old_max, old_max, img1)
    img_normalized = img2 * 1.0 / np.max(img2)
    log_res = old_max * (np.log(1 + v * img_normalized) / np.log(v + 1))
    img_new = np.uint8(np.minimum(np.maximum(log_res, 0), 255))
    return img_new

def gamaTransform(img, c=256, r=0.5):
    '''refer https://zhaoxuhui.top/blog/2018/06/12/GrayScaleStretch.html'''
    img1 = np.where(img < 1, 1, img)
    img2 = np.where(img1 > c, c, img1)
    img_normalized = img2 * 1.0 / np.max(img2)
    gama_res = c* (img_normalized ** r)
    img_new = np.uint8(np.minimum(np.maximum(gama_res, 0), 255))

    return img_new

def draw_hist(img):
    plt.hist(img.ravel(), bins=int(np.max(img)-np.min(img)))
    plt.show()

def draw_points_on_image(points, image, radius=1, thickness=1):
    '''
    points: numpy array, [[x,y], ...]
    image: must be 1-channel, uint8
    '''
    color_img = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    for point in points:
        cv2.circle(color_img, tuple(point), radius, (0,0,255), thickness)
    
    return color_img
