import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

path = 'lena_color.tiff'
image_tiff = Image.open(path)
rgb = np.array(image_tiff)

def RGB2YUV( rgb ): #RGB 2 YUV
    
    m = np.array([[ 0.299     ,  0.587     ,  0.114      ],
                  [-0.14714119, -0.28886916,  0.43601035 ],
                  [ 0.61497538, -0.51496512, -0.10001026 ]])

    yuv = np.dot(rgb,m.T)
    return yuv
yuv = RGB2YUV( rgb ) 

y = yuv[:, :, 0] #YUV - Channels
u = yuv[:, :, 1]
v = yuv[:, :, 2]
output = [yuv, y, u, v]
titles = ['YUV', 'Y-Channel', 'U-Channel', 'V-Channel']
for i in range(4):
    plt.subplot(2, 2, i+1)
    plt.axis('off')
    plt.title(titles[i])
    if i == 0:
        plt.imshow(output[i].astype(int))
    else:
        plt.imshow(output[i], cmap = 'gray')
plt.show()

#%% PSNR(RGB vs YUV)

def PSNR(img1, img2):
    mse = np.mean( (img1 - img2) ** 2 )
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * np.log10(PIXEL_MAX / np.sqrt(mse))

Difference = PSNR(rgb,yuv)
print('PSNR between RGB and YUV is :', Difference)

n = int(input("""
Enter a number:
For 4:4:4                : 0
For 4:1:1                : 1
For 4:2:2                : 2
For 4:1:1 with (Avarage) : 3
For 4:1:1 with (Median)  : 4
For 4:2:2 with (Avarage) : 5
For 4:2:2 with (Median)  : 6 \n\n"""))
    
def YUV2xRGB(yuv, n):
    
    if (n == 0): #4:4:4
        x1 = yuv[:, :, 0]
        x2 = yuv[:, :, 1]
        x3 = yuv[:, :, 2]
        
    if (n == 11): #4:1:1 (Compressed)
        x1 = yuv[:, :, 0]
        x2 = yuv[:, :, 1].reshape(-1,4)
        x2 = np.delete(x2, [1, 2, 3], 1)
        x2 = np.insert(x2, 1, [[0],[0],[0]], axis=1).reshape(512,512)
        x3 = yuv[:, :, 2].reshape(-1,4)
        x3 = np.delete(x3, [1, 2, 3], 1)
        x3 = np.insert(x3, 1, [[0],[0],[0]], axis=1).reshape(512,512)
    
    if (n == 22): #4:2:2 (Compressed)
        x1 = yuv[:, :, 0]
        x2 = yuv[:, :, 1].reshape(-1,4)
        x2 = np.delete(x2, [2, 3], 1)
        x2 = np.insert(x2, (1,2), 0, axis=1).reshape(512,512)
        x3 = yuv[:, :, 2].reshape(-1,4)
        x3 = np.delete(x3, [2, 3], 1)
        x3 = np.insert(x3, (1,2), 0, axis=1).reshape(512,512)
    
    if (n == 1): #4:1:1
        x1 = yuv[:, :, 0]
        x2 = yuv[:, :, 1].reshape(-1,4)
        x2 = np.delete(x2, [1, 2, 3], 1)
        x2 = np.insert(x2, 1, [[0],[0],[0]], axis=1)
        for i in range(65536):
            x2[i][1], x2[i][2], x2[i][3] = x2[i][0], x2[i][0], x2[i][0]
        x2 = x2.reshape(512,512)
        x3 = yuv[:, :, 2].reshape(-1,4)
        x3 = np.delete(x3, [1, 2, 3], 1)
        x3 = np.insert(x3, 1, [[0],[0],[0]], axis=1)
        for i in range(65536):
            x3[i][1], x3[i][2], x3[i][3] = x3[i][0], x3[i][0], x3[i][0]
        x3 = x3.reshape(512,512)
    
    if (n == 2): #4:2:2
        x1 = yuv[:, :, 0]
        x2 = yuv[:, :, 1].reshape(-1,4)
        x2 = np.delete(x2, [2, 3], 1)
        x2 = np.insert(x2, (1,2), 0, axis=1)
        for i in range(65536):
            x2[i][1], x2[i][3] = x2[i][0], x2[i][2]
        x2 = x2.reshape(512,512)
        x3 = yuv[:, :, 2].reshape(-1,4)
        x3 = np.delete(x3, [2, 3], 1)
        x3 = np.insert(x3, (1,2), 0, axis=1)
        for i in range(65536):
            x3[i][1], x3[i][3]= x3[i][0], x3[i][2]
        x3 = x3.reshape(512,512)

    if (n==3): #4:1:1 with (Avarage)
        x1 = yuv[:, :, 0]
        x2 = yuv[:, :, 1].reshape(-1,4)
        x2a = [np.average(i) for i in x2]
        x2 = np.delete(x2, [1, 2, 3], 1)
        x2 = np.insert(x2, 1, [[0],[0],[0]], axis=1)
        for i in range(65536):
            x2[i][1], x2[i][2], x2[i][3] = x2a[i], x2a[i], x2a[i]
        x2 = x2.reshape(512,512)
        x3 = yuv[:, :, 2].reshape(-1,4)
        x3a = [np.average(i) for i in x3]
        x3 = np.delete(x3, [1, 2, 3], 1)
        x3 = np.insert(x3, 1, [[0],[0],[0]], axis=1)
        for i in range(65536):
            x3[i][1], x3[i][2], x3[i][3] = x3a[i], x3a[i], x3a[i]
        x3 = x3.reshape(512,512)
    
    if (n==4): #4:1:1 with (Median)
        x1 = yuv[:, :, 0]
        x2 = yuv[:, :, 1].reshape(-1,4)
        x2a = [np.median(i) for i in x2]
        x2 = np.delete(x2, [1, 2, 3], 1)
        x2 = np.insert(x2, 1, [[0],[0],[0]], axis=1)
        for i in range(65536):
            x2[i][1], x2[i][2], x2[i][3] = x2a[i], x2a[i], x2a[i]
        x2 = x2.reshape(512,512)
        x3 = yuv[:, :, 2].reshape(-1,4)
        x3a = [np.median(i) for i in x3]
        x3 = np.delete(x3, [1, 2, 3], 1)
        x3 = np.insert(x3, 1, [[0],[0],[0]], axis=1)
        for i in range(65536):
            x3[i][1], x3[i][2], x3[i][3] = x3a[i], x3a[i], x3a[i]
        x3 = x3.reshape(512,512)


    if (n==5): #4:2:2 with (Avarage)
        x1 = yuv[:, :, 0]
        x2 = yuv[:, :, 1].reshape(-1,4)
        x2a = [np.average(i) for i in x2]
        x2 = np.delete(x2, [2, 3], 1)
        x2 = np.insert(x2, (1,2), 0, axis=1)
        for i in range(65536):
            x2[i][1], x2[i][3] = x2a[i], x2a[i]
        x2 = x2.reshape(512,512)
        x3 = yuv[:, :, 2].reshape(-1,4)
        x3a = [np.average(i) for i in x3]
        x3 = np.delete(x3, [2, 3], 1)
        x3 = np.insert(x3, (1,2), 0, axis=1)
        for i in range(65536):
            x3[i][1], x3[i][3]= x3a[i], x3a[i]
        x3 = x3.reshape(512,512)

    
    if (n==6): #4:2:2 with (Median)
        x1 = yuv[:, :, 0]
        x2 = yuv[:, :, 1].reshape(-1,4)
        x2a = [np.median(i) for i in x2]
        x2 = np.delete(x2, [2, 3], 1)
        x2 = np.insert(x2, (1,2), 0, axis=1)
        for i in range(65536):
            x2[i][1], x2[i][3] = x2a[i], x2a[i]
        x2 = x2.reshape(512,512)
        x3 = yuv[:, :, 2].reshape(-1,4)
        x3a = [np.median(i) for i in x3]
        x3 = np.delete(x3, [2, 3], 1)
        x3 = np.insert(x3, (1,2), 0, axis=1)
        for i in range(65536):
            x3[i][1], x3[i][3]= x3a[i], x3a[i]
        x3 = x3.reshape(512,512)
        
    reim = np.stack((x1, x2, x3))
    reim = np.rollaxis(reim, 0, 3)
    return reim
reim = YUV2xRGB(yuv, n)

def YUV2RGB( yuv ): #Returning RGB'
     
    m = np.array([[ 1     ,  0      ,  1.13983 ],
                  [ 1     , -0.39465, -0.58060 ],
                  [ 1     ,  2.03211,  0       ]])
     
    xrgb = np.dot(yuv,m.T)
    return xrgb
xrgb = YUV2RGB( reim )

#%% Plotting

titles = ['Encoded', 'Decoded']
titlex = ['Orginal','Converted']
for i in range(2):
    plt.subplot(1, 2, i+1)
    plt.axis('off')
    plt.title(titles[i])
    if (n == 1 or 3 or 4):
        plt.imshow(YUV2RGB( YUV2xRGB(yuv, 11) ).astype(int))
    if (n == 2 or 5 or 6):
        plt.imshow(YUV2RGB( YUV2xRGB(yuv, 22) ).astype(int))
    if (n == 0):
        plt.title(titlex[i])
        plt.imshow( rgb )
    
plt.imshow(xrgb.astype(int))
plt.show()
print('PSNR between RGB and RGB\' :', PSNR(rgb,xrgb))
