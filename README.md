# Images-Color-Space-Conversion
Color Space Conversion | Encoding and Decoding images (RGB:YUV)

Converting a RAW rgb image to a YUV format with Compression to the original image then Decompression and Reconstructing the RGB channels. 

Steps:
1. Converting RGB to YUV Color Space
2. Computing Peak Signal to Noise Ratio (PSNR)
3. Compression and Decompression:
    1. Original Image [4:4:4]
    2. Compressing [4:1:1]
    3. Compressing [4:2:2]
    4. Compressing and Decompressing using Average [4:1:1]
    5. Compressing and Decompressing using Median [4:1:1]
    6. Compressing and Decompressing using Average [4:2:2]
    7. Compressing and Decompressing using Median [4:2:2]
4. YUV to RGB Reconstruction
5. Plotting the Encoded and Decoded Images
