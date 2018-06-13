from matplotlib import pyplot as plt
import scipy.misc
import numpy as np


img_path = 'no_reg_mc/cifar_mc%d.png'
index_min, index_max = 0, 9
#img_path = 'elbo_mc/cifar_mc%d.png'
# index_min, index_max = 0, 2
do_view = True

img_list = []
for ind in range(index_min, index_max+1):
    raw_img = scipy.misc.imread(img_path % ind)
    img = np.zeros((raw_img.shape[0] // 32, raw_img.shape[1] // 32, 32, 32, 3), dtype=raw_img.dtype)
    for i in range(raw_img.shape[0] // 32):
        for j in range(raw_img.shape[1] // 32):
            img[i, j, :, :, :] = raw_img[i * 32:(i + 1) * 32, j * 32:(j + 1) * 32, :]
    if ind != index_min:
        img = img[:, 1:]
    img_list.append(img)
all_img = np.concatenate(img_list, axis=1)
print(np.min(all_img), np.max(all_img))
print(all_img.shape)
n_row = 4
n_col = 15
all_img = all_img[:, :n_row*n_col]
if do_view:
    for ind in range(all_img.shape[0]):
        img = all_img[ind, :, :, :, :]
        img = np.reshape(img, (n_row, n_col, 32, 32, 3))
        images = np.zeros((n_row * 32, n_col * 32, 3), dtype=img.dtype)
        for i in range(n_row):
            for j in range(n_col):
                images[i * 32:(i + 1) * 32, j * 32:(j + 1) * 32, :] = img[i, j, :, :, :]
        scipy.misc.imsave('result.png', images)
        plt.imshow(images)
        plt.show()

else:
    image_list = []
    use_index = [8, 11]
    for ind in use_index:
        img = all_img[ind, :, :, :, :]
        img = np.reshape(img, (n_row, n_col, 32, 32, 1))
        images = np.zeros((n_row * 32, n_col * 32))
        for i in range(n_row):
            for j in range(n_col):
                images[i * 32:(i + 1) * 32, j * 32:(j + 1) * 32] = img[i, j, :, :, 0]
        image_list.append(images)
    image = np.concatenate(image_list, 0)
    plt.imshow(1 - image, cmap=plt.get_cmap('Greys'))
    plt.show()
    scipy.misc.imsave('samples/long2_reshape.jpg', image)
