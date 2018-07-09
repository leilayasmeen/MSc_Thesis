### File to convert latent-space interpolated images into 10 one-hot images


## Interpolated images are done by finding the mean of the encoded image and then adding random noise epsilon ~ N(0,1) * the standard deviation

## Option 1: requires producing 45,000 images. unfeasible
# Take the num_classes-dimensional label vector and the encoded image
# sample 10 image latents from a normal distribution with a mean centered at the image code
# distribute the labels for these images based on the interpolated latent vector


## Option 2: just requires array copying
# Take the num_classes-dimensional label vector and the image array
# Make 10 copies of the image array, and distribute labels for these images based on the interpolated latent vector


## Code for option 2:

x_augmentations_array = np.load(BLAH BLAH)
y_augmentations_array = np.load(BLAH BLAH)

y_augmentations_array_onehot = np.zeros(1, NUM_CLASSES*y_augmentations_array.shape[0])
x_augmentations_array_onehot = np.zeros(x_augmentations_array.shape[0]*NUM_CLASSES, x_augmentations_array.shape[1],
	x_augmentations_array.shape[2], x_augmentations_array.shape[3])

#for k in xrange(y_augmentations_array_onehot):
#	y_augmentations_array_onehot[k] = np.sum(y_augmentations_array[np.round(k/NUM_CLASSES, 0),:])
#	x_augmentations_array_onehot[k,:] = x_augmentations_array[np.round(k/NUM_CLASSES, 0),:]

for k in range(y_augmentations_array.shape[0]):
	for w in range(NUM_CLASSES):
		y_augmentations_array_onehot[k*NUM_CLASSES + w,:] = y_augmentations_array[k,:]*NUM_CLASSES

np.save(y_augmentations_array_onehot, '/Users/wildflowerlyi/Desktop/y_augmentations_array_onehot.npy')
np.save(x_augmentations_array_onehot, '/Users/wildflowerlyi/Desktop/x_augmentations_array_onehot.npy')