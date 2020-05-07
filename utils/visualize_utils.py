from matplotlib import pyplot as plt


def visualize_keypoints(image, key_pts):
    """ Plot keypoints on face """
    plt.figure(figsize=(10, 10))
    plt.imshow(image, cmap='gray')
    plt.scatter(key_pts[:, 0], key_pts[:, 1], s=2, c='g')
    
    plt.show()
    
    return 0