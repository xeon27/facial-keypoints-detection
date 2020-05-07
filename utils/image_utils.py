import cv2


def read_image(path):
    """ Read an image and convert to RGB """
    image = cv2.imread(path)
    
    # Convert to RGB since cv2 reads as BGR
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    return image
	

def rotate_image(image, angle):
    """
        Rotate image by the given angle 
        Parameters:
        image (ndarray): source image
        angle (int): angle value in degrees
        Returns:
        image (ndarray): Rotated image
    """
    if angle in [-90, 90]:
        if angle > 0:
            image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        else:
            image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    elif angle in [-180, 180]:
        image = cv2.rotate(image, cv2.ROTATE_180)
        
    return image