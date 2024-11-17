import numpy as np
import cv2
import albumentations as A
import random

class RandomDistortion():
    def __init__(self, prob=0.9):
        super(RandomDistortion, self).__init__()
        self.prob = prob

    def __call__(self, image, label):
        if np.random.rand() < self.prob:
            if len(image.shape) == 3:
                new_image = image[0]
            else:
                new_image = image
            trans = A.Compose([
                A.OneOf([
                    A.ElasticTransform(
                        p=1, alpha=300, sigma=20, border_mode=cv2.BORDER_CONSTANT),
                    A.GridDistortion(
                        num_augteps=2, distort_limit=0.8, p=1, border_mode=cv2.BORDER_CONSTANT),
                    A.OpticalDistortion(p=1, distort_limit=-0.9, shift_limit=1,
                                        border_mode=cv2.BORDER_CONSTANT, always_apply=True),
                    A.Compose([
                        A.OpticalDistortion(
                            p=1, distort_limit=0.5, shift_limit=1, border_mode=cv2.BORDER_CONSTANT, always_apply=True),
                        A.GridDistortion(
                            num_augteps=2, distort_limit=0.3, p=1, border_mode=cv2.BORDER_CONSTANT)
                    ], p=1)
                ], p=1)
            ], p=1)
            x = trans(image=new_image, mask=label)
            new_image = x['image']
            label = x['mask']

            if(len(image.shape)) == 3:
                image = new_image[np.newaxis, :, :]
            else:
                image = new_image
        return image, label

class RandomFlip():
    def __init__(self, prob=0.5):
        super(RandomFlip, self).__init__()
        self.prob = prob

    def __call__(self, image, label):
        if len(image.shape) == 3:
            new_image = image[0]
        else:
            new_image = image

        trans = A.Compose([
            A.OneOf([
                A.HorizontalFlip(always_apply=True, p=1.0),
                A.VerticalFlip(p=1),
                A.Rotate(limit=270, p=1)
            ], p=1)
        ], p=1)
        x = trans(image=new_image, mask=label)
        new_image = x['image']
        label = x['mask']

        if(len(image.shape)) == 3:
            image = new_image[np.newaxis, :, :]
        else:
            image = new_image
        return image, label

class RandomBlur():
    def __init__(self, prob=0.9):
        super(RandomBlur, self).__init__()
        self.prob = prob

    def __call__(self, image, label):
        if np.random.rand() < self.prob:
            if len(image.shape) == 3:
                new_image = image[0]
            else:
                new_image = image

            if np.random.rand() < 0.5:
                crop_rate = [0.2, 0.4]
                trans = A.MotionBlur(p=1, blur_limit=(
                    3, 21), allow_aughifted=True, always_apply=False)

                x = trans(image=new_image, mask=label)
                blur_map = x['image']
                blur_label = x['mask']

                h = new_image.shape[0]
                w = new_image.shape[1]
                crop_h = np.random.randint(h*crop_rate[0], h*crop_rate[1])
                crop_w = np.random.randint(w*crop_rate[0], w*crop_rate[1])

                h1 = np.random.randint(0, h-crop_h-1)
                w1 = np.random.randint(0, w-crop_w-1)

                new_image[w1:w1+crop_w, h1:h1 +
                          crop_h] = blur_map[w1:w1+crop_w, h1:h1+crop_h]
                label[w1:w1+crop_w, h1:h1 +
                      crop_h] = blur_label[w1:w1+crop_w, h1:h1+crop_h]
            else:
                crop_rate = [0.5, 0.8]
                h = new_image.shape[0]
                w = new_image.shape[1]
                crop_h = np.random.randint(h*crop_rate[0], h*crop_rate[1])
                crop_w = np.random.randint(w*crop_rate[0], w*crop_rate[1])

                h1 = np.random.randint(0, h-crop_h-1)
                w1 = np.random.randint(0, w-crop_w-1)

                blur_map = new_image[w1:w1+crop_w, h1:h1 +
                                     crop_h]
                blur_label = label[w1:w1+crop_w, h1:h1 +
                                   crop_h]

                trans = A.ZoomBlur(
                    max_factor=1.1, step_factor=(0.01, 0.02), p=1)
                x = trans(image=blur_map, mask=blur_label)
                blur_map = x['image']
                blur_label = x['mask']

                new_image[w1:w1+crop_w, h1:h1 +
                          crop_h] = blur_map[:, :]
                label[w1:w1+crop_w, h1:h1 +
                      crop_h] = blur_label[:, :]

            if(len(image.shape)) == 3:
                image = new_image[np.newaxis, :, :]
            else:
                image = new_image
        return image, label

class RandomNoise():
    def __init__(self, noise_range=5, prob=0.9):
        super(RandomNoise, self).__init__()
        self.noise_range = noise_range
        self.prob = prob

    def __call__(self, image, label):
        if np.random.rand() < self.prob:
            w, h, c = image.shape
            clip_min = image.min()
            clip_max = image.max()
            noise = np.random.randint(
                -self.noise_range,
                self.noise_range,
                (w, h, c)
            )

            image = (image + noise).clip(clip_min,
                                         clip_max).astype(image.dtype)
        return image, label

class RandomRoll():
    def __init__(self, prob=0.9):
        super(RandomRoll, self).__init__()
        self.prob = prob

    def __call__(self, image, label):
        if np.random.rand() < self.prob:
            if(len(image.shape)) == 3:
                new_image = image[0]
            else:
                new_image = image
            # new_image = cv2.resize(new_image,[1000,1000])
            A = new_image.shape[0] / 2.0
            w = 1 / new_image.shape[1]

            randomroll = round(np.random.rand(), 2)

            def sin(x): return A * np.sin(max(0.5, randomroll)*np.pi*x * w)

            def cos(x): return A * np.cos(max(0.5, randomroll)*np.pi*x * w)

            if np.random.rand() < 0.5:
                for i in range(new_image.shape[0]):
                    new_image[:, i] = np.roll(new_image[:, i], int(sin(i)))
                    label[:, i] = np.roll(label[:, i], int(sin(i)))
            else:
                for i in range(new_image.shape[0]):
                    new_image[:, i] = np.roll(new_image[:, i], int(cos(i)))
                    label[:, i] = np.roll(label[:, i], int(cos(i)))

            if(len(image.shape)) == 3:
                image = new_image[np.newaxis, :, :]
            else:
                image = new_image
        return image, label

class RandomBrightness():
    def __init__(self, bright_range=0.15, prob=0.9):
        super(RandomBrightness, self).__init__()
        self.bright_range = bright_range
        self.prob = prob

    def __call__(self, image, label):
        if np.random.rand() < self.prob:
            bright_factor = np.random.uniform(
                1-self.bright_range, 1+self.bright_range)
            new_image = (image*bright_factor)
            new_image[new_image > 255] = 255
            new_image[new_image < 0] = 0
            image = new_image.astype(image.dtype)
            if np.random.rand() < 0.3:
                if len(image.shape) == 3:
                    new_image = image[0]
                else:
                    new_image = image
                crop_rate = [0.6, 0.7]
                h = new_image.shape[0]
                w = new_image.shape[1]
                # crop_h = np.random.randint(h*crop_rate[0], h*crop_rate[1])
                crop_h = np.random.randint(h*crop_rate[0], h*crop_rate[1])
                crop_w = np.random.randint(w*crop_rate[0], w*crop_rate[1])

                if np.random.rand() < 0.5:
                    bright_factor = 1+self.bright_range*1.3
                else:
                    bright_factor = 1-self.bright_range*1.3

                bright_map = (new_image*bright_factor)
                bright_map[bright_map > 255] = 255
                bright_map[bright_map < 0] = 0

                h1 = np.random.randint(0, h-crop_h-1)
                w1 = np.random.randint(0, w-crop_w-1)

                new_image[w1:w1+crop_w, h1:h1 +
                          crop_h] = bright_map[w1:w1+crop_w, h1:h1+crop_h]
                if(len(image.shape)) == 3:
                    image = new_image[np.newaxis, :, :]
                else:
                    image = new_image
        return image, label

class RandomScratch():
    def __init__(self, prob=0.9):
        super(RandomScratch, self).__init__()
        self.prob = prob
        self.iteration = random.randint(1,5)
    
    def consecutive_points(self,height,width,x,y,n):
        d = random.randint(46, 89)  #Direction of the consecutive line
        direction_rad = np.deg2rad(d)  # Convert degrees to radians
        dx = np.cos(direction_rad)  # The x-component in the direction
        dy = np.sin(direction_rad)  # The y-component in the direction
        x_new = min(max(x, 0), width - 1)
        y_new = min(max(y, 0), height - 1)
        # points = [(y_new, x_new)]
        points = []
        for i in range(n):
            x_new = int(x + dx * i)
            y_new = int(y + dy * i)
            x_new = min(max(x_new, 0), width - 1)
            y_new = min(max(y_new, 0), height - 1)
            points.append((y_new, x_new))
        return points

    def __call__(self, image, label):
        if np.random.rand() < self.prob:
            if(len(image.shape)) == 3:
                new_image = image[0]
            else:
                new_image = image
            base_image = new_image
            base_label = label
            
            for i in range(self.iteration):
                n = random.randint(20,100)  # Number of consecutive points
                d = random.randint(45,120)  # Direction
                l = random.randint(50,200)  # Length of extension
                height, width = image.shape[:2]
                x_augtart = random.randint(0, width - n)
                y_augtart = random.randint(0, height - 1)

                points = self.consecutive_points(height,width,x_augtart,y_augtart,n)

                # Calculate the unit vector of direction d 
                direction_rad = np.deg2rad(d)  # Convert degrees to radians
                dx = np.cos(direction_rad)  # The x-component in the direction
                dy = np.sin(direction_rad)  # The y-component in the direction

                # extend operation
                extended_points = []
                for y, x in points:
                    x_new = int(x + dx * l)
                    y_new = int(y + dy * l)
                    x_new = min(max(x_new, 0), width - 1)
                    y_new = min(max(y_new, 0), height - 1)
                    extended_points.append((y_new, x_new))

        
                for i in range(len(points)):
                    x_augtart,y_augtart = points[i]
                    x,y = extended_points[i]
                    color = int(base_image[x_augtart,y_augtart])
                    new_image = cv2.line(new_image, (x_augtart,y_augtart), (x,y), color, 2)
                    color = int(base_label[x_augtart,y_augtart])
                    label = cv2.line(base_label, (x_augtart,y_augtart), (x,y), color, 2)

            if(len(image.shape)) == 3:
                image = new_image[np.newaxis, :, :]
            else:
                image = new_image
        return image, label

from osgeo import gdal
if __name__ == "__main__":
    map_path = './map.tif'
    label_path = './label.tif'
    map_data = gdal.Open(map_path)
    label_data = gdal.Open(label_path)
    map = map_data.ReadAsArray(0,0,map_data.RasterXSize,map_data.RasterYSize)
    label = label_data.ReadAsArray(0,0,label_data.RasterXSize,label_data.RasterYSize)
    scratch = RandomScratch(prob=1)
    map,label = scratch.__call__(map,label)
    cv2.imwrite('./map_aug.tif',map)
    cv2.imwrite('./label_aug.tif',label)
