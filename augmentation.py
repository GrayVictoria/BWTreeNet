import numpy as np
import cv2
import albumentations as A

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
                        num_steps=2, distort_limit=0.8, p=1, border_mode=cv2.BORDER_CONSTANT),
                    A.OpticalDistortion(p=1, distort_limit=-0.9, shift_limit=1,
                                        border_mode=cv2.BORDER_CONSTANT, always_apply=True),
                    A.Compose([
                        A.OpticalDistortion(
                            p=1, distort_limit=0.5, shift_limit=1, border_mode=cv2.BORDER_CONSTANT, always_apply=True),
                        A.GridDistortion(
                            num_steps=2, distort_limit=0.3, p=1, border_mode=cv2.BORDER_CONSTANT)
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
                    3, 21), allow_shifted=True, always_apply=False)

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

    def __call__(self, image, label):
        if np.random.rand() < self.prob:
            if(len(image.shape)) == 3:
                new_image = image[0]
            else:
                new_image = image
                
            times = int(np.random.rand()/0.4)+1
            
            for t in range(times):

                crop_rate_h = [0.01, 0.1]
                crop_rate_w = [0.2, 0.3]
                h = new_image.shape[0]
                w = new_image.shape[1]
                crop_h = np.random.randint(h*crop_rate_h[0], h*crop_rate_h[1])
                crop_w = np.random.randint(w*crop_rate_w[0], w*crop_rate_w[1])

                h1 = np.random.randint(0, h-crop_h-1)
                w1 = np.random.randint(0, w-crop_w-1)

                scratch_map = new_image[w1:w1+crop_w, h1:h1 + crop_h]
                crop_map_ori = new_image[w1:w1+crop_w, h1:h1 + crop_h]
                scratch_label = label[w1:w1+crop_w, h1:h1 + crop_h]
                crop_label_ori = label[w1:w1+crop_w, h1:h1 + crop_h]

                scratch_map_pad = scratch_map[:, 0]
                scratch_map_pad = scratch_map_pad[:, np.newaxis]
                scratch_label_pad = scratch_label[:, 0]
                scratch_label_pad = scratch_label_pad[:, np.newaxis]

                scratch_map = np.pad(
                    scratch_map_pad,  ((0, 0), (scratch_map.shape[1]-1, 0)), mode='edge')
                scratch_label = np.pad(
                    scratch_label_pad,  ((0, 0), (scratch_label.shape[1]-1, 0)), mode='edge')

                if np.random.rand() < 0.8:
                    rand_rotate = np.random.randint(15, 45)
                    M = cv2.getRotationMatrix2D(
                        (crop_h/2, crop_w/2), rand_rotate, 1.0)
                    scratch_map = cv2.warpAffine(scratch_map, M, (crop_h, crop_w))
                    scratch_label = cv2.warpAffine(
                        scratch_label, M, (crop_h, crop_w))
                    location = scratch_map == 0
                    scratch_map[scratch_map == 0] = crop_map_ori[scratch_map == 0]
                    scratch_label[location] = crop_label_ori[location]
                new_image[w1:w1+crop_w, h1:h1 + crop_h] = scratch_map
                label[w1:w1+crop_w, h1:h1 + crop_h] = scratch_label

            if(len(image.shape)) == 3:
                image = new_image[np.newaxis, :, :]
            else:
                image = new_image
        return image, label
