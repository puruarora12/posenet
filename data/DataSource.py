import os
import torch.utils.data as data
from torchvision import transforms as T
from PIL import Image
import numpy as np

class DataSource(data.Dataset):
    def __init__(self, root, resize=256, crop_size=224, train=True):
        self.root = os.path.expanduser(root)
        self.resize = resize
        self.crop_size = crop_size
        self.train = train

        self.image_poses = []
        self.images_path = []

        self._get_data()

        # TODO: Define preprocessing

        # Load mean image
        self.mean_image_path = os.path.join(self.root, 'mean_image.npy')
        if os.path.exists(self.mean_image_path):
            self.mean_image = np.load(self.mean_image_path)
            print("Mean image loaded!")
        else:
            self.mean_image = self.generate_mean_image()

    def _get_data(self):

        if self.train:
            txt_file = self.root + 'dataset_train.txt'
        else:
            txt_file = self.root + 'dataset_test.txt'

        with open(txt_file, 'r') as f:
            next(f)  # skip the 3 header lines
            next(f)
            next(f)
            for line in f:
                fname, p0, p1, p2, p3, p4, p5, p6 = line.split()
                p0 = float(p0)
                p1 = float(p1)
                p2 = float(p2)
                p3 = float(p3)
                p4 = float(p4)
                p5 = float(p5)
                p6 = float(p6)
                self.image_poses.append((p0, p1, p2, p3, p4, p5, p6))
                self.images_path.append(self.root + fname)

    def generate_mean_image(self):
        print("Computing mean image:")

        # TODO: Compute mean image
        w,h = Image.open(self.images_path[0]).size
        mean_image = np.zeros((h,w,3) ,dtype =np.float64) 
        # Initialize mean_image

        # Iterate over all training images
        # Resize, Compute mean, etc...
        imagesLen = len(self.images_path)
        for images_path in self.images_path:
            img = Image.open(images_path)
            if(h<w):
                h_new = self.resize/h
                img = img.resize((int(w*h_new) , self.resize) , Image.NEAREST)
            else:
                w_new = self.resize/w
                img = img.resize((self.resize , int(h*w_new)) , Image.NEAREST )
            
            img = np.array(img , dtype = np.float64)/255
            if mean_image.shape == img.shape:
                mean_image = (mean_image+img)
            else:
                mean_image = np.zeros(img.shape , dtype = np.float64)
                mean_image = (mean_image+img)
        mean_image = mean_image/imagesLen

        # Store mean image
        np.save('mean_image' , mean_image)
        img = Image.fromarray(np.array(mean_image*255 ,  dtype = np.uint8) , mode='RGB')
        img.save('mean_image.jpg')
        print("Mean image computed!")

        return mean_image

    def __getitem__(self, index):
        """
        return the data of one image
        """
        img_path = self.images_path[index]
        img_pose = self.image_poses[index]

        data = Image.open(img_path)

        # TODO: Perform preprocessing
        Tresize = T.Resize(size = self.resize)
        data = Tresize(data)

        data_np = np.array(data, dtype=np.float64)/255
        data_np = np.subtract(data_np , self.mean_image)

        data_np= (data_np-np.amin(data_np))/(np.amax(data_np) - np.amin(data_np))
        data = Image.fromarray(np.array(data_np*255 , dtype = np.uint8))

        if self.train:
            Tcrop = T.RandomCrop((224,224))
        else:
            Tcrop = T.CenterCrop((224,224))
        data = Tcrop(data)

        Tnormal = T.Compose([T.ToTensor() , T.Normalize((0.5 , 0.5 ,0.5) , (0.5,0.5,0.5))])
        data = Tnormal(data)
        return data, img_pose

    def __len__(self):
        return len(self.images_path)