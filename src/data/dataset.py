import albumentations as A
import cv2
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset as TorchDataset


class Dataset(TorchDataset):
    def __init__(self, data_module, train):
        self.train = train

        self.num_classes = 0
        self.class2id = {}
        self.id2class = []
        self.class_count = []

        self.image_paths = []
        self.class_ids = []

        dataset_dir = data_module.data_path
        class_dirs = sorted((class_dir for class_dir in dataset_dir.iterdir() if class_dir.is_dir()),
                            key=lambda class_dir: class_dir.name)

        for class_dir in class_dirs:
            class_name = class_dir.name
            if class_name not in self.class2id:
                self.class2id[class_name] = self.num_classes
                self.id2class.append(class_name)
                self.class_count.append(0)
                self.num_classes += 1

            class_id = self.num_classes - 1

            image_paths = sorted((image_path for image_path in class_dir.iterdir() if image_path.is_file()),
                                 key=lambda image_path: image_path.name)

            self.class_count[class_id] += len(image_paths)

            split_size = int(len(image_paths) * 0.8)

            for image_path in (image_paths[:split_size] if self.train else image_paths[split_size:]):
                self.image_paths.append(image_path)
                self.class_ids.append(class_id)

        self.class_weights = [1 - count / sum(self.class_count) for count in self.class_count]

        self.train_transforms = A.Compose([
            A.SmallestMaxSize(max_size=850),
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15),
            A.RandomCrop(height=512, width=512),
            A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])

        self.transforms = A.Compose([
            A.Resize(height=512, width=512),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_path, class_id = self.image_paths[index], self.class_ids[index]

        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.train:
            image = self.train_transforms(image=image)['image']
        else:
            image = self.transforms(image=image)['image']

        return image, class_id
