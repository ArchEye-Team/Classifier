import torch
from torch.utils.data import Dataset as TorchDataset
from torchvision.io import read_image, ImageReadMode
from torchvision.transforms import v2


class Dataset(TorchDataset):
    def __init__(self, data_module, train):
        self.data_module = data_module
        self.train = train

        self.num_classes = 0
        self.class2id = {}
        self.id2class = []
        self.class_count = []

        self.dataset = []

        dataset_dir = self.data_module.data_path
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

            labeled_images = [(image_path, class_id) for image_path in
                              sorted((image_path for image_path in class_dir.iterdir() if image_path.is_file()),
                                     key=lambda image_path: image_path.name)]

            self.class_count[class_id] += len(labeled_images)

            split_size = int(len(labeled_images) * 0.8)

            self.dataset.extend(labeled_images[:split_size] if self.train else labeled_images[split_size:])

        self.class_weights = [1 - count / sum(self.class_count) for count in self.class_count]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        image_path, class_id = self.dataset[index]

        image = read_image(str(image_path), mode=ImageReadMode.RGB)

        transforms = [
            v2.Resize(size=(self.data_module.image_size, self.data_module.image_size), antialias=True)
        ]

        if self.train:
            transforms.extend([
                v2.RandomVerticalFlip(p=0.5),
                v2.RandomHorizontalFlip(p=0.5),
                v2.RandomRotation(degrees=60),
                v2.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3)
            ])

        transforms.extend([
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        image = v2.Compose(transforms)(image)

        return image, class_id
