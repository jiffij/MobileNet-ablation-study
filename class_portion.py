import matplotlib
from MyDataset import MyDataset
matplotlib.use('Agg')
import torch
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR100
from torch.utils.data import DataLoader, random_split
from collections import Counter
from tabulate import tabulate

training_set_mean = (0.5071, 0.4865, 0.4409)
training_set_std = (0.2673, 0.2564, 0.2762)

labels = [
    "apple",
    "aquarium_fish",
    "baby",
    "bear",
    "beaver",
    "bed",
    "bee",
    "beetle",
    "bicycle",
    "bottle",
    "bowl",
    "boy",
    "bridge",
    "bus",
    "butterfly",
    "camel",
    "can",
    "castle",
    "caterpillar",
    "cattle",
    "chair",
    "chimpanzee",
    "clock",
    "cloud",
    "cockroach",
    "couch",
    "crab",
    "crocodile",
    "cup",
    "dinosaur",
    "dolphin",
    "elephant",
    "flatfish",
    "forest",
    "fox",
    "girl",
    "hamster",
    "house",
    "kangaroo",
    "keyboard",
    "lamp",
    "lawn_mower",
    "leopard",
    "lion",
    "lizard",
    "lobster",
    "man",
    "maple_tree",
    "motorcycle",
    "mountain",
    "mouse",
    "mushroom",
    "oak_tree",
    "orange",
    "orchid",
    "otter",
    "palm_tree",
    "pear",
    "pickup_truck",
    "pine_tree",
    "plain",
    "plate",
    "poppy",
    "porcupine",
    "possum",
    "rabbit",
    "raccoon",
    "ray",
    "road",
    "rocket",
    "rose",
    "sea",
    "seal",
    "shark",
    "shrew",
    "skunk",
    "skyscraper",
    "snail",
    "snake",
    "spider",
    "squirrel",
    "streetcar",
    "sunflower",
    "sweet_pepper",
    "table",
    "tank",
    "telephone",
    "television",
    "tiger",
    "tractor",
    "train",
    "trout",
    "tulip",
    "turtle",
    "wardrobe",
    "whale",
    "willow_tree",
    "wolf",
    "woman",
    "worm"
]

def get_train_dataset(dataset_dir='./datasets', batch_size=128, seed=0):
    training_transformation = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(training_set_mean, training_set_std)
    ])
    valid_transformation = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(training_set_mean, training_set_std)
    ])
    train_set = CIFAR100(root=dataset_dir, train=True, download=True)

    train_set, validation_set = random_split(train_set, [40000, 10000], generator=torch.Generator().manual_seed(seed))
    train_set = MyDataset(train_set, transform=training_transformation)
    validation_set = MyDataset(validation_set, transform=valid_transformation)
    # train_loader = DataLoader(train_set, batch_size=batch_size)
    # valid_loader = DataLoader(validation_set, batch_size=batch_size)
    return train_set



if __name__ == '__main__':
    train = get_train_dataset()
    train_classes = [label for _, label in train]
    portions = Counter(train_classes)
    total_sum = sum(portions.values())
    # portions = {labels[old_key]: value for old_key, value in portions.items()}
    portions = [[labels[key], str(value/400) + '%'] for key, value in portions.items()]
    print(tabulate(portions, ["Class", "New portion"]))
    # print(total_sum)
    # print(dict(Counter(train.targets)))
