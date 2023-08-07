from loaders.potsdam.potsdam import PotsdamDataset
from loaders.inria.inria import InriaDataset

def get_datasets(dataset: str, image_size: int):
    if dataset == "potsdam":
        training_dataset = PotsdamDataset(data_root="data/potsdam/train", mode="train", img_size=image_size)
        validation_dataset = PotsdamDataset(data_root="data/potsdam/test", mode="val", img_size=image_size)
        return training_dataset, validation_dataset
    elif dataset == "inria":
        training_dataset = InriaDataset(data_root="data/inria/train", mode="train", img_size=image_size)
        validation_dataset = InriaDataset(data_root="data/inria/val", mode="val", img_size=image_size)
        return training_dataset, validation_dataset

if __name__ == "__main__":
    train, val = get_datasets("inria", 256)

    img_dict = train[0]
    img, mask = img_dict["img"], img_dict["mask"]
    print(img.shape, mask.shape)
    print(img.dtype, mask.dtype)
    print(img.min(), img.max())
    print(mask.unique())

    img_dict = val[0]
    img, mask = img_dict["img"], img_dict["mask"]
    print(img.shape, mask.shape)
    print(img.dtype, mask.dtype)
    print(img.min(), img.max())
    print(mask.unique())