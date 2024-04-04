import random
import logging

from coretex import ImageDataset, ImageSample, ImageDatasetClasses
from coretex.networking import networkManager


USERNAME = ""  # Coretex username
PASSWORD = ""  # Coretex password
PROJECT_ID = 1  # ID of the Project where the train/valid Datasets will be created
DATASET_ID = 1  # ID of the Dataset which will be split into train/valid parts
VALID_PCT  = 0.2  # % amount in range of 0-1 of how many Samples will be placed in validation Dataset


def createDataset(name: str, projectId: int, samples: list[ImageSample], classes: ImageDatasetClasses) -> None:
    logging.info(f">> [Coretex] Creating dataset \"{name}\"")

    dataset = ImageDataset.createDataset(name, projectId)
    if dataset is None:
        raise RuntimeError(f"Failed to create dataset \"{name}\"")

    if not dataset.saveClasses(classes):
        raise RuntimeError(f"Failed to create classes for dataset \"{name}\"")

    for sample in samples:
        logging.info(f">> [Coretex] Copying sample \"{sample.name}\"")

        sample.unzip()

        try:
            copy = dataset.add(sample.imagePath)
        except BaseException as ex:
            logging.info(f"\tFailed to copy sample \"{sample.name}\" - {ex}")
            continue

        annotation = sample.load().annotation
        if annotation is not None:
            if not copy.saveAnnotation(annotation):
                logging.info("\tFailed to copy sample annotation, deleting...")

                if not copy.delete():
                    logging.info(f"\tFailed to delete sample")


def main() -> None:
    random.seed(42)

    response = networkManager.authenticate(USERNAME, PASSWORD, storeCredentials = False)
    if response.hasFailed():
        raise RuntimeError("Failed to authenticate")

    dataset = ImageDataset.fetchById(DATASET_ID)
    dataset.download()

    # Shuffle before splitting
    random.shuffle(dataset.samples)

    validCount = int(dataset.count * VALID_PCT)
    trainCount = dataset.count - validCount

    trainSamples = dataset.samples[:trainCount]
    validSamples = dataset.samples[trainCount:]

    createDataset(f"{dataset.name}-train", PROJECT_ID, trainSamples, dataset.classes)
    createDataset(f"{dataset.name}-valid", PROJECT_ID, validSamples, dataset.classes)


if __name__ == "__main__":
    main()
