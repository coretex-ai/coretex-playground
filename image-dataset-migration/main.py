from typing import Iterator
from contextlib import contextmanager

import os
import logging

from coretex import ComputerVisionDataset, ComputerVisionSample
from coretex.networking import networkManager


# URL of the source environment
SOURCE_BACKEND_URL = ""
# Credentials of account which has access to dataset
SOURCE_USERNAME = ""
SOURCE_PASSWORD = ""
# Dataset which will be transfered
SOURCE_DATASET_ID = 30852

# URL of the destination environment
DESTINATION_BACKEND_URL = ""
# Credentials of the account which will be used to create a transfered dataset
DESTINATION_USERNAME = ""
DESTINATION_PASSWORD = ""
# Project in which the transfered dataset will be created
DESTINATION_PROJECT_ID = 1


@contextmanager
def backend(url: str, username: str, password: str) -> Iterator[None]:
    try:
        # Save old url
        oldUrl = os.environ["CTX_API_URL"]

        # Set env var to new one
        os.environ["CTX_API_URL"] = url

        response = networkManager.authenticate(username, password, storeCredentials = False)
        if response.hasFailed():
            raise RuntimeError("Failed to authenticate")

        # There is nothing to yield
        yield None
    finally:
        # Return the url to the old one
        os.environ["CTX_API_URL"] = oldUrl


def copyAnnotation(source: ComputerVisionSample, destination: ComputerVisionSample, retryCount: int = 0) -> None:
    if retryCount >= 3:
        raise RuntimeError(f"Failed to copy annotation from '{source.id} - {source.name}' to '{destination.id} - {destination.name}'")

    sourceAnnotation = source.load().annotation
    if sourceAnnotation is not None:
        if not destination.saveAnnotation(sourceAnnotation):
            copyAnnotation(source, destination, retryCount + 1)


def copySample(source: ComputerVisionSample, destinationDatasetId: int, retryCount: int = 0) -> ComputerVisionSample:
    if retryCount >= 3:
        raise RuntimeError(f"Failed to create sample from \"{source.id} - {source.name}\" at path \"{source.imagePath}\"")

    source.unzip()

    destination = ComputerVisionSample.createComputerVisionSample(destinationDatasetId, source.imagePath)
    if destination is None:
        return copySample(source, destinationDatasetId, retryCount + 1)

    return destination


def main() -> None:
    with backend(SOURCE_BACKEND_URL, SOURCE_USERNAME, SOURCE_PASSWORD):
        sourceDataset = ComputerVisionDataset.fetchById(SOURCE_DATASET_ID)
        sourceDataset.download()

    with backend(DESTINATION_BACKEND_URL, DESTINATION_USERNAME, DESTINATION_PASSWORD):
        logging.info(f">> [Coretex] Creating destination dataset \"{sourceDataset.name}\"")

        destinationDataset = ComputerVisionDataset.createDataset(sourceDataset.name, DESTINATION_PROJECT_ID)
        destinationDataset.saveClasses(sourceDataset.classes)

        for index, sourceSample in enumerate(sourceDataset.samples):
            logging.info(f"\tCopying sample {index + 1}/{sourceDataset.count} - {sourceSample.name}")

            destinationSample = copySample(sourceSample, destinationDataset.id)
            copyAnnotation(sourceSample, destinationSample)


if __name__ == "__main__":
    main()
