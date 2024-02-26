from pathlib import Path


def setup_datasets(datasets_folder, dataset_name):
    dataset_folder = datasets_folder / dataset_name
    save_model_to = Path("../../../../models")

    dataset_metadata = creator.DatasetMetadata(name=dataset_name)
    return