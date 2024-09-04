import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


def get_dataset(DATA_TYPE):
    if DATA_TYPE == "BEDLAM_V2":
        from hmr4d.dataset.bedlam.bedlam import BedlamDatasetV2

        return BedlamDatasetV2()

    if DATA_TYPE == "3DPW_TRAIN":
        from hmr4d.dataset.threedpw.threedpw_motion_train import ThreedpwSmplDataset

        return ThreedpwSmplDataset()

if __name__ == "__main__":
    DATA_TYPE = "3DPW_TRAIN"
    dataset = get_dataset(DATA_TYPE)
    print(len(dataset))

    data = dataset[0]

    from hmr4d.datamodule.mocap_trainX_testY import collate_fn

    loader = DataLoader(
        dataset,
        shuffle=False,
        num_workers=0,
        persistent_workers=False,
        pin_memory=False,
        batch_size=1,
        collate_fn=collate_fn,
    )
    i = 0
    for batch in tqdm(loader):
        i += 1
        # if i == 20:
        #     raise AssertionError
        # time.sleep(0.2)
        pass
