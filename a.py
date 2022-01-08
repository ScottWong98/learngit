import matplotlib.pyplot as plt
import numpy as np



from sklearn.model_selection import train_test_split
from torch import LongTensor, Tensor
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import TensorDataset



import os
from typing import Dict, List, Tuple


class ImageDataset:
    def __init__(self, dir_name: str, batch_size: int = 16) -> None:
        super().__init__()
        self.dir_name = dir_name
        self.batch_size = batch_size

    def _load_face_data(self) -> Dict[str, List[str]]:
        face_filename_dict: Dict[str, List[str]] = {}
        for sub_dir in os.listdir(self.dir_name):
            face_filename_dict[sub_dir] = [os.path.join(self.dir_name, sub_dir, filename)
                for filename in os.listdir(os.path.join(self.dir_name, sub_dir))]
        return face_filename_dict

    def _load_emotion_data(self) -> Dict[str, List[str]]:
        emotion_filename_dict: Dict[str, List[str]] = {}
        for sub_dir in os.listdir(self.dir_name):
            for filename in os.listdir(os.path.join(self.dir_name, sub_dir)):
                emotion = filename.split("_")[2]
                full_filename = os.path.join(self.dir_name, sub_dir, filename)
                if emotion not in emotion_filename_dict:
                    emotion_filename_dict[emotion] = [full_filename]
                else:
                    emotion_filename_dict[emotion].append(full_filename)
        return emotion_filename_dict

    def load_data(self, target: str = "face", test_size: float = 0.3) -> Tuple[DataLoader, DataLoader]:
        assert target in ["face", "emotion"], f"Please use valid target name..."
        if target == "face":
            target_filename_dict = self._load_face_data()
        else:
            target_filename_dict = self._load_emotion_data()
        cnt = 0
        print(len(target_filename_dict))
        train_images, train_labels, test_images, test_labels = [], [], [], []
        for target, filename_list in target_filename_dict.items():
            tmp_images, tmp_labels = [], []
            for filename in filename_list:
                im = self._read_img(filename)
                tmp_images.append(im)
                tmp_labels.append(cnt)
            cnt += 1
            tmp_images, tmp_labels = np.array(tmp_images), np.array(tmp_labels)
            train_ids, test_ids = train_test_split(
                range(len(tmp_images)), test_size=test_size, random_state=12
            )
            train_images += tmp_images[train_ids].tolist()
            train_labels += tmp_labels[train_ids].tolist()
            test_images += tmp_images[test_ids].tolist()
            test_labels += tmp_labels[test_ids].tolist()

        return self._to_dataloader(train_images, train_labels, test_images, test_labels)

    def _to_dataloader(self, train_images, train_labels, test_images, test_labels) -> Tuple[DataLoader, DataLoader]:
        return (
            DataLoader(
                TensorDataset(Tensor(train_images), LongTensor(train_labels)),
                batch_size=self.batch_size,
                shuffle=False,
            ),
            DataLoader(
                TensorDataset(Tensor(test_images), LongTensor(test_labels)),
                batch_size=self.batch_size,
            ),
        )

    @classmethod
    def _read_img(cls, filename: str) -> np.ndarray:
        """ read image from .pgm file """
        with open(filename, "rb") as pgmf:
            im = plt.imread(pgmf) / 255.0
        return np.expand_dims(im, axis=0)
