"""CaptionDataset variant that reads captions inline from the JSON, no annotation files needed."""
from .caption_datasets import CaptionDataset
import json
import random


class InlineCaptionDataset(CaptionDataset):
    """
    Like CaptionDataset but each data item has {'image': ..., 'caption': ...}
    instead of {'image': ..., 'annotation': 'path/to/caption.json'}.
    """

    def __getitem__(self, idx):
        try:
            data_sample = self.data_list[idx]
            image = self._read_image(data_sample['image']).convert('RGB')
            data = self._process_image(image)
            del image
            caption = data_sample[self.cap_source]
            data.update(self._process_text(caption))
            data.update(image_dir=self.local_folder, image_file=data_sample['image'])
            return data
        except Exception as e:
            print(f"Error at idx={idx}: {e}", flush=True)
            return self._retry()
