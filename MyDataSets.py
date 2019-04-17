import torchvision.datasets as datasets
import os

# 定义自己的数据集


class MyDatasets(datasets.ImageFolder):
    def __init__(self, root_dir, transform=None):
        super(MyDatasets, self).__init__(root_dir, transform=transform)
        self.myextensions = ['.jpg', '.jpeg',
                             '.png', '.ppm', '.bmp', '.pgm', '.tif']
        self.samples = self.make_imgdatasets(
            root_dir, self.class_to_idx, self.myextensions)

    def has_file_allowed_extension(self, filename, extensions):
        filename_lower = filename.lower()
        return any(filename_lower.endswith(ext) for ext in extensions)

    def make_imgdatasets(self, dir, class_to_idx, extensions):
        images = []
        dir = os.path.expanduser(dir)
        for target in sorted(os.listdir(dir)):
            d = os.path.join(dir, target)
            if not os.path.isdir(d):
                continue

            for root, _, fnames in sorted(os.walk(d)):
                flag = len(os.listdir(root))
                if flag > 5:
                    flag = 5
                for fname in sorted(fnames[:flag]):
                    if self.has_file_allowed_extension(fname, extensions):
                        path = os.path.join(root, fname)
                        item = (path, class_to_idx[target])
                        images.append(item)
        return images
