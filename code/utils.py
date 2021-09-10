import cv2
cv2.setNumThreads(0)
import glob
import sys
import collections
import time
import numpy as np
import random
import albumentations as A


def transformImg(image, seed=42, trans=False):
    # TODO: 这样设置随机种子是否合理？有没有更好的方法？
    # set a certain random seed to make the same transformation for all images for one person
    if trans == False:
        return image
    else:
        random.seed(seed)
        transform = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.OneOf([
                A.GaussNoise(p=0.3),
                A.Blur(blur_limit=3, p=0.2),
            ], p=0.5)
        ])

        aug_image = transform(image=image)["image"]

        return aug_image


def read_image(path,img_size=256):
    img = cv2.imread(path,0)
    width = img.shape[0]
    height = img.shape[1]
    # cropped = img[int(1/16*height):height-int(1/16*height), int(1/16*width):width-int(1/16*width)]#想着要不要把图片裁剪一下……
    return cv2.resize(cv2.imread(path, 0), (img_size, img_size))
    # return cv2.resize(cropped, (img_size, img_size))


def load_images(path, scan_id, split, img_size=256, length=64, trans=False):
    # path='/home/pei_group/jupyter/Wujingjing/data'
    flair = sorted(glob.glob(f"{path}/{split}/{scan_id}/FLAIR/*.png"))
    t1w = sorted(glob.glob(f"{path}/{split}/{scan_id}/T1w/*.png"))
    t1wce = sorted(glob.glob(f"{path}/{split}/{scan_id}/T1wCE/*.png"))
    t2w = sorted(glob.glob(f"{path}/{split}/{scan_id}/T2w/*.png"))

    seed = random.randint(1, 10000)

    flair_img=np.array([transformImg(read_image(a,img_size), seed, trans=trans) for a in flair[len(flair)//2-(int)(length/2):len(flair)//2+(int)(length/2)]]).T
    if len(flair_img) == 0:
        flair_img = np.zeros((img_size, img_size, length))
    elif flair_img.shape[-1] < length:
        flair_img = np.concatenate((flair_img, np.zeros((img_size, img_size, length-flair_img.shape[-1]))), axis=-1)

    t1w_img = np.array([transformImg(read_image(a,img_size), seed, trans=trans) for a in t1w[len(t1w) // 2 - (int)(length/2):len(t1w) // 2 + (int)(length/2)]]).T
    if len(t1w_img) == 0:
        t1w_img = np.zeros((img_size, img_size, length))
    elif t1w_img.shape[-1] < length:
        n=length - t1w_img.shape[-1]
        t1w_img = np.concatenate((t1w_img, np.zeros((img_size, img_size, n))), axis=-1)

    t1wce_img = np.array([transformImg(read_image(a, img_size), seed, trans=trans) for a in t1wce[len(t1wce) // 2 - (int)(length/2):len(t1wce) // 2 +(int)(length/2)]]).T
    if len(t1wce_img) == 0:
        t1wce_img = np.zeros((img_size, img_size, length))
    elif t1wce_img.shape[-1] < length:
        t1wce_img = np.concatenate((t1wce_img, np.zeros((img_size, img_size, length - t1wce_img.shape[-1]))), axis=-1)

    t2w_img = np.array([transformImg(read_image(a,img_size), seed, trans=trans) for a in t2w[len(t2w) // 2 - (int)(length/2):len(t2w) // 2 + (int)(length/2)]]).T
    if len(t2w_img) == 0:
        t2w_img = np.zeros((img_size, img_size,length))
    elif t2w_img.shape[-1] < length:
        t2w_img = np.concatenate((t2w_img, np.zeros((img_size, img_size, length - t2w_img.shape[-1]))), axis=-1)

    return np.array((flair_img, t1w_img, t1wce_img, t2w_img))


def one_hot(arr):
    return [[1, 0] if a_i == 0 else [0, 1] for a_i in arr]


class Progbar(object):
    """Displays a progress bar.
    # Arguments
        target: Total number of steps expected, None if unknown.
        width: Progress bar width on screen.
        verbose: Verbosity mode, 0 (silent), 1 (verbose), 2 (semi-verbose)
        stateful_metrics: Iterable of string names of metrics that
            should *not* be averaged over time. Metrics in this list
            will be displayed as-is. All others will be averaged
            by the progbar before display.
        interval: Minimum visual progress update interval (in seconds).
    """

    def __init__(self, target, width=30, verbose=1, interval=0.05,
                 stateful_metrics=None):
        self.target = target
        self.width = width
        self.verbose = verbose
        self.interval = interval
        if stateful_metrics:
            self.stateful_metrics = set(stateful_metrics)
        else:
            self.stateful_metrics = set()

        self._dynamic_display = ((hasattr(sys.stdout, 'isatty') and
                                  sys.stdout.isatty()) or
                                 'ipykernel' in sys.modules)
        self._total_width = 0
        self._seen_so_far = 0
        self._values = collections.OrderedDict()
        self._start = time.time()
        self._last_update = 0

    def update(self, current, values=None):
        """Updates the progress bar.
        # Arguments
            current: Index of current step.
            values: List of tuples:
                `(name, value_for_last_step)`.
                If `name` is in `stateful_metrics`,
                `value_for_last_step` will be displayed as-is.
                Else, an average of the metric over time will be displayed.
        """
        values = values or []
        for k, v in values:
            if k not in self.stateful_metrics:
                if k not in self._values:
                    self._values[k] = [v * (current - self._seen_so_far),
                                       current - self._seen_so_far]
                else:
                    self._values[k][0] += v * (current - self._seen_so_far)
                    self._values[k][1] += (current - self._seen_so_far)
            else:
                self._values[k] = v
        self._seen_so_far = current

        now = time.time()
        info = ' - %.0fs' % (now - self._start)
        if self.verbose == 1:
            if (now - self._last_update < self.interval and
                    self.target is not None and current < self.target):
                return

            prev_total_width = self._total_width
            if self._dynamic_display:
                sys.stdout.write('\b' * prev_total_width)
                sys.stdout.write('\r')
            else:
                sys.stdout.write('\n')

            if self.target is not None:
                numdigits = int(np.floor(np.log10(self.target))) + 1
                barstr = '%%%dd/%d [' % (numdigits, self.target)
                bar = barstr % current
                prog = float(current) / self.target
                prog_width = int(self.width * prog)
                if prog_width > 0:
                    bar += ('=' * (prog_width - 1))
                    if current < self.target:
                        bar += '>'
                    else:
                        bar += '='
                bar += ('.' * (self.width - prog_width))
                bar += ']'
            else:
                bar = '%7d/Unknown' % current

            self._total_width = len(bar)
            sys.stdout.write(bar)

            if current:
                time_per_unit = (now - self._start) / current
            else:
                time_per_unit = 0
            if self.target is not None and current < self.target:
                eta = time_per_unit * (self.target - current)
                if eta > 3600:
                    eta_format = '%d:%02d:%02d' % (eta // 3600, (eta % 3600) // 60, eta % 60)
                elif eta > 60:
                    eta_format = '%d:%02d' % (eta // 60, eta % 60)
                else:
                    eta_format = '%ds' % eta

                info = ' - ETA: %s' % eta_format
            else:
                if time_per_unit >= 1:
                    info += ' %.0fs/step' % time_per_unit
                elif time_per_unit >= 1e-3:
                    info += ' %.0fms/step' % (time_per_unit * 1e3)
                else:
                    info += ' %.0fus/step' % (time_per_unit * 1e6)

            for k in self._values:
                info += ' - %s:' % k
                if isinstance(self._values[k], list):
                    avg = np.mean(
                        self._values[k][0] / max(1, self._values[k][1]))
                    if abs(avg) > 1e-3:
                        info += ' %.4f' % avg
                    else:
                        info += ' %.4e' % avg
                else:
                    info += ' %s' % self._values[k]

            self._total_width += len(info)
            if prev_total_width > self._total_width:
                info += (' ' * (prev_total_width - self._total_width))

            if self.target is not None and current >= self.target:
                info += '\n'

            sys.stdout.write(info)
            sys.stdout.flush()

        elif self.verbose == 2:
            if self.target is None or current >= self.target:
                for k in self._values:
                    info += ' - %s:' % k
                    avg = np.mean(
                        self._values[k][0] / max(1, self._values[k][1]))
                    if avg > 1e-3:
                        info += ' %.4f' % avg
                    else:
                        info += ' %.4e' % avg
                info += '\n'

                sys.stdout.write(info)
                sys.stdout.flush()

        self._last_update = now

    def add(self, n, values=None):
        self.update(self._seen_so_far + n, values)