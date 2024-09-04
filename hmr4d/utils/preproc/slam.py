import cv2
import time
import torch
from multiprocessing import Process, Queue

try:
    from dpvo.utils import Timer
    from dpvo.dpvo import DPVO
    from dpvo.config import cfg
except:
    pass


from hmr4d import PROJ_ROOT
from hmr4d.utils.geo.hmr_cam import estimate_focal_length


class SLAMModel(object):
    def __init__(self, video_path, width, height, intrinsics=None, stride=1, skip=0, buffer=2048, resize=0.5):
        """
        Args:
            intrinsics: [fx, fy, cx, cy]
        """
        if intrinsics is None:
            print("Estimating focal length")
            focal_length = estimate_focal_length(width, height)
            intrinsics = torch.tensor([focal_length, focal_length, width / 2.0, height / 2.0])
        else:
            intrinsics = intrinsics.clone()

        self.dpvo_cfg = str(PROJ_ROOT / "third-party/DPVO/config/default.yaml")
        self.dpvo_ckpt = "inputs/checkpoints/dpvo/dpvo.pth"

        self.buffer = buffer
        self.times = []
        self.slam = None
        self.queue = Queue(maxsize=8)
        self.reader = Process(target=video_stream, args=(self.queue, video_path, intrinsics, stride, skip, resize))
        self.reader.start()

    def track(self):
        (t, image, intrinsics) = self.queue.get()

        if t < 0:
            return False

        image = torch.from_numpy(image).permute(2, 0, 1).cuda()
        intrinsics = intrinsics.cuda()  # [fx, fy, cx, cy]

        if self.slam is None:
            cfg.merge_from_file(self.dpvo_cfg)
            cfg.BUFFER_SIZE = self.buffer
            self.slam = DPVO(cfg, self.dpvo_ckpt, ht=image.shape[1], wd=image.shape[2], viz=False)

        with Timer("SLAM", enabled=False):
            t = time.time()
            self.slam(t, image, intrinsics)
            self.times.append(time.time() - t)

        return True

    def process(self):
        for _ in range(12):
            self.slam.update()

        self.reader.join()
        return self.slam.terminate()[0]


def video_stream(queue, imagedir, intrinsics, stride, skip=0, resize=0.5):
    """video generator"""
    assert len(intrinsics) == 4, "intrinsics should be [fx, fy, cx, cy]"

    cap = cv2.VideoCapture(imagedir)
    t = 0
    for _ in range(skip):
        ret, image = cap.read()

    while True:
        # Capture frame-by-frame
        for _ in range(stride):
            ret, image = cap.read()
            # if frame is read correctly ret is True
            if not ret:
                break

        if not ret:
            break

        image = cv2.resize(image, None, fx=resize, fy=resize, interpolation=cv2.INTER_AREA)
        h, w, _ = image.shape
        image = image[: h - h % 16, : w - w % 16]

        intrinsics_ = intrinsics.clone() * resize
        queue.put((t, image, intrinsics_))

        t += 1

    queue.put((-1, image, intrinsics))  # -1 will terminate the process
    cap.release()

    # wait for the queue to be empty, otherwise the process will end immediately
    while not queue.empty():
        time.sleep(1)
