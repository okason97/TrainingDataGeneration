import pyopenpose as op
import cv2
import numpy as np

params = dict()
params["model_folder"] = "/mnt/sda2/old_home/grios/openpose/models"
params["hand"] = True
params["hand_detector"] = 2
params["body"] = 0
params["hand_render"] = 0
params["hand_scale_number"] = 6
params["hand_scale_range"] = 0.4

opWrapper = op.WrapperPython()
opWrapper.configure(params)
opWrapper.start()

imageToProcess = cv2.imread("/mnt/sda2/datasets/rwth-small/train/0/train0.png")
print(np.array(imageToProcess).shape)

handRectangles = [
    # Left/Right hands person 0
    [
    op.Rectangle(0, 0, max(np.array(imageToProcess).shape), max(np.array(imageToProcess).shape)),
    op.Rectangle(0., 0., 0., 0.),
    ],
]

datum = op.Datum()
datum.cvInputData = imageToProcess
datum.handRectangles = handRectangles
opWrapper.emplaceAndPop(op.VectorDatum([datum]))
print(datum.handKeypoints[0])