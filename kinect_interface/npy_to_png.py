import numpy as np
import png

images = [
    {
        'from': 'D:/Werk/Coralie Vogelaar/sdk1/dev/data/depth_ortho_0_0.npy',
        'to': '../data/depth_scans/test/0000_00.npy'
    },
    {
        'from': 'D:/Werk/Coralie Vogelaar/sdk1/dev/data/depth_ortho_1_0.npy',
        'to': '../data/depth_scans/test/0000_01.npy'
    }
]

w = png.Writer(255, 1, greyscale=True)

for image in images:
    img = np.load(image['from'])
    np.save(image['to'], img)