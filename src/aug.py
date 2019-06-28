import os
import sys
import numpy as np
from PIL import Image, ImageEnhance
from patch_extractor import PatchExtractor

PATCH_SIZE = 512
stride = 256

def comb(imgs):

    min_shape = sorted( [(np.sum(i.size), i.size ) for i in imgs])[0][1]
    imgs_comb = np.hstack( (np.asarray( i.resize(min_shape) ) for i in imgs))
    return Image.fromarray( imgs_comb)

def aug(path,outdir,rotation=0,flip=0,enhance=0):

    with Image.open(path) as img:
        extractor = PatchExtractor(img=img, patch_size=PATCH_SIZE, stride=stride)
        patches = extractor.extract_patches()

        for i,patch in enumerate(patches):

            npatch = patch.copy()

            if rotation != 0:
                npatch = npatch.rotate(rotation * 90)

            if flip != 0:
                npatch = npatch.transpose(Image.FLIP_LEFT_RIGHT)

            if enhance != 0:
                factors = np.random.uniform(.5, 1.5, 3)
                npatch = ImageEnhance.Color(npatch).enhance(factors[0])
                npatch = ImageEnhance.Contrast(npatch).enhance(factors[1])
                npatch = ImageEnhance.Brightness(npatch).enhance(factors[2])

            new = comb([patch,npatch])
            new.save(os.path.join(outdir,f"patch_{i}.jpg"))

if __name__ == "__main__":
    
    path = sys.argv[0]
    outdir = sys.argv[1]
    aug(path,outdir,enhance=1)