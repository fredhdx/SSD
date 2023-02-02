import os
import shutil
import random

basedir = os.getcwd() + '/data'

sourcedir = basedir + '/train'
destdir = basedir + '/mytest'

images = ["{:05d}".format(_) for _ in random.sample(range(6392), 100)]

for img in images:
    shutil.copy(sourcedir + f"/images/{img}.jpg",
                destdir + f"/images/{img}.jpg")
    shutil.copy(sourcedir + f"/annotations/{img}.txt",
                destdir + f"/annotations/{img}.txt"
                )

print(f'generated {len(images)} images')
