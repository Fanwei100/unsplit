import os

from ISR.models import RDN
import numpy as np
from PIL import Image
from tqdm import tqdm

def ProcesImages(folder="Images/Images_train_64/",outfolder="Images/Images_Train_ISR_256/"):
	os.makedirs(outfolder,exist_ok=True)
	rdn = RDN(weights='psnr-large')
	for i,f in enumerate(tqdm(os.listdir(folder))):
		if os.path.exists(outfolder+f): continue
		samll_img = np.array(Image.open(folder+f))
		lr_img = rdn.predict(rdn.predict(samll_img))
		Image.fromarray(lr_img).save(outfolder+f)


# ProcesImages()
ProcesImages("Images/Images_test_64/",outfolder="Images/Images_Test_ISR_256/")