import os,torch,numpy as np
import shutil
import sys,PIL

import pandas as pd

if os.getcwd().endswith("evaluation"):sys.path.append("../")

from tqdm import tqdm
from torchvision import  models,transforms
from PIL import Image
from torch.utils.data import Dataset
from DataSet import getDataAndModels
from torchvision.utils import save_image

class CustomDataset(Dataset):
	def __init__(self, root, transform=None):
		self.root = root
		self.transform,self.images,self.labels = transform,[],[]
		self._load_images()

	def _load_images(self):
		image_files = os.listdir(self.root)
		for file_name in image_files:
			file_path = os.path.join(self.root, file_name)
			# Extract label from the image file name
			label = int(file_name.split('_')[1])  # Assumes the label is before the extension
			self.images.append(file_path)
			self.labels.append(label)

	def __getitem__(self, index):
		image_path = self.images[index]
		label = self.labels[index]
		# Load image
		image = Image.open(image_path).convert("RGB")
		# Apply transformations
		if self.transform is not None:
			image = self.transform(image)
		return image, label

	def __len__(self):
		return len(self.images)

def SaveImages(outdir="Images/"):
	os.makedirs(outdir,exist_ok=True)
	trainset, testset, clientModel, serverModel, _ = getDataAndModels("food", model="Vgg8Net", datasize= (256,256))
	transform=transforms.Compose([transforms.Resize((256,256)),transforms.Resize((64,64)),transforms.ToTensor()])
	trainset.transform, testset.transform = transform, transform
	folder=f'{outdir}/Images_test_64/'
	os.makedirs(folder,exist_ok=True)
	for i,(d,l) in enumerate(testset):
		save_image(d,f'{folder}/Image_Test_{l}_{i}_.png')
	folder=f'{outdir}/Images_train_64/'
	os.makedirs(folder,exist_ok=True)
	for i,(d,l) in enumerate(trainset):
		save_image(d,f'{folder}/Image_{l}_{i}_.png')


def CheckSavedImageAccuracy(ImagePath="Images/Images_test_64/",batch_size=32):
	model = torch.load("Models/Size_256_256/efficientnet_v2_s_food_134_1.000_0.894.pt")
	device=torch.device("cuda")
	transform = transforms.Compose([
		transforms.Resize((256, 256)),  # Resize the images to a specific size
		transforms.ToTensor(),  # Convert the images to tensors
	])
	dataset=CustomDataset(ImagePath,transform)
	loader = torch.utils.data.DataLoader(dataset, shuffle=True, batch_size=batch_size)
	with torch.no_grad():
		ytrue, ypred = [], []
		for data, label in tqdm(loader):
			ypred.append(model(data.to(device)).cpu())
			ytrue.append(label)
		ytrue, ypred = torch.cat(ytrue).numpy(), torch.argmax(torch.cat(ypred), dim=1).numpy()
	print(ImagePath.split("/")[-2]," have",np.sum(ytrue == ypred) / ytrue.shape[0] ,"for ",ytrue.shape[0],"instances")

def calculateEfficientPredicton(Results="../Results/Size_64_64/CSVFiles/Losses.csv",modelname="efficientnet_v2_s"):
	model = torch.load("Models/Size_256_256/"+modelname+"_food_134_1.000_0.894.pt",map_location=torch.device('cpu'))
	df=pd.read_csv(Results)
	df["TrainFolder"] = df["Train"].replace("No", "0").replace("0", "WithoutTraining").replace("30", "WithTraining_30").replace("60", "WithTraining_60")
	imagebasepath="/".join(Results.split("/")[:-2])
	transform=transforms.Compose([transforms.Resize((64, 64)), transforms.ToTensor()])
	predlist=[]
	for i,row in tqdm(df.iterrows()):
		imgpath=f"{imagebasepath}/{row['TrainFolder']}/{row['Model']}_{row['Dataset']}_{row['LossName']}_{row['SplitLayer']}_{row['Class']}_{row['Index']}_{row['SEED']}.png"
		if not os.path.exists(imgpath):
			predlist.append(None)
		else:
			image = PIL.Image.open(imgpath).convert("RGB")
			image = torch.unsqueeze(transform(image),dim=0)
			predlist.append(torch.argmax(model(image)[0]).numpy())
	df[modelname]=np.array(predlist)
	del df["TrainFolder"]
	df.to_csv("/".join(Results.split("/")[:-1])+"/Prediction.csv",index=None)


def getAccuacyForGroups(Results="../Results/Size_64_64/CSVFiles/Prediction.csv"):
	df=pd.read_csv(Results)
	df["Train"]=df["Train"].replace("No","0")
	df=df[df["Dataset"]=="food"]
	del df["Dataset"],df["Model"]
	df.dropna(inplace=True)
	print(df.columns)
	results=[]
	gcolumns=["Train", "LossName", "SplitLayer"]
	for d, dfg in df.groupby(gcolumns):
		results.append([*d,np.sum(dfg["PredIndex"] == dfg["efficientnet_v2_s"]) / dfg.shape[0]])
	outpath="/".join(Results.split("/")[:-1])+"/Accuracy.csv"
	pd.DataFrame(data=results,columns=gcolumns+["Accuracy"]).to_csv(outpath,index=None)


# CheckSavedImageAccuracy("Images/Images_train_64/")
# CheckSavedImageAccuracy("Images/Images_test_64/")
# CheckSavedImageAccuracy("Images/Images_Train_256/")
# CheckSavedImageAccuracy("Images/Images_test_256/")
# CheckSavedImageAccuracy("Images/Images_Test_L_256/")
# CheckSavedImageAccuracy("Images/Images_Train_ISR_256/")
# CheckSavedImageAccuracy("Images/Images_Test_ISR_256/")
# CheckSavedImageAccuracy("/media/parmpal/Data/python/upwork/Yvette_Zoran/FederatadLearning/SplitLearning/SuperResolution/HAT/results/HAT_SRx4_ImageNet-pretrain_archived_20230704_145918/visualization/Set5/")
# SaveImages()
calculateEfficientPredicton()
getAccuacyForGroups()