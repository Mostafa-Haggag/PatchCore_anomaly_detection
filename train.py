import os
import glob
import shutil

import numpy as np
from PIL import Image
import cv2

from sklearn.random_projection import SparseRandomProjection
from sklearn.metrics import roc_auc_score
from sampling_methods.kcenter_greedy import kCenterGreedy
from scipy.ndimage import gaussian_filter
import torch
from torch.nn import functional as F
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
import faiss


def distance_matrix(x, y=None, p=2):  # pairwise distance of vectors
    y = x if type(y) == type(None) else y

    n = x.size(0)
    m = y.size(0)
    d = x.size(1)

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    dist = torch.pow(x - y, p).sum(2)
    return dist


class NN():
    def __init__(self, X=None, Y=None, p=2):
        self.p = p
        self.train(X, Y)

    def train(self, X, Y):
        self.train_pts = X
        self.train_label = Y

    def __call__(self, x):
        return self.predict(x)

    def predict(self, x):
        if type(self.train_pts) == type(None) or type(self.train_label) == type(None):
            name = self.__class__.__name__
            raise RuntimeError(f"{name} wasn't trained. Need to execute {name}.train() first")

        dist = distance_matrix(x, self.train_pts, self.p) ** (1 / self.p)
        labels = torch.argmin(dist, dim=1)
        return self.train_label[labels]

class KNN(NN):
    def __init__(self, X=None, Y=None, k=3, p=2):
        self.k = k
        super().__init__(X, Y, p)

    def train(self, X, Y):
        super().train(X, Y)
        if type(Y) != type(None):
            self.unique_labels = self.train_label.unique()

    def predict(self, x):
        dist = torch.cdist(x, self.train_pts, self.p)
        knn = dist.topk(self.k, largest=False)
        return knn

def prep_dirs(root):

    # you retun path of embeeding, sample and the src
    # the root is the logger path
    # make embeddings dir
    # you create path of embedding
    embeddings_path = os.path.join('./', 'embeddings', args.category)
    # you create a folders inside the embedding
    os.makedirs(embeddings_path, exist_ok=True)
    # what is the root dir ???
    # you save inside the root path a folder called sample
    # make sample dir
    sample_path = os.path.join(root, 'sample')
    os.makedirs(sample_path, exist_ok=True)
    # make source code record dir & copy
    # you have somethign calleld source
    source_code_save_path = os.path.join(root, 'src')
    os.makedirs(source_code_save_path, exist_ok=True)
    return embeddings_path, sample_path, source_code_save_path

def embedding_concat(x, y):
    # from https://github.com/xiahaifeng1995/PaDiM-Anomaly-Detection-Localization-master
    # This function is designed to concatenate feature maps from two different layers
    # (or embeddings), x and y,
    # while ensuring that the spatial sizes of the embeddings are compatible.
    # It reshapes
    # and concatenates the embeddings in a way that can be useful for
    # tasks like anomaly detection.
    B, C1, H1, W1 = x.size()
    _, C2, H2, W2 = y.size()
    # The function assumes that H1 is an integer multiple of H2,
    # and similarly for the widths (W1 and W2).
    # This ensures that the feature maps can be processed and concatenated correctly.
    s = int(H1 / H2)
    # s = int(H1 / H2) calculates the scale factor between the heights (and widths) of x and y.
    # This factor, s, represents how much larger the dimensions of x are compared to y.
    x = F.unfold(x, kernel_size=s, dilation=1, stride=s)
    # The unfold operation converts the feature map x into a series of non-overlapping
    # patches (each of size s).
    # This transforms x from a 4D tensor of shape (B, C1, H1, W1)
    # to a 3D tensor with shape (B, C1, num_patches), where num_patches depends on H2 and W2.
    x = x.view(B, C1, -1, H2, W2)
    # x = x.view(B, C1, -1, H2, W2): Reshape x so that it is back into a 4D
    # tensor but with reduced spatial dimensions (H2, W2) that match the size of y.
    # The third dimension -1 corresponds to the number of patches extracted from x.
    z = torch.zeros(B, C1 + C2, x.size(2), H2, W2)
    # z = torch.zeros(B, C1 + C2, x.size(2), H2, W2):
    # Create an empty tensor z to store the concatenated feature maps
    # from x and y. The number of channels in z will be C1 + C2 to
    # accommodate both feature maps.


    # The loop for i in range(x.size(2))
    # iterates over the patches in x,
    # and torch.cat((x[:, :, i, :, :], y), 1)
    # concatenates the feature maps x and y along the channel dimension (1st dimension).
    # z[:, :, i, :, :] = torch.cat(...):
    # Stores the concatenated result for each patch into the corresponding position in z.
    for i in range(x.size(2)):
        z[:, :, i, :, :] = torch.cat((x[:, :, i, :, :], y), 1)
    # z = z.view(B, -1, H2 * W2):
    # Reshape z to be compatible for folding back into the original spatial dimensions.
    z = z.view(B, -1, H2 * W2)
    # z = F.fold(z, kernel_size=s, output_size=(H1, W1), stride=s):
    # The fold operation reverses the unfold operation,
    # reassembling the patches into a full feature map with spatial dimensions (H1, W1).
    z = F.fold(z, kernel_size=s, output_size=(H1, W1), stride=s)
    # z: A tensor of shape (B, C1 + C2, H1, W1),
    # where the feature maps from x and y have been concatenated and reassembled.
    return z

def reshape_embedding(embedding):
    embedding_list = []
    for k in range(embedding.shape[0]):
        for i in range(embedding.shape[2]):
            for j in range(embedding.shape[3]):
                embedding_list.append(embedding[k, :, i, j])
    return embedding_list

#imagenet
# similar to everything we did before
mean_train = [0.485, 0.456, 0.406]
std_train = [0.229, 0.224, 0.225]

class MVTecDataset(Dataset):
    def __init__(self, root, transform, gt_transform, phase):
        # typical way to with with mvtec dataset
        if phase=='train':
            self.img_path = os.path.join(root, 'train')
        else:
            self.img_path = os.path.join(root, 'test')
            self.gt_path = os.path.join(root, 'ground_truth')
        self.transform = transform
        self.gt_transform = gt_transform
        # load dataset
        # this is very intersting
        self.img_paths, self.gt_paths, self.labels, self.types = self.load_dataset() # self.labels => good : 0, anomaly : 1

    def load_dataset(self):

        img_tot_paths = []
        gt_tot_paths = []
        tot_labels = []
        tot_types = []

        defect_types = os.listdir(self.img_path)
        # you are listing everythign in this directory
        
        for defect_type in defect_types:
            if defect_type == 'good':
                img_paths = glob.glob(os.path.join(self.img_path, defect_type) + "/*.png")
                img_paths.sort()
                img_tot_paths.extend(img_paths)
                # list filled with 0 repeated to number of image paths
                gt_tot_paths.extend([0]*len(img_paths))
                # list filled with 0 repeated to number of image paths
                tot_labels.extend([0]*len(img_paths))
                # list filled with good repeated to number of image paths
                tot_types.extend(['good']*len(img_paths))
            else:
                img_paths = glob.glob(os.path.join(self.img_path, defect_type) + "/*.png")
                # this is not needed if you have no ground truths
                gt_paths = glob.glob(os.path.join(self.gt_path, defect_type) + "/*.png")
                img_paths.sort()
                gt_paths.sort()
                img_tot_paths.extend(img_paths)
                gt_tot_paths.extend(gt_paths)
                # you are putting alot of 1s in hrere
                tot_labels.extend([1]*len(img_paths))
                # you have different types of defects in here
                tot_types.extend([defect_type]*len(img_paths))

        assert len(img_tot_paths) == len(gt_tot_paths), "Something wrong with test and ground truth pair!"
        
        return img_tot_paths, gt_tot_paths, tot_labels, tot_types

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path, gt, label, img_type = self.img_paths[idx], self.gt_paths[idx], self.labels[idx], self.types[idx]
        # takes teh image as rgb
        img = Image.open(img_path).convert('RGB')
        # apply the transformation
        img = self.transform(img)
        if gt == 0:
            # createa fake mask
            gt = torch.zeros([1, img.size()[-2], img.size()[-2]])
        else:
            gt = Image.open(gt)
            gt = self.gt_transform(gt)
        
        assert img.size()[1:] == gt.size()[1:], "image.size != gt.size !!!"
        # return to me the basnemd missing the .png thigns
        # this is the idea
        return img, gt, label, os.path.basename(img_path[:-4]), img_type

def cvt2heatmap(gray):
    # What does that do exactly ???
    # This function converts a grayscale image into a heatmap using
    # OpenCV's applyColorMap function.
    # gray: A grayscale image, typically a 2D array of pixel intensity values.
    # you must first
    # Converts the input grayscale image to an 8-bit unsigned integer type (uint8),
    # ensuring that pixel values
    # are within the 0-255 range. This is necessary because applyColorMap expects an 8-bit image.
    heatmap = cv2.applyColorMap(np.uint8(gray), cv2.COLORMAP_JET)
    #  This function applies a color map (in this case, COLORMAP_JET)
    #  to the grayscale image. A color map assigns
    #  different colors to different intensity levels,
    #  effectively creating a "heatmap" where lower intensities might appear
    #  blue, mid-range intensities green, and high intensities red.
    #  A 3-channel (RGB) heatmap image where colors represent the intensity levels of the original grayscale image.
    return heatmap

def heatmap_on_image(heatmap, image):
    # i donot like this function so much but we can improve for abit
    # This function overlays a heatmap on an original image,
    # blending the two together. The detailed explanation:
    if heatmap.shape != image.shape:
        # to avoid error
        heatmap = cv2.resize(heatmap, (image.shape[0], image.shape[1]))
    out = np.float32(heatmap)/255 + np.float32(image)/255
    # This step ensures that blending the two images gives proportional weight to both.
    # Adds the two normalized images together, pixel by pixel.
    out = out / np.max(out)
    #  Normalizes the result to ensure that the maximum value across the image is 1.
    #  This is done to prevent any pixel value from exceeding 1 after the addition.
    return np.uint8(255 * out)

def min_max_norm(image):
    # like this you make sure that everything is between 0 to 1
    # This function performs min-max normalization, ensuring that all pixel values
    # in the input image are scaled between 0 and 1. The detailed steps:
    a_min, a_max = image.min(), image.max()
    return (image-a_min)/(a_max - a_min)    
    

class PatchCore(pl.LightningModule):
    def __init__(self, hparams):
        super(PatchCore, self).__init__()

        self.save_hyperparameters(hparams)
        # this is part of lighting module can be usinged
        # to save hyper paramters

        self.init_features()
        def hook_t(module, input, output):
            self.features.append(output)

        self.model = torch.hub.load('pytorch/vision:v0.9.0', 'wide_resnet50_2', pretrained=True)

        for param in self.model.parameters():
            # making sure that you are not optimizing anything at all in here
            param.requires_grad = False

        self.model.layer2[-1].register_forward_hook(hook_t)
        self.model.layer3[-1].register_forward_hook(hook_t)

        # mse loss in here
        self.criterion = torch.nn.MSELoss(reduction='sum')

        self.init_results_list()
        # a group of  lists that needs to be defiened
        # you use to put the results
        # you load with 256
        # you go down to 224 with center crop
        # you apply normal mean and std
        # plreace Image.ANTIALIAS with Image.LANCZOS
        self.data_transforms = transforms.Compose([
                        transforms.Resize((args.load_size, args.load_size), Image.LANCZOS),
                        transforms.ToTensor(),
                        transforms.CenterCrop(args.input_size),
                        transforms.Normalize(mean=mean_train,
                                            std=std_train)])
        self.gt_transforms = transforms.Compose([
                        transforms.Resize((args.load_size, args.load_size)),
                        transforms.ToTensor(),
                        transforms.CenterCrop(args.input_size)])
        # this is denormalization method to remove the fucking normalization
        # if you want to view the iamge
        self.inv_normalize = transforms.Normalize(mean=[-0.485/0.229, -0.456/0.224, -0.406/0.255], std=[1/0.229, 1/0.224, 1/0.255])

    def init_results_list(self):
        self.gt_list_px_lvl = []
        self.pred_list_px_lvl = []
        self.gt_list_img_lvl = []
        self.pred_list_img_lvl = []
        self.img_path_list = []        

    def init_features(self):
        # being called to create an empty lsit that getts filled with the features
        # every forwards pass you call it
        # called when you intizailize the class
        self.features = []

    def forward(self, x_t):
        self.init_features()
        _ = self.model(x_t)
        return self.features

    def save_anomaly_map(self, anomaly_map, input_img, gt_img, file_name, x_type):
        if anomaly_map.shape != input_img.shape:
            anomaly_map = cv2.resize(anomaly_map, (input_img.shape[0], input_img.shape[1]))
        anomaly_map_norm = min_max_norm(anomaly_map)
        anomaly_map_norm_hm = cvt2heatmap(anomaly_map_norm*255)

        # anomaly map on image
        heatmap = cvt2heatmap(anomaly_map_norm*255)
        hm_on_img = heatmap_on_image(heatmap, input_img)

        # save images
        cv2.imwrite(os.path.join(self.sample_path, f'{x_type}_{file_name}.jpg'), input_img)
        cv2.imwrite(os.path.join(self.sample_path, f'{x_type}_{file_name}_amap.jpg'), anomaly_map_norm_hm)
        cv2.imwrite(os.path.join(self.sample_path, f'{x_type}_{file_name}_amap_on_img.jpg'), hm_on_img)
        cv2.imwrite(os.path.join(self.sample_path, f'{x_type}_{file_name}_gt.jpg'), gt_img)

    def train_dataloader(self):
        image_datasets = MVTecDataset(root=os.path.join(args.dataset_path,args.category), transform=self.data_transforms, gt_transform=self.gt_transforms, phase='train')
        train_loader = DataLoader(image_datasets, batch_size=args.batch_size, shuffle=True, num_workers=0)
        return train_loader

    def test_dataloader(self):
        test_datasets = MVTecDataset(root=os.path.join(args.dataset_path,args.category), transform=self.data_transforms, gt_transform=self.gt_transforms, phase='test')
        test_loader = DataLoader(test_datasets, batch_size=1, shuffle=False, num_workers=0)
        return test_loader

    def configure_optimizers(self):
        return None

    def on_train_start(self):
        # these are dir to save everthing
        self.model.eval() # to stop running_var move (maybe not critical)        
        self.embedding_dir_path, self.sample_path, self.source_code_save_path = prep_dirs(self.logger.log_dir)
        self.embedding_list = []
    
    def on_test_start(self):
        # these are the paths to save everyting
        self.embedding_dir_path, self.sample_path, self.source_code_save_path = prep_dirs(self.logger.log_dir)
        self.index = faiss.read_index(os.path.join(self.embedding_dir_path,'index.faiss'))
        if torch.cuda.is_available():
            res = faiss.StandardGpuResources()
            self.index = faiss.index_cpu_to_gpu(res, 0 ,self.index)
        self.init_results_list()
        
    def training_step(self, batch, batch_idx): # save locally aware patch features
        x, _, _, _, _ = batch
        features = self(x)
        embeddings = []
        for feature in features:
            m = torch.nn.AvgPool2d(3, 1, 1)
            embeddings.append(m(feature))
        embedding = embedding_concat(embeddings[0], embeddings[1])
        self.embedding_list.extend(reshape_embedding(np.array(embedding)))

    def training_epoch_end(self, outputs): 
        total_embeddings = np.array(self.embedding_list)
        # Random projection
        self.randomprojector = SparseRandomProjection(n_components='auto', eps=0.9) # 'auto' => Johnson-Lindenstrauss lemma
        self.randomprojector.fit(total_embeddings)
        # Coreset Subsampling
        selector = kCenterGreedy(total_embeddings,0,0)
        selected_idx = selector.select_batch(model=self.randomprojector, already_selected=[], N=int(total_embeddings.shape[0]*args.coreset_sampling_ratio))
        self.embedding_coreset = total_embeddings[selected_idx]
        
        print('initial embedding size : ', total_embeddings.shape)
        print('final embedding size : ', self.embedding_coreset.shape)
        #faiss
        self.index = faiss.IndexFlatL2(self.embedding_coreset.shape[1])
        self.index.add(self.embedding_coreset) 
        faiss.write_index(self.index,  os.path.join(self.embedding_dir_path,'index.faiss'))


    def test_step(self, batch, batch_idx): # Nearest Neighbour Search
        x, gt, label, file_name, x_type = batch
        # extract embedding
        features = self(x)
        embeddings = []
        for feature in features:
            m = torch.nn.AvgPool2d(3, 1, 1)
            embeddings.append(m(feature))
        embedding_ = embedding_concat(embeddings[0], embeddings[1])
        embedding_test = np.array(reshape_embedding(np.array(embedding_)))
        score_patches, _ = self.index.search(embedding_test , k=args.n_neighbors)
        anomaly_map = score_patches[:,0].reshape((28,28))
        N_b = score_patches[np.argmax(score_patches[:,0])]
        w = (1 - (np.max(np.exp(N_b))/np.sum(np.exp(N_b))))
        score = w*max(score_patches[:,0]) # Image-level score
        gt_np = gt.cpu().numpy()[0,0].astype(int)
        anomaly_map_resized = cv2.resize(anomaly_map, (args.input_size, args.input_size))
        anomaly_map_resized_blur = gaussian_filter(anomaly_map_resized, sigma=4)
        self.gt_list_px_lvl.extend(gt_np.ravel())
        self.pred_list_px_lvl.extend(anomaly_map_resized_blur.ravel())
        self.gt_list_img_lvl.append(label.cpu().numpy()[0])
        self.pred_list_img_lvl.append(score)
        self.img_path_list.extend(file_name)
        # save images
        x = self.inv_normalize(x)
        input_x = cv2.cvtColor(x.permute(0,2,3,1).cpu().numpy()[0]*255, cv2.COLOR_BGR2RGB)
        self.save_anomaly_map(anomaly_map_resized_blur, input_x, gt_np*255, file_name[0], x_type[0])

    def test_epoch_end(self, outputs):
        print("Total pixel-level auc-roc score :")
        pixel_auc = roc_auc_score(self.gt_list_px_lvl, self.pred_list_px_lvl)
        print(pixel_auc)
        print("Total image-level auc-roc score :")
        img_auc = roc_auc_score(self.gt_list_img_lvl, self.pred_list_img_lvl)
        print(img_auc)
        print('test_epoch_end')
        values = {'pixel_auc': pixel_auc, 'img_auc': img_auc}
        self.log_dict(values)

def get_args():
    import argparse
    parser = argparse.ArgumentParser(description='ANOMALYDETECTION')
    parser.add_argument('--phase', choices=['train','test'], default='train')
    parser.add_argument('--dataset_path', default=r'/media/mostafahaggag/Shared_Drive/selfdevelopment/datasets/mvtec_anomaly_detection')
    parser.add_argument('--category', default='screw')
    parser.add_argument('--num_epochs', default=1)
    parser.add_argument('--batch_size', default=32)
    parser.add_argument('--load_size', default=256)
    parser.add_argument('--input_size', default=224)
    parser.add_argument('--coreset_sampling_ratio', default=0.001)
    parser.add_argument('--project_root_path', default=r'./test')
    parser.add_argument('--save_src_code', default=True)
    parser.add_argument('--save_anomaly_map', default=True)
    parser.add_argument('--n_neighbors', type=int, default=9)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args = get_args()
    trainer = pl.Trainer.from_argparse_args(args, default_root_dir=os.path.join(args.project_root_path, args.category), max_epochs=args.num_epochs, gpus=1)
    model = PatchCore(hparams=args)
    if args.phase == 'train':
        trainer.fit(model)
        trainer.test(model)
    elif args.phase == 'test':
        trainer.test(model)

