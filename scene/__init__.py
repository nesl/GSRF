import os
import random
import json

from utils.system_utils import searchForMaxIteration
from scene.gaussian_model import GaussianModel
from arguments import ModelParams

from scene.dataset_readers import readRFSceneInfo

class Scene:

    gaussians : GaussianModel

    def __init__(self,
                 args: ModelParams, 
                 gaussians: GaussianModel, 
                 load_iteration=None, 
                 shuffle=True
                 ):
        
        self.model_path  = args.model_path
        self.cuda_dev    = args.data_device
        self.loaded_iter = None
        self.gaussians   = gaussians

        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))

            else:
                self.loaded_iter = load_iteration

            print("\nLoading saved point cloud data at iteration {}\n".format(self.loaded_iter))

        scene_info = readRFSceneInfo(args)

        if shuffle:
            random.shuffle(scene_info.train_spectrums)
            random.shuffle(scene_info.test_spectrums)
            
        self.cameras_extent = scene_info.nerf_normalization["radius"]

        self.train_set = scene_info.train_spectrums

        self.test_set = scene_info.test_spectrums
        
        if self.loaded_iter:

            self.gaussians.load_from_ply(os.path.join(self.model_path, 
                                                      "point_cloud", 
                                                      "iteration_" + str(self.loaded_iter), 
                                                      "point_cloud.ply"))
        else:
            self.gaussians.load_from_pcd(scene_info.point_cloud, 
                                           self.cameras_extent)


    def save(self, iteration):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))


    def getTrainSpectrums(self):
        return self.train_set


    def getTestSpectrums(self):
        return self.test_set
    

