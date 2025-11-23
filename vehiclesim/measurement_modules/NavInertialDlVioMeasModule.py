import numpy as np
from numpy.typing import NDArray
import torch
from torchvision.transforms import v2
import cv2

class NavInertialDlVioMeasModule():
    """
    Generates a mixed measurement model for inertial (vx and yaw rate) measurements
    and camera image measurements. The camera images, a history of IMU and CAN
    measurements, and a history of yaw angle estimates are provided to a transformer
    model to generate aiding yaw predictions. The measurements are mapped to the standard
    9-state navigation model defined in NavFullStateModule.
    """
    def __init__(self,
                 error_model:NDArray, 
                 model:torch.nn.Module,
                 device:torch.device,
                 model_weights:str,
                 img_transforms:v2.Compose=None):
        """
        Inertial and deep learning VIO mesaurement module for 9-state navigation model.

        Args:
            error_model (NDArray): Measurement error model. Equivalent to measurement noise matrix, R.
            model (torch.nn.Module): Network model with loaded parameters.
            device (torch.device): Device to run model.
            model_weights (str): Path to network weights.
            img_transforms (v2.Compose): Image transforms from torchvision library.
        """
        self.error_model = error_model
        self.device = device
        self.img_transforms = img_transforms
        # load model weights and csst to device
        state_dict = torch.load(model_weights)
        model.load_state_dict(state_dict)
        model = model.to(device)
        self.model = model
    
    def preprocess_network_inputs(self,
                                  imu_bank:NDArray,
                                  can_bank:NDArray,
                                  yaw_hist_bank:NDArray,
                                  image_paths:list):
        """
        Helper function to load and preprocess inputs for the network model.

        Args:
            imu_bank (NDArray): IMU measurements between the two consecutive images with shape (T, C_imu).
            can_bank (NDArray): CAN measurements between the two consecutive images with shape (T, C_can).
            yaw_hist_bank(NDArray): Yaw history in the lookback window up to time T-1 with shape (T-1, 1).
            image_paths (list(list)): List of lists of string paths to the left and right images (in sequential, left-right order).
                                      Ex) image_paths = [[left_path_k-1, left_path_k], [right_path_k-1, right_path_k]].
            
        Returns:
            inputs (list): Input list for network model.
        """
        # preprocess images
        left_paths = image_paths[0]
        right_paths = image_paths[1]
        num_imgs = len(left_paths)
        processed_imgs = []
        for i in range(0,num_imgs):
            left_img = cv2.imread(left_paths[i])
            right_img = cv2.imread(right_paths[i])
            if left_img is None or right_img is None:
                raise Exception("Left or right image is None.")
            concat_img = cv2.hconcat([right_img, left_img])
            if self.img_transforms is not None:
                concat_img = self.img_transforms(concat_img)
            processed_imgs.append(concat_img)
        # for image in processed_imgs:
        #     image = processed_imgs[0].permute(1,2,0).numpy()
        #     cv2.imshow("test", image)
        #     cv2.waitKey(0)
        #     image = processed_imgs[1].permute(1,2,0).numpy()
        #     cv2.imshow("tes2", image)
        #     cv2.waitKey(0)
        #     stop=1
        input_cam = torch.stack(processed_imgs)
        input_cam = input_cam.unsqueeze(dim=0) # emulate a batch size of 1 for model
        
        # preprocess inertial (assumes both imu and can measurements recieved for now)
        torch_imu = torch.from_numpy(imu_bank)
        torch_can = torch.from_numpy(can_bank)
        input_inertial = torch.cat((torch_can, torch_imu), dim=1)
        input_inertial = input_inertial.unsqueeze(dim=0) # emulate a batch size of 1 for model
        
        # preprocess yaw history input
        input_yaw_hist = torch.from_numpy(yaw_hist_bank)
        sin_yaw_hist = torch.sin(input_yaw_hist)
        cos_yaw_hist = torch.cos(input_yaw_hist)
        input_yaw_hist = torch.column_stack((sin_yaw_hist, cos_yaw_hist))
        input_yaw_hist = input_yaw_hist.unsqueeze(dim=0) # emulate a batch size of 1 for model
        
        # generate input list
        inputs = [input_cam, input_inertial, input_yaw_hist]
        return inputs
    
    def generate_meas_model(self,
                            vx:float,
                            yaw_rate:float,
                            imu_bank:NDArray,
                            can_bank:NDArray,
                            yaw_hist_bank:NDArray,
                            image_paths:list):
        """
        Generates the measurement model for longitudinal vel, yaw rate gyro, and yaw prediction
        output from the transformer network.
        
        Args:
            vx (float): Longitudinal velocity measurement as reported from sensor.
            yaw_rate (float): Yaw rate measurement as reported from sensor.
            imu_bank (NDArray): IMU measurements between the two consecutive images with shape (T, C_imu).
            can_bank (NDArray): CAN measurements between the two consecutive images with shape (T, C_can).
            yaw_hist_bank(NDArray): Yaw history in the lookback window up to time T-1 with shape (T-1, 1).
            image_paths (list(list)): List of lists of string paths to the left and right images (in sequential, left-right order).
                            Ex) image_paths = [[left_path_k-1, left_path_k], [right_path_k-1, right_path_k]].
        Returns:
            z (NDArray): Measurements vector.
            H (NDArray): Measurement observation matrix.
            R (NDArray): Measurement noise matrix.
        """
        # call preprocessor to get network compliant input list
        inputs = self.preprocess_network_inputs(
            imu_bank,
            can_bank,
            yaw_hist_bank,
            image_paths
        )
        inputs[0] = inputs[0].to(device=self.device, dtype=torch.float32) # images
        inputs[1] = inputs[1].to(device=self.device, dtype=torch.float32) # IMU
        inputs[2] = inputs[2].to(device=self.device, dtype=torch.float32) # yaw history
        # call model for yaw estimate
        with torch.no_grad():
            self.model.eval()
            _, _, yaw_est = self.model(inputs)
        # convert yaw est from tensor to numpy
        if yaw_est.get_device() != -1: # on gpu
            yaw_est = yaw_est.cpu().numpy()
        else: # on cpu
            yaw_est = yaw_est.numpy()
        # extract sin and cos components from estimate and reconstruct absolute yaw estimate
        sin_yaw = yaw_est[0,0]
        cos_yaw = yaw_est[0,1]
        yaw_est = np.arctan2(sin_yaw, cos_yaw)
        # yaw_est_final = np.unwrap((yaw_est_final + 2*np.pi) % (2*np.pi))
        # unwrap estimate using history window
        yaw_window = np.append(yaw_hist_bank, yaw_est)
        yaw_window_unwrapped = np.unwrap(yaw_window)
        yaw_est_final = yaw_window_unwrapped[-1] # pull back unwrapped estimate
        
        z = np.array([
            [vx],
            #[yaw_rate],
            [yaw_est_final]
        ])    
        H = np.array([
            [0, 0, 1, 0, 0, 0, 0, 0, 0],
            #[0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0, 0]
        ])
        R = self.error_model
        
        return z, H, R