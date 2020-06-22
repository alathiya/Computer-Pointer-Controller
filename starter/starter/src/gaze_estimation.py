'''
This is a sample class for a model. You may choose to use it as-is or make any changes to it.
This has been provided just to give you an idea of how to structure your model class.
'''
import cv2
import numpy as np
import logging as log
from openvino.inference_engine import IENetwork, IECore
import math


class gaze_estimation:
    '''
    Class for the Face Detection Model.
    '''
    def __init__(self, model_name, device='CPU', extensions=None):
        '''
        TODO: Use this to set your instance variables.
        '''
        self.model_weights=model_name+'.bin'
        self.model_structure=model_name+'.xml'
        self.device=device
        self.extension = extensions

        self.model=IENetwork(self.model_structure, self.model_weights)
        self.input_name=next(iter(self.model.inputs))
        self.input_shape=self.model.inputs[self.input_name].shape
        self.output_name=next(iter(self.model.outputs))
        self.output_shape=self.model.outputs[self.output_name].shape

    def load_model(self):
        '''
        TODO: You will need to complete this method.
        This method is for loading the model to the device specified by the user.
        If your model requires any Plugins, this is where you can load them.
        '''
        self.plugin = IECore()
        
        if self.extension and self.device=='CPU':
            self.plugin.add_extension(self.extension, self.device)
        
            # Check for supported layers ###
            supported_layers = self.plugin.query_network(network=self.model, device_name=self.device)
        
            unsupported_layers = [l for l in self.model.layers.keys() if l not in supported_layers]
        
            if len(unsupported_layers) != 0:
                logger.error("Unsupported layers found: {}".format(unsupported_layers))
                logger.error("Check whether extensions are available to add to IECore.")
                exit(1)
          
        self.net = self.plugin.load_network(network=self.model, device_name=self.device, num_requests=1)

        return self.net

    def predict(self, left_eye_image, right_eye_image, head_pose_output):
        '''
        TODO: You will need to complete this method.
        This method is meant for running predictions on the input image.
        '''
        left_eye_image_preprocess, right_eye_image_preprocess = self.preprocess_input(left_eye_image, right_eye_image)

        self.results = self.net.infer(
            inputs={'left_eye_image': left_eye_image_preprocess, 'right_eye_image': right_eye_image_preprocess,
                    'head_pose_angles': head_pose_output})
        
        mouse_coord, gaze_vector = self.preprocess_output(self.results, head_pose_output)

        return mouse_coord, gaze_vector

    def check_model(self):
        pass

    def preprocess_input(self, left_eye_image, right_eye_image):
        '''
        Before feeding the data into the model for inference,
        you might have to preprocess it. This function is where you can do that.
        '''
        left_eye_image_preprocess = cv2.resize(left_eye_image, (60, 60))
        left_eye_image_preprocess = left_eye_image_preprocess.transpose((2, 0, 1))
        left_eye_image_preprocess = left_eye_image_preprocess.reshape(1, *left_eye_image_preprocess.shape)

        right_eye_image_preprocess = cv2.resize(right_eye_image, (60, 60))
        right_eye_image_preprocess = right_eye_image_preprocess.transpose((2, 0, 1))
        right_eye_image_preprocess = right_eye_image_preprocess.reshape(1, *right_eye_image_preprocess.shape)

        return left_eye_image_preprocess, right_eye_image_preprocess

    def preprocess_output(self, outputs, head_pose_estimation_output):
        '''
        Before feeding the output of this model to the next model,
        you might have to preprocess the output. This function is where you can do that.
        '''
        roll_value = head_pose_estimation_output[2]
        outputs = outputs[self.output_name][0]
        
        cos_theta = math.cos(roll_value * math.pi / 180)
        sin_theta = math.sin(roll_value * math.pi / 180)

        x_value = outputs[0] * cos_theta + outputs[1] * sin_theta
        y_value = outputs[1] * cos_theta - outputs[0] * sin_theta

        return (x_value, y_value), outputs
