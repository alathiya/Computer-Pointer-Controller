'''
This is a sample class for a model. You may choose to use it as-is or make any changes to it.
This has been provided just to give you an idea of how to structure your model class.
'''

import cv2
import numpy as np
import logging as log
from openvino.inference_engine import IENetwork, IECore

class facial_landmarks_detection:
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

    def predict(self, image):
        '''
        TODO: You will need to complete this method.
        This method is meant for running predictions on the input image.
        '''
        preprocess_image = self.preprocess_input(image)

        results = self.net.infer({self.input_name: preprocess_image})

        eye_coords = self.preprocess_output(results, image.shape[1], image.shape[0])

        lefteye_x_min = eye_coords ['left_eye_x_coord'] - 10
        lefteye_x_max = eye_coords ['left_eye_x_coord'] + 10
        lefteye_y_min = eye_coords ['left_eye_y_coord'] - 10
        lefteye_y_max = eye_coords ['left_eye_y_coord'] + 10

        righteye_x_min = eye_coords ['right_eye_x_coord'] - 10
        righteye_x_max = eye_coords ['right_eye_x_coord'] + 10
        righteye_y_min = eye_coords ['right_eye_y_coord'] - 10
        righteye_y_max = eye_coords ['right_eye_y_coord'] + 10

        eye_coord = [[lefteye_x_min, lefteye_y_min, lefteye_x_max, lefteye_y_max],
                          [righteye_x_min, righteye_y_min, righteye_x_max, righteye_y_max]]

        left_eye_image = image[lefteye_x_min:lefteye_x_max, lefteye_y_min:lefteye_y_max]
        right_eye_image = image[righteye_x_min:righteye_x_max, righteye_y_min:righteye_y_max]

        return left_eye_image, right_eye_image, eye_coord

    def check_model(self):
        pass

    def preprocess_input(self, image):
        '''
        Before feeding the data into the model for inference,
        you might have to preprocess it. This function is where you can do that.
        '''
        
        image = cv2.resize(image, (self.input_shape[3], self.input_shape[2]))
        image = image.transpose((2,0,1))
        image = image.reshape(1, *image.shape)

        return image 

    def preprocess_output(self, outputs, width, height):
        '''
        Before feeding the output of this model to the next model,
        you might have to preprocess the output. This function is where you can do that.
        '''
        eye_coords = {}

        outputs = outputs[self.output_name][0]

        eye_coords['left_eye_x_coord'] = int(outputs[0] * width)
        eye_coords['left_eye_y_coord'] = int(outputs[1] * height)
        eye_coords['right_eye_x_coord'] = int(outputs[2] * width)
        eye_coords['right_eye_y_coord'] = int(outputs[3] * height)

        return eye_coords
