#!/usr/bin/env python3
"""
 Copyright (c) 2018 Intel Corporation.

 Permission is hereby granted, free of charge, to any person obtaining
 a copy of this software and associated documentation files (the
 "Software"), to deal in the Software without restriction, including
 without limitation the rights to use, copy, modify, merge, publish,
 distribute, sublicense, and/or sell copies of the Software, and to
 permit persons to whom the Software is furnished to do so, subject to
 the following conditions:

 The above copyright notice and this permission notice shall be
 included in all copies or substantial portions of the Software.

 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
 LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
 OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
 WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

import os
import sys
import logging as log
from openvino.inference_engine import IENetwork, IECore


class Network:
    """
    Load and configure inference plugins for the specified target devices 
    and performs synchronous and asynchronous modes for the specified infer requests.
    """

    def __init__(self):
        ### TODO: Initialize any class variables desired ###
        self.plugin = None
        self.network = None
        self.input_blob = None
        self.output_blob = None
        self.exec_network = None
        self.infer_request = None

    def load_model(self,model,device,input_size,output_size,num_request,cpu_extension=None,plugin = None):
        ### Load the model ###
        model_xml = model
        model_bin = os.path.splitext(model_xml)[0] + ".bin"
        if not plugin:
            log.info("Intializing plugin for {} device".format(device))
            self.plugin = IECore()
        else:
            self.plugin = plugin
        
        #Necessary extensions
        if cpu_extension and "CPU" in device:
            self.plugin.add_extension(cpu_extension, device)
            
        #Reading the IR 
        log.info('Reading the Intermediate Representation(IR)')
        self.network = IENetwork(model=model_xml, weights=model_bin)
        self.exec_network = self.plugin.load_network(self.network, device)
        
        ##Check for the supported Layers
        supported_layers = self.plugin.query_network(self.network,device)
        unsupported_layers =[l for l in self.network.layers.keys() if l not in supported_layers]
        if (len(unsupported_layers) != 0) and (cpu_extension and "CPU" in device):
            self.plugin.add_extension(cpu_extension, device)
            
        if num_request ==0:
            self.exec_network = self.plugin.load_network(network = self.network,device_name = device)
        else:
            self.exec_network = self.plugin.load_network(network = self.network,num_requests = num_request)
            

        self.input_blob = next(iter(self.network.inputs))
        self.out_blob = next(iter(self.network.outputs))
        
        assert len(self.network.inputs.keys()) == input_size, \
            "Supports only {} input topologies".format(len(self.network.inputs))
        assert len(self.network.outputs) == output_size, \
            "Supports only {} output topologies".format(len(self.network.outputs))            
            
        ### Note: You may need to update the function parameters. ###
        return self.exec_network,self.get_input_shape()

    def get_input_shape(self):
        ### Return the shape of the input layer ###
        return self.network.inputs[self.input_blob].shape

    def exec_net(self,req_id,frame):
        ### Start an asynchronous request ###
        self.infer_request = self.exec_network.start_async(request_id=req_id, 
            inputs={self.input_blob: frame})
        ### Note: You may need to update the function parameters. ###
        return self.exec_network

    def wait(self,req_id):
        ### Wait for the request to be complete. ###
        status = self.exec_network.requests[req_id].wait(-1)
        ### Return any necessary information ###
        ### Note: You may need to update the function parameters. ###
        return status

    def get_output(self,req_id,output=None):
        ### Extract and return the output results
        if output:
            r = self.infer_request.outputs[output]
        else:
            r = self.exec_network.requests[req_id].outputs[self.out_blob]
        ### Note: You may need to update the function parameters. ###
        return r
