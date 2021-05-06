from openvino.inference_engine import IECore

class Network:
    def __init__(self):
        self.ie = IECore()
        self.net = None
        self.exec_net = None
        self.input_name = None
        self.output_name = None

    def load_network(self, path_xml, path_bin):
        self.net = self.ie.read_network(model=path_xml, weights=path_bin)
        self.exec_net = self.ie.load_network(self.net, "CPU")
        self.input_name = list(self.net.input_info.keys())[0]
        self.output_name = list(self.net.outputs.keys())[0]
        
    def get_input_shape(self):
        return self.net.input_info[self.input_name].input_data.shape

    def get_output_shape(self):
        return self.net.outputs[self.output_name].shape
    
    def sync_inference(self, image):
        input_blob = {self.input_name: image}
        result = self.exec_net.infer(input_blob)
        return result[self.output_name]