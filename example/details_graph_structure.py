from curses import doupdate
from dbm import ndbm
import pandas as pd
import numpy as np 
from tqdm import tqdm
import re
import os
from torch.nn.functional import normalize

import torch 
import torch_geometric

#I check some stuff - standard way

print(f"torch version: {torch.__version__}")
print(f"Cuda: {torch.cuda.is_available()}")
print(f"PyG version: {torch_geometric.__version__}")

import os.path as osp
from torch_geometric.data import Dataset, download_url, Data

global_dict = {}

def set_global_dict():
    global global_dict
    global_dict = {
                "input" :[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                "x"  :[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                "y"  :[0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                "z"  :[0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                "h"  :[0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                "s"  :[0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                "sd":[0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                "cx" :[0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0], 
                "cy" :[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                "cz" :[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                "measurement":[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                "rx" :[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                "ry" :[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                "rz" :[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]}
set_global_dict()

class MyOwnDataset(Dataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        """ Root is where the dataset is going to be stored aka data/raw and data/processed

        Args:
            root (_type_): _description_
            transform (_type_, optional): _description_. Defaults to None.
            pre_transform (_type_, optional): _description_. Defaults to None.
            pre_filter (_type_, optional): _description_. Defaults to None.
        """
        super(MyOwnDataset, self).__init__(root, transform, pre_transform, pre_filter)
        

    @property
    def raw_file_names(self):
        #name of dataset or multiple datasets I have
        list_files = []
        og = "/home/leonardo/MINISQ-datasets/data_engineering/data/raw"
        for file in os.listdir(og):
            list_files.append(og+"/"+file)
        #print(list_files)
        return list_files

    @property
    def processed_file_names(self):
        return "not_implemented yet" #['data_1.pt', 'data_2.pt', ...]

    def download(self):
        # Download to `self.raw_dir`.
        pass
    def process(self):
        """Here I construct the graph and pass it to data object
        """
        idx4 = 0
        idx8 = 0
        idx12 = 0
        idx16 = 0
        for raw_path in self.raw_paths:
            #print(raw_path)
            # Use regular expression to extract the number
            match = re.search(r'(\d+)qubit\.csv', raw_path)

            if match:
                qub_number = int(match.group(1))
                print("Extracted number:", qub_number)
            else:
                print("No match found.")
            #break
            #qub_number = int(raw_path[-10])
            # Read data from `raw_path`iterating over all csv
            data_csv = pd.read_csv(raw_path)
            #loop in the dataset, each row 1 graph
            #print("The length", len(data_csv))
            for row_idx, row in tqdm(data_csv.iterrows(), total = len(data_csv)):
                
                _string = row["Data"]
                _class = row["Class"]
                #check line by line and build the graph
                edge_index, y, nodes_list = self.Graph_builder(_string, _class)
                #print("edge_index", edge_index)
                #print("y", y)
                #print("nodes_list", nodes_list)
                data = Data(x=nodes_list, edge_index=edge_index.t().contiguous(), y = y) #.t()
                #print("DIRECTORY", self.processed_dir)
                dir = self.processed_dir
                idx = 1
                if qub_number == 8:
                    idx = idx8
                    idx8 += 1
                elif qub_number == 4:
                    idx = idx4
                    idx4 += 1
                elif qub_number == 12:
                    idx = idx12
                    idx12 += 1
                    
                elif qub_number == 16:
                    idx = idx16
                    idx16 += 1
                else:
                    pass
                    
                dir_ = osp.join(dir, f'{qub_number}_qubits')

                # Check if the qubit number subdirectory exists
                if not os.path.exists(dir_):
                    os.makedirs(dir_)
                    
                torch.save(data, osp.join(dir_, f'graph_{idx}.pt'))
        print("I have created graphs for {} files of 4 qubits and {} files of 8 qubits, {} qubits and {} qubits".format(idx4, idx8, idx12, idx16))
            
            #break #temporary to test

    def Graph_builder(self, _string, _class):
        """Build a graph from the string description and class of a circuit

        Args:
            _string (_type_): row of dataset, the string of the code in qasm
            _class (_type_): the class of the circuit e.g. 1 etc, single int
            
        Returns:    edge_index: edges
                    edge_attributes: quantum gates as edge attrs
                    y: the class 
                    x: 2d nodes (qubit number, time number)
        """
        lines = _string.splitlines()[2:]
        #print(lines)
        
        count = 0
        edge_index = []
        x = [] 
        index_tracker = {}
        for line in lines:
            #line where I define input qubits
            if count == 0:
                qubits = [s for s in re.findall(r'-?\d*\.{0,1}\d+', line)]
                #print("qubits", qubits)
                #there are no edges at this step obviously
                #declare the n input qubits
                nodes = [global_dict["input"] for i in range(int(qubits[0]))]
                #print(nodes)
                x.extend(nodes)
                for i in range(len(nodes)):
                    index_tracker[str(i)] = i
            else:
                #any other line
                numbers = [s for s in re.findall(r'-?\d*\.{0,1}\d+', line)] #these are the numbers appearing in the line of OPENQASM2.0
                #print("numbers appearing", numbers)
                gate_string = line.split()[0] #this is the legit string because is always 2 chars max
                ###################
                ###################
                #new code
                
                #print(gate_string)
                #print(len(gate_string))
                if len(gate_string) > 2:
                    gate_string = gate_string[0:2]
                #print(gate_string)
                
                #now remember that if the gate is cx, cz, cy, then you have 2 qubit updates
                #otherwise is 1 qubit update and parameter when the numbers are 2
                if gate_string == "cx" or gate_string == "cy" or gate_string == "cz":
                    #extend the nodes
                    new_node = self.array_from_gate(gate_string)
                    x.extend([new_node])
                    #edges
                    num_q_0 = numbers[-2]
                    num_q_1 = numbers[-1]
                    old_idx_0 = index_tracker[str(num_q_0)]
                    old_idx_1 = index_tracker[str(num_q_1)]
                    index_tracker[num_q_0] = len(x) - 1
                    index_tracker[num_q_1] = len(x) - 1
                    edge_index.extend([[old_idx_0, index_tracker[str(num_q_0)]]])
                    edge_index.extend([[old_idx_1, index_tracker[str(num_q_1)]]])
                    #print("x end", x)
                    #print("index_tracker", index_tracker)
                    #print("edges now", edge_index)
                else:
                    if len(numbers) > 1:
                        #print("numbers", numbers)
                        #print("numbers[0]", float(numbers[0]))
                        #print(gate_string)
                        a = float(numbers[0])
                        new_node = self.array_from_gate(gate_string, a) #TODO: check this float, in the past was creating issues
                    else:
                        new_node = self.array_from_gate(gate_string)
                    x.extend([new_node])
                    num_q = numbers[-1]  #because the only 2 qubit gates are the cx, cy, cz
                    #print(index_tracker)
                    old_idx = index_tracker[num_q]
                    index_tracker[num_q] = len(x) - 1
                    #add edge
                    edge_index.extend([[old_idx, index_tracker[str(num_q)]]])
                
            count += 1
        edge_index = torch.tensor(np.array(edge_index))
        x = torch.from_numpy(np.array(x)).type(torch.float32)
        x = normalize(x, p = 1.0)  #normalization, attention
        #print("x", x)
        y = int(_class)
        return edge_index , y, x
    
    def array_from_gate(self, gate_str, theta_param = 0):
        """_summary_

        Args:
            gate_str (_type_): the string of the gate from OPENQASM2.0
            theta_param (_type_): the parameter (if none is set to 0), for the gate

        Returns:
            array: the feature vector associated with the node we are dealing with
        """
        feature_vector_node = global_dict[gate_str]
        feature_vector_node[-1] = theta_param
        return feature_vector_node
        
    
    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        data = torch.load(osp.join(self.processed_dir, f'data_{idx}.pt'))
        return data
    

dataset = MyOwnDataset(root = "/home/leonardo/MINISQ-datasets/data_engineering/data")