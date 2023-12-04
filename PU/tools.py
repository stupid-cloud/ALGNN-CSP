import torch
import numpy as np
from pyfiglet import Figlet

def print_data_info(dataset):
    print()
    print(f'Dataset: {dataset}:')
    print('====================')
    print(f'Number of graphs: {len(dataset)}')
    print(f'Number of features: {dataset.num_features}')
    # print(f'Number of classes: {dataset.num_classes}')
    
    data = dataset[0]  # Get the first graph object.
    print(data)
    print()
    print('=============================================================')

    # Gather some statistics about the first graph.
    print(f'Number of nodes: {data.num_nodes}')
    print(f'Number of edges: {data.num_edges}')
    print(f'Average node degree: {data.num_edges / data.num_nodes:.2f}')
    print(f'Has isolated nodes: {data.has_isolated_nodes()}')
    print(f'Has self-loops: {data.has_self_loops()}')
    print(f'Is undirected: {data.is_undirected()}')
  
def set_random_seed(seed): 
    '''Fixes random number generator seeds for reproducibility'''
    torch.backends.cudnn.deterministic = True 
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    torch.manual_seed(seed)  # cpu
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_flatten_dict(now_dict, results={}):
	"""多维字典变一维"""
	for key, value in now_dict.items():  # 当前迭代的字典
		data = now_dict[key]  # 当前key所对应的value赋给data
		if isinstance(data, dict):  # 如果data是一个字典，就递归遍历
			get_flatten_dict(data,  results=results)  
		if isinstance(data, dict) != True:  # 找到了目标key，并且它的value不是一个字典
			results[key] = value		
	return results

def my_print(text, width=150):
	"""自定义打印"""
	print()
	f = Figlet(font='banner3', width=width)
	text = f.renderText(text)
	print(text)

def print1():
	data = r"""
     *-*,    	     *-*,            *-*,            *-*,
 ,*\/|`| \       ,*\/|`| \       ,*\/|`| \       ,*\/|`| \
 \'  | |'| *,	 \'  | |'| *,	 \'  | |'| *,	 \'  | |'| *,
  \ `| | |/ )	  \ `| | |/ )	  \ `| | |/ )	  \ `| | |/ )
   | |'| , /	   | |'| , /	   | |'| , /	   | |'| , /
   |'| |, /	   |'| |, /	   |'| |, /	   |'| |, /
 __|_|_|_|__	 __|_|_|_|__	 __|_|_|_|__	 __|_|_|_|__
[___________]   [___________]   [___________]	[___________]
 |         |	 |         |	 |         |	 |         |
 |         |	 |         |	 |         |	 |         |
 |         |	 |         |	 |         |	 |         |
 |_________|	 |_________|	 |_________|	 |_________|
 
"""
	print(data)
	

if __name__ == "__main__":
	print1()