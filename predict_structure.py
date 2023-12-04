# -*- coding: utf-8 -*-

import pandas as pd
import os
import time
import warnings
import pickle
import yaml
import shutil
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from crystal.deeplearn.process import datasets, get_graph_data
import numpy as np
import hyperopt as hy
from sko.GA import GA
from sko.PSO import PSO
from pymatgen.core.structure import Structure, Lattice
import torch.nn.functional as F
from GN_OA.utils.file_utils import check_and_rename_path
from GN_OA.utils.algo_utils import hy_parameter_setting
from GN_OA.utils.print_utils import print_header, print_run_info
from GN_OA.utils.wyckoff_position import WyckoffPosition
import torch
from torch_geometric.data import DataLoader, Dataset
from pymatgen.core.composition import Composition
from pymatgen.core.periodic_table import Element

class PredictStructure(object):

    @print_header
    def __init__(self, config):
        self.config = config
        
        composition = Composition(config['global']['composition'])
        # self.formula = composition.formula # Ca4S4    
        elements_count = composition.get_el_amt_dict() # [4, 4]
        space_group = range(config['lattice']['space_group'][0], config['lattice']['space_group'][1]+1)

        self.elements = list(elements_count.keys()) # ['Ca', 'S']

        self.nn_model = torch.load(config['global']['gn_model'])["full_model"]

        self.all_atoms = [Element(key) for key, value in elements_count.items() for _ in range(int(value))]

        self.sys_weight = config["program"]["sys_weight"]
        # 根据空间群以及原子数获得wyckoffs以及数量。 dict, 145894
        self.wyckoffs_dict, self.max_wyckoffs_count = WyckoffPosition().get_all_wyckoff_combination(space_group, list(elements_count.values()))
        self.total_atom_count = int(composition.num_atoms)  # 8

        self.output_path = os.path.join("./results", config['global']['composition']+"-{}".format(config['program']['sys_weight'][0]))
        os.makedirs(self.output_path)
        # ./results/structures
        self.structures_path = os.path.join(self.output_path, 'structures')

        # self.energy = 999 
        self.is_sko = [None, None]
        if config['program']["algorithm"][0] in ['bo', 'rs']:
            self.is_sko[0] = False
            self.find_all_structure_hyperopt()    
        elif config['program']["algorithm"][0] in ['ga', 'pso']:
            self.is_sko[0] = True
            self.find_all_structure_sko()
        else:
            
            self.find_all_structure_enumerate()

    @print_run_info('Predict crystal structure')
    def find_all_structure_enumerate(self):
        print('1.enumerate all space group and wyckoff positions')
        self.step_number1 = 0
        
        with open(os.path.join(self.output_path, 'energy_data.csv'), 'w+') as f:
            f.writelines("step, sg_number, wp_number, time, energy, sys_prop, score\n")
       
        self.start_time1 = time.time() 
        for sg in range(self.config['lattice']['space_group'][0], self.config['lattice']['space_group'][1]+1):
            # for wp in range(len(self.wyckoffs_dict[sg])):

                wp = 1
                if self.config['program']["algorithm"][1] in ['bo', 'rs']:
                    self.is_sko[1] = False 
                    self.find_stable_structure_hyperopt((sg, wp))  
                else:
                    self.is_sko[1] = True
                    self.find_stable_structure_sko((sg, wp))


    @print_run_info('Predict crystal structure')
    def find_all_structure_hyperopt(self):
        print('1.Opitmizing space group and wyckoff positions')
        self.step_number1 = 0
        if self.config['program']["algorithm"][1] in ['bo', 'rs']:
            self.is_sko[1] = False 
            func = self.find_stable_structure_hyperopt     
        else:
            self.is_sko[1] = True
            func = self.find_stable_structure_sko
            
        algo, max_step, rand_seed = self.hyperopt_initial(index=0)

        with open(os.path.join(self.output_path, 'energy_data.csv'), 'w+') as f:
            f.writelines("step, sg_number, wp_number, time, energy, sys_prop, score\n")
    
        sg = hy_parameter_setting('sg', self.config["lattice"]['space_group'], ptype='int') 
        wp = hy_parameter_setting('wp', [0, self.max_wyckoffs_count])
        pbounds = {'sg': sg, 'wp': wp}
        trials = hy.Trials()
        self.start_time1 = time.time()
        best = hy.fmin(fn=func,
                    space=pbounds,
                    algo=algo,
                    max_evals=max_step,
                    trials=trials,
                    rstate=rand_seed  # 随机种子
                    )
        
        
    def find_all_structure_sko(self):
        print('1.Opitmizing space group and wyckoff positions')
        self.step_number1 = 0
        if self.config['program']["algorithm"][1] in ['bo', 'rs']:
            self.is_sko[1] = False 
            func = self.find_stable_structure_hyperopt     
        else:
            self.is_sko[1] = True
            func = self.find_stable_structure_sko
 
        
        with open(os.path.join(self.output_path, 'energy_data.csv'), 'w+') as f:
            f.writelines("step, sg_number, wp_number, time, energy, sys_prop, score\n")
       
        sg = self.config["lattice"]['space_group']
        l_b = [sg[0], 0]
        u_b = [sg[1], self.max_wyckoffs_count]

        algo = self.sko_initial(0, func, l_b, u_b)
        self.start_time1 = time.time() 
        algo.run()
        
            
    def find_stable_structure_sko(self, *args):

        self.step_number1 += 1
        self.min_score = np.inf 
        self.step_number2 = 0
        self.formation_energy = np.inf
        self.sys_prop = 0

        args = args[0]  
        
        if self.is_sko[0] == True:
            self.sg = int(args[0])
            wp_list = self.wyckoffs_dict[self.sg]
            self.wp = int(args[1] * len(wp_list) / self.max_wyckoffs_count) 
        elif self.is_sko[0] == False:
            self.sg = int(args['sg'])
            wp_list = self.wyckoffs_dict[self.sg]
            self.wp = int(args['wp'] * len(wp_list) / self.max_wyckoffs_count)
        else:
            self.sg = args[0]
            wp_list = self.wyckoffs_dict[self.sg]
            self.wp = args[1]

        try:
            if len(self.wyckoffs_dict[self.sg]) == 0:
                raise Exception()
        except:
            # return {'loss': np.inf, 'status': hy.STATUS_FAIL}
            # result = 999
            if self.is_sko[0]:
                return np.inf
            else:
                return {'loss': np.inf, 'status': hy.STATUS_FAIL}
        print("space group: {}, wyckoff number: {}/{}".format(self.sg, self.wp, len(wp_list)))
        a = self.config["lattice"]['lattice_a']
        b = self.config["lattice"]['lattice_b']
        c = self.config["lattice"]['lattice_c']
        alpha = self.config["lattice"]['lattice_alpha']
        beta = self.config["lattice"]['lattice_beta']
        gamma =self.config["lattice"]['lattice_gamma']

        l_b = [a[0], b[0], c[0], alpha[0], beta[0], gamma[0]]
        u_b = [a[1], b[1], c[1], alpha[1], beta[1], gamma[1]]

        for i in range(self.total_atom_count):
            l_b += [0, 0, 0]
            u_b += [1, 1, 1]


        print('2.Opitmizing structures')
        self.start_time2 = time.time() 
        func = self.predict_structure_energy
        algo = self.sko_initial(1, func, l_b, u_b)  
        algo.run()
        # print('best_x is ', pso.gbest_x, 'best_y is', pso.gbest_y)
        

        if self.sys_weight[0] == 0.0:
            score = self.formation_energy
        else:
            score = (1-self.sys_prop)*(self.sys_weight[0])+float(F.sigmoid(torch.Tensor([self.formation_energy])))*(1-self.sys_weight[0])
        
        if self.min_score != np.inf:
            with open(os.path.join(self.output_path, 'energy_data.csv'), 'a+') as f:
               f.write(','.join([str(self.step_number1),  str(self.sg), str(self.wp), 
                                  str(time.time()-self.start_time1), 
                                  str(self.formation_energy), str(self.sys_prop), str(score)]) + '\n') 
               
        if self.is_sko[0] == True:
            return score
        elif self.is_sko[0] == False:
            return {'loss': score, 'status': hy.STATUS_OK}
        else:
            pass

    def find_stable_structure_hyperopt(self, *args):      
        
        self.step_number1 += 1
        self.min_score = np.inf 
        self.step_number2 = 0
        self.formation_energy = np.inf
        self.sys_prop = 0

        args = args[0] 
        if self.is_sko[0] == True:
            self.sg = int(args[0])
            wp_list = self.wyckoffs_dict[self.sg]
            self.wp = int(args[1] * len(wp_list) / self.max_wyckoffs_count) 
        elif self.is_sko[0] == False:
            self.sg = int(args['sg'])
            wp_list = self.wyckoffs_dict[self.sg]
            self.wp = int(args['wp'] * len(wp_list) / self.max_wyckoffs_count)
        else:
            self.sg = args[0]
            wp_list = self.wyckoffs_dict[self.sg]
            self.wp = args[1]
        
        try:
            if len(self.wyckoffs_dict[self.sg]) == 0:
                raise Exception()
        except:
            # return {'loss': np.inf, 'status': hy.STATUS_FAIL}
            # result = 999
            if self.is_sko[0]:
                return np.inf
            else:
                return {'loss': np.inf, 'status': hy.STATUS_FAIL}
        print("space group: {}, wyckoff number: {}/{}".format(self.sg, self.wp, len(wp_list)))
        # 超参数优化
        a = hy_parameter_setting('a', self.config["lattice"]['lattice_a'])
        b = hy_parameter_setting('b', self.config["lattice"]['lattice_b'])
        c = hy_parameter_setting('c', self.config["lattice"]['lattice_c'])
        alpha = hy_parameter_setting('alpha', self.config["lattice"]['lattice_alpha'])
        beta = hy_parameter_setting('beta', self.config["lattice"]['lattice_beta'])
        gamma = hy_parameter_setting('gamma', self.config["lattice"]['lattice_gamma'])
       
        pbounds = {'a': a, 'b': b, 'c': c,
                   'alpha': alpha, 'beta': beta, 'gamma': gamma
                   }
        
        # i_atoms = 0
        # compound_times = self.total_atom_count / sum(self.elements_count)
        for i in range(self.total_atom_count):
            pbounds['x' + str(i)] = hy.hp.uniform('x' + str(i), 0, 1)
            pbounds['y' + str(i)] = hy.hp.uniform('y' + str(i), 0, 1)
            pbounds['z' + str(i)] = hy.hp.uniform('z' + str(i), 0, 1)
            # self.all_atoms.append(self.elements)
        print('2.Opitmizing structures')
        algo, max_step, rand_seed = self.hyperopt_initial(index=1)
        trials = hy.Trials()   
        self.start_time2 = time.time()    
        best = hy.fmin(fn=self.predict_structure_energy,
                       space=pbounds,
                       algo=algo,
                       max_evals=max_step,
                       trials=trials,
                       rstate=rand_seed  # 随机种子
                       )
        # result = self.min_energy
        print("*" * 100)
        if self.sys_weight[0] == 0.0:
            score = self.formation_energy
        else:
            score = (1-self.sys_prop)*(self.sys_weight[0])+float(F.sigmoid(torch.Tensor([self.formation_energy])))*(1-self.sys_weight[0])
  
        if self.min_score != np.inf:
            with open(os.path.join(self.output_path, 'energy_data.csv'), 'a+') as f:
                f.write(','.join([str(self.step_number1),  str(self.sg), str(self.wp), 
                                  str(time.time()-self.start_time1), 
                                  str(self.formation_energy), str(self.sys_prop), str(score)]) + '\n') 
        
        
        if self.is_sko[0] == True:
            return score
        elif self.is_sko[0] == False:
            return {'loss': score, 'status': hy.STATUS_OK}
        else:
            pass
   
    def predict_structure_energy(self, *args):
        # print(args)
        # input()
        
        self.step_number2 += 1
        # self.wp = 1
        # self.a, self.b, self.c, self.alpha, self.beta, self.gamma = args
        args = args[0]
        if self.is_sko[1]:
            _dict = {'a': args[0], 'b': args[1], 'c': args[2],
                     'alpha': args[3], 'beta': args[4], 'gamma': args[5]}
            for i in range(self.total_atom_count):
                _dict['x' + str(i)] = args[6 + i * 3 + 0]
                _dict['y' + str(i)] = args[6 + i * 3 + 1]
                _dict['z' + str(i)] = args[6 + i * 3 + 2]
        else:
            _dict = args
        
        structure = self.get_structure(_dict)  

        try:
            self.atomic_dist_and_volume_limit(structure)  
        except:
            if self.is_sko[1]:
                return np.inf
            else:
                return {'loss': np.inf, 'status': hy.STATUS_OK}
            # pass 
            # 结构可用性检测
        
        energy = self.get_energy(structure)
        sys_prop = self.get_synthesizability(structure)

        if self.sys_weight[1] == 1.0:
            score = energy
        else:
            score = (1-self.sys_prop)*(self.sys_weight[1])+float(F.sigmoid(torch.Tensor([self.formation_energy])))*(1-self.sys_weight[1])
       
       
        
        structure_file_name = os.path.join(
                self.structures_path, "temp",
                '%d_%d_%f_%f.cif' % (self.step_number1, self.step_number2, time.time()-self.start_time2, score)
            )
        os.makedirs(os.path.join(self.structures_path, "temp"), exist_ok=True)
        structure.to(fmt='cif', filename=structure_file_name)

        if score < self.min_score:
            self.min_score = score
            self.formation_energy = energy
            self.sys_prop = sys_prop
            structure_file_name = os.path.join(
                self.structures_path,
                '%d_%d_%d.cif' % (self.step_number1, self.sg, self.wp)
            )
            structure.to(fmt='cif', filename=structure_file_name) 
        if self.is_sko[1]:
            return score
        else:
            return {'loss': score, 'status': hy.STATUS_OK}

    def get_energy(self, structure):
        
        assert os.path.exists('./crystal/config.yml'), "Config file was not found!"
        
        with open('./crystal/config.yml', "r") as ymlfile:
            config = yaml.load(ymlfile, Loader=yaml.FullLoader) 

        config['Processing']['run_mode'] = 'Predicting'    
        data = get_graph_data(config['Processing'], structure)
        
        self.nn_model.to('cuda')
        data = data.to('cuda')
        result = float(self.nn_model(data).cpu().detach())
        return result

    def get_synthesizability(self, structure):
        assert os.path.exists('./PU/config.yml'), "Config file was not found!"
        with open('./PU/config.yml', "r") as ymlfile:
            config = yaml.load(ymlfile, Loader=yaml.FullLoader) 

        config['Processing']['run_mode'] = 'Predicting'
        data = get_graph_data(config['Processing'], structure) 
        data = data.to('cuda')
        model_list = os.listdir('./PU/saved_model')
        prediction_list = []
        for i, model in enumerate(model_list):
            model = torch.load(os.path.join('./PU/saved_model', model))['full_model']
            model.eval()
            # model_summary(model)
            model.to('cuda')
            # mean_file
            with torch.no_grad():
                prediction = model(data)
                prediction = F.softmax(prediction.cpu(), dim=1).numpy().flatten()
                prediction_list.append(prediction)   
        mean_value = np.array(prediction_list).mean(axis=0)
        return mean_value[1]
        # mean_file.to_csv(os.path.join(pred_params['save_path'], "mean.csv"), index=False)
        # return result


    def get_structure(self, struc_parameters):
        atom_positions = []
        count = 0
        
        wp_list = self.wyckoffs_dict[self.sg]
        if self.wp == len(wp_list):
            self.wp -= 1 
        wp = wp_list[self.wp]
       

        for i, wp_i in enumerate(wp):
            for wp_i_j in wp_i:
                for wp_i_j_k in wp_i_j:
                                                    
                    if 'x' in wp_i_j_k:
                        wp_i_j_k = wp_i_j_k.replace('x', str(struc_parameters['x' + str(count)]))
                    if 'y' in wp_i_j_k:
                        wp_i_j_k = wp_i_j_k.replace('y', str(struc_parameters['y' + str(count)]))
                    if 'z' in wp_i_j_k:
                        wp_i_j_k = wp_i_j_k.replace('z', str(struc_parameters['z' + str(count)]))
                    count += 1 

                    atom_positions.append(list(eval(wp_i_j_k)))
                    
        if self.sg in [0, 1, 2]:
            lattice = Lattice.from_parameters(a=struc_parameters['a'], b=struc_parameters['b'], c=struc_parameters['c'],
                                              alpha=struc_parameters['alpha'], beta=struc_parameters['beta'], gamma=struc_parameters['gamma'])
        elif self.sg in list(range(3, 15 + 1)):
            lattice = Lattice.from_parameters(a=struc_parameters['a'], b=struc_parameters['b'], c=struc_parameters['c'],
                                              alpha=90, beta=struc_parameters['beta'], gamma=90)
        elif self.sg in list(range(16, 74 + 1)):
            lattice = Lattice.from_parameters(a=struc_parameters['a'], b=struc_parameters['b'], c=struc_parameters['c'],
                                              alpha=90, beta=90, gamma=90)
        elif self.sg in list(range(75, 142 + 1)):
            lattice = Lattice.from_parameters(a=struc_parameters['a'], b=struc_parameters['a'], c=struc_parameters['c'],
                                              alpha=90, beta=90, gamma=90)
        elif self.sg in list(range(143, 194 + 1)):
            lattice = Lattice.from_parameters(a=struc_parameters['a'], b=struc_parameters['a'], c=struc_parameters['c'],
                                              alpha=90, beta=90, gamma=120)
        elif self.sg in list(range(195, 230 + 1)):
            lattice = Lattice.from_parameters(a=struc_parameters['a'], b=struc_parameters['a'], c=struc_parameters['a'],
                                              alpha=90, beta=90, gamma=90)
        else:
            lattice = Lattice.from_parameters(a=struc_parameters['a'], b=struc_parameters['b'], c=struc_parameters['c'],
                                              alpha=struc_parameters['alpha'], beta=struc_parameters['beta'], gamma=struc_parameters['gamma'])
        
        structure = Structure(lattice, self.all_atoms, atom_positions)
        return structure
        

    def atomic_dist_and_volume_limit(self, struc: Structure):
        atom_radii = []
        # 单位 埃
        for element in self.all_atoms:
            atom_radii.append(element.van_der_waals_radius)

        # 原子半径和判据
        for i in range(self.total_atom_count - 1):
            for j in range(i + 1, self.total_atom_count):
                if struc.get_distance(i, j) < (atom_radii[i] + atom_radii[j]) * 0.5:
                    raise Exception()
        
        # 原子半径和   
        atom_volume = [4.0 * np.pi * r ** 3 / 3.0 for r in atom_radii]
        sum_atom_volume = sum(atom_volume) 
        
        # 原子体积和判据
        if not (sum_atom_volume * 0.5 <= struc.volume <= sum_atom_volume * 1.5):
            raise Exception()
        
        # self.vacuum_size_limit(struc=struc.copy(), max_size=7.0)
    
    def hyperopt_initial(self, index):
        algorithm = self.config['program']['algorithm'][index]
        n_init = self.config['program']['n_init'][index]
        max_step = self.config['program']['max_step'][index]
        rand_seed = self.config['global']['rand_seed']
        if algorithm == 'rs':
            print('using Random Search ...')
            algo = hy.rand.suggest
        else:
            print('using Bayesian Optimization ...')
            algo = hy.partial(hy.tpe.suggest, n_startup_jobs=n_init)
        rand_seed = np.random.seed(rand_seed)
       
        return algo, max_step, rand_seed
    
    def sko_initial(self, index, func, l_b, u_b):
        algorithm = self.config['program']['algorithm'][index]
        params = self.config['program']['oa_params']
        max_step = self.config['program']['max_step'][index]
        if algorithm == 'pso':
            print('using Particle Swarm Optimization ...')
            algo = PSO(func=func, dim=len(l_b), lb=l_b, ub=u_b, max_iter=max_step, **params)
        else:
            print('using Genetic algorithm ...')
            algo = GA(func=func, n_dim=len(l_b), lb=l_b, ub=u_b, max_iter=max_step, **params)
        return algo

def main():
    # data_list = ['Na4I4', 'Cs4I4', 'Rb4I4', 'La4Te4', 'K8Se4']
    # sg_list = [141, 225, 225, 225, 225, 225, 225, 225, 225, 225, 225, 225]
    ##Open provided config file
    assert os.path.exists('./GN_OA/config.yml'), "Config file was not found!"
    with open('./GN_OA/config.yml', "r") as ymlfile:
        config = yaml.load(ymlfile, Loader=yaml.FullLoader) 
    # for index in range(len(data_list)):
    #     config['global']['composition'] = data_list[index]
        
    #     # my_print("Settings")
    #     print(yaml.dump(config, sort_keys=False, default_flow_style=False, indent=4))
    #     csp = PredictStructure(config)
    csp = PredictStructure(config)


if __name__ == '__main__':
    main()
    # csp = PredictStructure(input_file_path='./GN_OA/gnoa.in')
