# -*- coding: utf-8 -*-
# ------------------------------------------------------------------------
#  2021/03/17 下午 08:57
#                 _____   _   _   _____   __   _   _____  
#                /  ___| | | | | | ____| |  \ | | /  ___| 
#                | |     | |_| | | |__   |   \| | | |     
#                | |     |  _  | |  __|  | |\   | | |  _  
#                | |___  | | | | | |___  | | \  | | |_| | 
#                \_____| |_| |_| |_____| |_|  \_| \_____/ 
# ------------------------------------------------------------------------

import copy
import itertools
import functools
import os
import random
import time
import logging

import numpy as np
import pandas as pd

# from GN_OA.utils.file_utils import save_data_bin, read_data_bin
from GN_OA.utils.print_utils import print_run_info


class WyckoffPosition(object):
    def __init__(self):
        self.wyckoff_df = pd.read_csv(os.path.split(__file__)[0] + "/wyckoff_list.csv")

    def get_wyckoffs(self, sg, atom_num_list, is_shuffle=False, max_count=500):
        wyckoff_position = eval(self.wyckoff_df["0"][sg])
        if is_shuffle:
            random.shuffle(wyckoff_position)
        wp_part_list = []
        for an in atom_num_list:  
            wp_part = self.find_combinations(int(an), wyckoff_position, max_count)
            if not wp_part:
                return []
            wp_part_list.append(wp_part)

        # 去除重复
        wp_all_list = []
        wp_product = itertools.product(*wp_part_list)
        for p in wp_product:
            pp = [set(k for j in i for k in j) for i in p]
  
            res = (pp[0].intersection(*pp[1:]))
            # 不包含纯数字项
            if False not in [any([s in data for s in ['x', 'y', 'z']]) for data in res]:
                wp_all_list.append(p)
          
        return wp_all_list
    
    @staticmethod
    def find_combinations(target, wyckoff_position, max_count):
        numbers = [len(i) for i in wyckoff_position]
        def backtrack(current_combination, current_subset, current_sum, start_index):
            if current_sum == target:
                # result.append(current_combination[:])
                result1.append(current_subset[:])
                return
            if current_sum > target:
                return
            if len(result1)>max_count:
                raise Exception
            for i in range(start_index, len(numbers)):
                # 已在数据集中并且包含纯数字项
                if wyckoff_position[i] in current_subset and False in [any([s in data for s in ['x', 'y', 'z']]) for data in wyckoff_position[i]]:
                    continue
                current_combination.append(numbers[i])
                current_subset.append(wyckoff_position[i])
                current_sum += numbers[i]
                backtrack(current_combination, current_subset, current_sum, i)
                current_combination.pop()
                current_sum -= numbers[i]
                current_subset.pop()
        # result = []
        result1 = []
        try:
            backtrack([], [], 0, 0)
        except:
            pass
        return result1

    @print_run_info("Get the Wyckoff position combinations")
    def get_all_wyckoff_combination(self, sg_list, atom_num_list, is_shuffle=False, max_count=1E6):
        wyckoffs_dict = {}
        wyckoffs_count = []
        for sg_i in sg_list:
            wp = self.get_wyckoffs(sg_i, atom_num_list, is_shuffle, max_count)   
            
            wyckoffs_dict[sg_i] = wp
            wyckoffs_count.append(len(wp))
        print("total wyckoffs count:{}, max wyckoffs count: {}".format(sum(wyckoffs_count), max(wyckoffs_count)))
        max_wyckoffs_count = max(wyckoffs_count)
        return wyckoffs_dict, max_wyckoffs_count



# class GetWyckoffPosition:
#     def __init__(self,
#                  sg, atom_num,
#                  is_random=True,
#                  # max_count=1,
#                  verbose=False,
#                  save_path='.'
#                  ):
#         start_time = time.time()
#         self.save_path = save_path

#         self.sg = sg
#         self.wyckoff_df = pd.read_csv(os.path.split(__file__)[0] + "/wyckoff_list.csv")

#         # self.pool = Pool(8)

#         self.wyckoffs = []

#         self.get_wyckoffs(atom_num, is_random=is_random)

#         if verbose:
#             end_time = time.time()
#             print(sg, 'OK! time used:', end_time - start_time, 's', 'Count:', len(self.wyckoffs))

#     def get_wyckoffs(self, atom_num, is_random):
#         wyckoff_position = eval(self.wyckoff_df["0"][self.sg])

#         if is_random:
#             self.wyckoffs = self.combination_wp_random(wyckoff_position, atom_num, is_shuffle=True)
#         else:
#             self.wyckoffs = self.combination_wp_all(wyckoff_position, atom_num)

#     def combination_wp_all(self,
#                            wyckoff_position: list,
#                            atom_num: list,
#                            is_fast: bool = True,
#                            max_count: int = 100e4):
#         wp_part_list = []
#         for an in atom_num:
#             part_wp_an_path = os.path.join(self.save_path, 'part_wp', str(an) + '_' + str(self.sg))
#             if os.path.isfile(part_wp_an_path):
#                 wp_part = read_data_bin(None, part_wp_an_path)
#             else:
#                 wp_part = self.combination_wp_part(wyckoff_position, an)
#                 save_data_bin(None, part_wp_an_path, data=wp_part)
#             if not wp_part:
#                 return []
#             wp_part_list.append(wp_part)

#         is_use_fast = False
#         if is_fast:
#             wp_part_len_list = [len(wpp) for wpp in wp_part_list]
#             wp_all_count = functools.reduce(lambda x, y: x * y, wp_part_len_list)
#             if wp_all_count > max_count:
#                 wp_part_list_tmp = []
#                 for wpp in wp_part_list:
#                     random.shuffle(wpp)
#                     wp_part_list_tmp.append(wpp)
#                 wp_part_list = wp_part_list_tmp
#                 is_use_fast = True
#             # print(wp_all_count)

#         wp_all_list = []
#         wp_product = itertools.product(*wp_part_list)
#         for p in wp_product:
#             pp = [set(k for j in i for k in j) for i in p]
#             res = list(pp[0].intersection(*pp[1:]))
#             for ri in res:
#                 if not ('x' in ri or 'y' in ri or 'z' in ri):
#                     p = []
#                     break
#             if p:
#                 wp_all_list.append(p)
#                 if is_use_fast and len(wp_all_list) >= max_count:
#                     return wp_all_list

#         return wp_all_list

#     def combination_wp_random(self,
#                               wyckoff_position: list,
#                               atom_num: list,
#                               max_count: int = 1,
#                               is_shuffle: bool = True):
#         """
#         max_count only one now
#         :param wyckoff_position:
#         :param atom_num:
#         :param max_count:
#         :param is_shuffle:
#         :return:
#         """

#         if max_count and is_shuffle:
#             random.shuffle(wyckoff_position)

#         wp_part_list = []
#         for an in atom_num:
#             # print(len(wyckoff_position))
#             wp_part = self.combination_wp_part(wyckoff_position, an, max_count=max_count)
#             # wp_part = wp_part[0]

#             if not wp_part:
#                 return []

#             wp_part_list.append(wp_part)

#             for wpp in wp_part:
#                 for wp in wpp:
#                     if not ('x' in '_'.join(wp) or 'y' in '_'.join(wp) or 'z' in '_'.join(wp)):
#                         wyckoff_position.remove(wp)

#         wp_all_list = list(itertools.product(*wp_part_list))

#         return wp_all_list

#     @staticmethod
#     def combination_wp_part(wyckoff_position: list,
#                             atom_num_part: int,
#                             max_count: int = -1) -> list:

#         def dfs(target: int,
#                 index: int,
#                 temp: list,
#                 temp_num: list):
#             if sum(temp_num) == de:
#                 temp.sort()
#                 if temp not in result:
#                     result.append(temp)
#                 if len(result) == max_count:  # 达到max_count个组合数，引发异常，结束搜索
#                     raise Exception()

#             for i in range(index, len(wp_num)):
#                 if wp[i] in temp and not ('x' in ','.join(wp[i]) or 'y' in ','.join(wp[i]) or 'z' in ','.join(wp[i])):
#                     continue

#                 if target > wp_num[i]:
#                     dfs(target - wp_num[i], i, temp + [wp[i]], temp_num + [wp_num[i]])

#                 elif target == wp_num[i]:
#                     dfs(target - wp_num[i], i, temp + [wp[i]], temp_num + [wp_num[i]])

#                 elif target < wp_num[i]:
#                     continue

#         result = []
#         _index = 0
#         _temp = []
#         _temp_num = []
#         de = atom_num_part
#         wp = copy.deepcopy(wyckoff_position)
#         wp_num = [len(i) for i in wp]
#         try:
#             dfs(atom_num_part, _index, _temp, _temp_num)
#         except:
#             pass
#         # print(result)

#         return result


# @print_run_info("Get the Wyckoff position combinations")
# def get_all_wyckoff_combination(sg_list, atom_num):
#     current_path = os.path.split(__file__)[0]

#     alphabet = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
#     wyckoff_combination_type = ''.join([alphabet[i] + str(atom_num[i]) for i in range(len(atom_num))])

#     wyckoffs_dict = {}
#     max_wyckoffs_count = 0

#     for sg_i in sg_list:
#         sg_i_path = os.path.join(current_path, 'wp_sg', wyckoff_combination_type + '_' + str(sg_i))
#         if os.path.isfile(sg_i_path):
#             wp = read_data_bin(None, sg_i_path)
#         else:
#             wp = GetWyckoffPosition(sg_i, atom_num, is_random=False, verbose=False, save_path=current_path).wyckoffs
#             save_data_bin(None, sg_i_path, data=wp)
#         wyckoffs_dict[sg_i] = wp
#         max_wyckoffs_count = max(len(wp), max_wyckoffs_count)

#     return wyckoffs_dict, max_wyckoffs_count


