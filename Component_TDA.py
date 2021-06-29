'''
Module developed for applying TDA techniques to connected components of interest in a dataset. This module is specially useful for datasets
with too many points.
'''


import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import gudhi
import pandas as pd
import velour

def balanced_splitter(X, y, size, random_state = 42, shuffle_result = True):
    '''
    Takes umbalanced X and y dataset and returns balanced version of it of approximately size size*len(X)
    '''
    total_size =  len(X)*size
    y_set = list(set(y))
    number_of_elements_class = int(total_size/len(y_set))
    
    X, y = shuffle(X, y, random_state = random_state) #only shuffles the dataset
    
    result_X = []
    result_y = []
    
    for element in y_set:
        X_class = []
        y_class = []
        
        for label_id in range(len(y)):
            label = y[label_id]
            if len(X_class) < number_of_elements_class:
                if label == element:
                    X_class.append(X[label_id])
                    y_class.append(y[label_id])
            else:
                break
                
        result_X += X_class
        result_y += y_class
    
    result_X = np.array(result_X)
    result_y = np.array(result_y)
    
    
    if shuffle_result:
        result_X, result_y = shuffle(result_X, result_y, random_state = random_state) 
    
    return result_X, result_y     




def union_find(lis):
    lis = map(set, lis)
    unions = []
    for item in lis:
        temp = []
        for s in unions:
            if not s.isdisjoint(item):
                item = s.union(item)
            else:
                temp.append(s)
        temp.append(item)
        unions = temp
    return unions



def edges_finder(data, max_len):
    '''
    Build the simplex trees and calculates persistence pairs of form i, [j,k], meaning that i was merged to the
    edge [j,k] (in particular, i was connected to k); and i, [] if i is isolated or is a basis for a 
    connected component. Then builds a dictionary for every vertex i given by 
    i : k if i is merged to [j,k] and i:i if i is not merged.
    '''
    #builds simplex tress
    rips = gudhi.RipsComplex(points = data, max_edge_length = max_len)
    st = rips.create_simplex_tree(max_dimension = 1)
    barcodes =st.persistence(homology_coeff_field = 2)
    
    #calculates persistence pairs
    per_pairs = st.persistence_pairs()
    
    #builds the dictionary
    dic = {}
    for element in range(len(data)):
        for pair in range(len(per_pairs)):
            if per_pairs[pair][0] == [element]:
                if per_pairs[pair][1] != []:
                    dic[element] = per_pairs[pair][1][1]
                else:
                    dic[element] = element
    
    return dic



def connected_calculator(data, max_len):
    dic = edges_finder(data, max_len) 
    
    #applies union find to figure out the disjointed connected components
    list_of_edges = [[key, dic[key]] for key in dic]
    return union_find(list_of_edges)
   
   
   
def component_topology(data, max_len, component, components, dim = 1,topology_len = 1):
    '''
    Plots the Rips filtration of some connected component of the dataset
    
    data input
    max_len in calculating the connected components
    topology_len = 1 Max length in calculating the topology
    dim = 1 Max dimension in calculating the topology
    '''
    new_dataset = [data[i] for i in components[component]] #forms the dataset

   
    rips = gudhi.RipsComplex(points = new_dataset, max_edge_length = topology_len)
    st = rips.create_simplex_tree(max_dimension = dim)
    barcodes = st.persistence(homology_coeff_field = 2)
    fig = plt.figure(figsize=(15,5))
    ax1 = fig.add_subplot(1,2,1); ax2 = fig.add_subplot(1,2,2)
    gudhi.plot_persistence_barcode(barcodes, axes = ax1)
    gudhi.plot_persistence_diagram(barcodes, axes = ax2)
    plt.show()
    
    return None
 
 
 
def DTM(data, max_len, component, components, dim = 1, topology_len = 1, m = 0.1, p = 1):
    '''
    Plots the DTM filtration of some connected component of the dataset
    
    data input
    max_len in calculating the connected components
    topology_len = 1 Max length in calculating the topology
    dim = 1 Max dimension in calculating the topology
    m = 0.1
    p = 1
    '''
    new_dataset = [data[i] for i in components[component]] #forms the dataset
    new_dataset = np.array(new_dataset)
    st_DTM = velour.DTMFiltration(new_dataset, m, p, dimension_max = dim, filtration_max=topology_len)
    velour.PlotPersistenceDiagram(st_DTM)
    
    return None

def MergeTree(X, y, label  = False):
    # Parameters
    infinity_plus = 0.1
    Vertices = range(len(X))
    
    # Compute Rips complex    
    rips = gudhi.RipsComplex(points = X)
    st = rips.create_simplex_tree()
    st.persistence(min_persistence=-1)
    
    # Build merging structure
    merging = {v:None for v in Vertices}
    merging_time = {v:None for v in Vertices}
    for pair in st.persistence_pairs():
        if pair[1] != []:
            merging[pair[0][0]] = min(pair[1])
            merging_time[pair[0][0]] = st.filtration(pair[1])
        else:
            merging[pair[0][0]] = pair[0][0]
    
    infinity = max([merging_time[v] for v in Vertices if merging_time[v]!=None])+infinity_plus
    for v in [v for v in Vertices if merging_time[v]==None]:
        merging_time[v] = infinity
        
#    merging_time = {v:merging_time[v]/infinity for v in merging_time} #normalizing coordinates
    
    # Union find
    merging_union = merging
    continuer = True
    while continuer == True:
        continuer = False
        for v in Vertices:
            w = merging_union[v]
            if merging_time[v] > merging_time[w]:
                merging_union[v] = merging_union[w]
        for v in Vertices:
            w = merging_union[v]
            if merging_time[v] > merging_time[w]:
                continuer = True
    
    # Find indices
    Vertices_sorted = list(dict(sorted(merging_time.items(), key = lambda item: -item[1] )).keys())
    
    index = [v for v in Vertices if merging_union[v]==v]
    for v in Vertices_sorted:
        if v not in index:
            i = next(x for x in range(len(index)) if merging_union[v] == index[x])
        #    index.insert(i+1, v)
            index.insert(i, v)
        
    index = {index[v]:v for v in Vertices}    
    
    #Make color list
    Vertices_permutation=np.array([Vertices[index[v]] for v in Vertices])
   
    # Plot
    fig = plt.figure( figsize=(8,8) )
    if type(y) != type(None):
        plt.scatter(np.array(Vertices)*0,Vertices_permutation, c = y, cmap = "jet", lw = 3)
    else: 
        plt.scatter(np.array(Vertices)*0,Vertices_permutation, lw = 3)
        
    if label == True:
        for v in Vertices:
            plt.text(-0.07, index[v], str(v))
    
    for v in Vertices:
        plt.arrow(0, index[v], merging_time[v], 0, color = 'grey',lw = 1)
    
    for v in Vertices:
        if merging[v] != None:
            plt.arrow(merging_time[v], index[v], 0, -index[v]+index[merging_union[v]], color = 'grey',lw = 1)
            
            
class ConnectedComponent:
    '''
    Calculates topological information on dataset
    X  input data
    split  size of data actually used before split
    y = None If not None, is the dataset targets
    '''
    def __init__(self, X, split, y = None, balanced = False):
        '''
        Splits dataset and creates self.X and self.y objects. If balanced = True, splits the data so that all BPs occur at
        almost the same frequency
        '''
        X = np.array(X)
        if type(y) == type(None): 
            _, X = train_test_split(X, test_size = split, random_state=42)
            self.y = None
            self.X = np.array(X)
        else:
            if balanced == False:
                _, X, _, y = train_test_split(X, y, test_size = split, random_state=42)
                self.y = y
                self.X = np.array(X)
            else:
                y = np.array(y)
                X, y = balanced_splitter(X, y, split)
                self.y = y
                self.X = np.array(X)
        return None
    
    #connected component analysis
    def connnected_components(self, max_len, treshold = None):
        '''
        Calculates connected components at thickening equal max_len. Shows only up to components
        of length treshold, if treshold is not None
        '''
        connected_components_list = connected_calculator(self.X, max_len)
        if treshold != None:
            connected_components_list = [element for element in connected_components_list if len(element) >= treshold]
        return  connected_components_list
    
    def components_lenghts(self, max_len, treshold = None):
        '''
        Gives connected components lenghts and keys for other methods at thickening equal max_len. Shows only up to components
        of length treshold, if treshold is not None
        '''
        components_list = self.connnected_components(max_len = max_len, treshold = treshold)
        components_lenghts_list = [[(key, len(components_list[key])) for key in range(len(components_list))]]
        return components_lenghts_list
    
    def extract_component(self, max_len, component, treshold = None):
        '''
        Gives all points in a certain component of interest.
        '''
        components_list =  self.connnected_components(max_len = max_len, treshold= treshold)
        return [self.X[i] for i in components_list[component]]
    
    #label dependent methods
    def component_label(self, max_len, treshold = None):
        '''
        Gives components and their labels
        '''
        if type(self.y) == type(None):
            raise Exception("Targets not given when assigning object")

        components_list = self.connnected_components(max_len = max_len, treshold = treshold)
        components_labels_list =  [[self.y[i] for i in component] for component in components_list]
        return components_labels_list
    
    def label_percentage(self, max_len, treshold = None):
        '''
        Gives the percentage of each label in all connected components. Returns a list of lits, in which every entry is of
        (index of the component, length of the compont, (label_i, percentage of occurance of label_i in the component))
        '''
        if type(self.y) == type(None):
            raise Exception("Targets not given when assigning object")
        
        label_list  = self.component_label(max_len = max_len, treshold = treshold)
        percent_list = []
        i = 0
        for component in label_list:
            comp_list = [(i),(len(component))]
            i += 1
            for label in set(component):
                label_number = 0
                for element in component:
                    if element == label:
                        label_number += 1
                comp_list.append((label, label_number/len(component)))
            percent_list.append(comp_list)
        return percent_list
    
    def label_percentage_no_index(self, max_len, treshold = None):
        '''
        Gives the percentage of each label in all connected components. Returns a list of lits, in which every entry is of
        (index of the component, length of the compont, (label_i, percentage of occurance of label_i in the component))
        '''
        if type(self.y) == type(None):
            raise Exception("Targets not given when assigning object")
        
        label_list  = self.component_label(max_len = max_len, treshold = treshold)
        percent_list = []
        i = 0
        for component in label_list:
            comp_list = [(len(component))]
            i += 1
            for label in set(component):
                label_number = 0
                for element in component:
                    if element == label:
                        label_number += 1
                comp_list.append((label, label_number/len(component)))
            percent_list.append(comp_list)
        return percent_list
    
    def MergeTree_component(self, max_len, component, treshold = None):
        '''
        Calculates merge trees for a certain connected component, following the same ordering as in component lengths. If 
        targets not provided (y = None), an exception is raised
        '''
        if type(self.y) == type(None):
            raise Exception("Targets not given when assigning object")
            
        components_list =  self.connnected_components(max_len = max_len, treshold= treshold)
        chosen_component = components_list[component]
        X = [self.X[i] for i in chosen_component]
        y = [self.y[i] for i in chosen_component]
        X, y = np.array(X), np.array(y)
        MergeTree(X,y)
        return None
    
    #calculates topological information
    def Rips_component(self, max_len, component, treshold = None, dim = 1, topology_len = 1):
        '''
        Calculates Rips filtration for a certain connected component, following the same ordering as in component lengths
        '''
        components_list =  self.connnected_components(max_len = max_len, treshold= treshold)
        component_topology(self.X, max_len = max_len, component = component, components = components_list,
                           dim = dim, topology_len = topology_len)
        return None
    
    def DTM_component(self, max_len, component, treshold = None, dim = 1, topology_len = 1, m = 0.1, p = 1):
        '''
        Calculates DTM filtration for a certain connected component, following the same ordering as in component lengths
        '''
        components_list =  self.connnected_components(max_len = max_len, treshold= treshold)
        DTM(self.X,  max_len = max_len, component = component, components = components_list,
                           dim = dim, topology_len = topology_len, m = m, p = p)
        return None
    
    #allows for analyzing the important topology of the set created through picking the mean of all connected componets
    def representatives(self, max_len, treshold = None):
        '''
        Crates the dataset of mean points in each connected component cluster at a given thickening value equal to the
        max length
        '''
        components_list =  self.connnected_components(max_len = max_len, treshold = treshold)
        means = []
        for component in components_list:
            X_list = np.array([self.X[i] for i in component])
            means.append(np.mean(X_list, axis = 0))
        return means
    
    def Rips_representatives(self, max_len, treshold = None, dim = 1, topology_len = 1):
        '''
        Calculates Rips filtration of the dataset created by picking mean values of connected component as representatives
        '''
        data = self.representatives(max_len = max_len, treshold = treshold)
        
        rips = gudhi.RipsComplex(points = data, max_edge_length = topology_len)
        st = rips.create_simplex_tree(max_dimension = dim)
        barcodes = st.persistence(homology_coeff_field = 2)
        fig = plt.figure(figsize=(15,5))
        ax1 = fig.add_subplot(1,2,1); ax2 = fig.add_subplot(1,2,2)
        gudhi.plot_persistence_barcode(barcodes, axes = ax1)
        gudhi.plot_persistence_diagram(barcodes, axes = ax2)
        plt.show()
        return None
    
    def DTM_representatives(self, max_len, treshold = None, dim = 1, topology_len = 1, m = 0.1, p = 1):
        '''
        Calculates DTM filtration of the dataset created by picking mean values of connected component as representatives
        '''
        data = self.representatives(max_len = max_len, treshold = treshold)

        st_DTM = velour.DTMFiltration(X, m, p, dimension_max = dim, filtration_max = topology_len)
        velour.PlotPersistenceDiagram(st_DTM)  
        return None
    
    def MergeTree_representatives(self, max_len, treshold = None, label = True):
        data = self.representatives(max_len = max_len, treshold = treshold)        
        MergeTree(data, y = None, label = True)

        return None