#!/usr/bin/python3


import copy as cp


# PROTOTYPE
# prod ( list [ [ string param1_name, list [param1_values] ] , ... ] params )
def prod(params):
    dic = prod_dic(params)
    keys = list(dic.keys())
    keys = keys[::-1]
    L = []
    dic_depth=len(dic[keys[0]])
    for i in range(dic_depth):
        l = {}
        for k in keys:
            l[k]=dic[k][i]
        L.append(l)
    return(L)


def prod_dic(params):
    dic = {}
    for p in params:
        dict_param(dic,p)
    return(dic)

# PROTOTYPE
# the dictionary resutling from the prod_dic function
def nprod(p):
    return(len(list(p.values())[0]))


# PROTOTYPE
# dict_param(  dict {string 'param_name': list [param_values]} dic ,
#              list [string new_param_name, list [new_param_values]] param
#           )
def dict_param(dic,param):
    # Convert arrays in param to list object:
    param[1]=list(param[1])

    n_keys=len(dic.keys())
    if (n_keys==0):
        dic[param[0]]=param[1]
    else:
        keys=list(dic.keys())
        values=cp.deepcopy(list(dic.values()))
        dic0_depth=len(dic[keys[0]])
        dic[param[0]]=[ param[1][0] for dummy in range(dic0_depth) ]
        n_values=len(param[1])
        for i in range (1, n_values):
            for j in range(n_keys):
                dic[keys[j]]+=values[j]
            dic[param[0]] += [ param[1][i] for dummy in range(dic0_depth) ]



def test():
    grid_alpha = [1,2,3]
    grid_beta = [10,100]
    for p in prod([ ['alpha', grid_alpha], ['beta', grid_beta] ]):
        print("alpha = %s\nbeta = %s" %(p['alpha'],p['beta']))
 
