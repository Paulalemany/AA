from skl2onnx import to_onnx #esto me da error aunque he instalado los paquetes del enunciado
from onnx2json import convert 
import pickle
import json
import numpy as np
from sklearn.preprocessing import OneHotEncoder

def ExportONNX_JSON_TO_Custom(onnx_json,mlp,verbose=False):
    graphDic = onnx_json["graph"]
    initializer = graphDic["initializer"]
    s= "num_layers:"+str(mlp.n_layers_)+"\n"
    index = 0
    parameterIndex = 0
    for parameter in initializer:
        name = parameter["name"]
        if (verbose): print("Capa ",name)
        if name != "classes" and name != "shape_tensor":
            if (verbose): print("procesando ",name)
            s += "parameter:"+str(parameterIndex)+"\n"
            if (verbose): print(parameter["dims"])
            s += "dims:"+str(parameter["dims"])+"\n"
            if (verbose): print(parameter["name"])
            s += "name:"+str(parameter["name"])+"\n"
            if (verbose): print(parameter["doubleData"])
            s += "values:"+str(parameter["doubleData"])+"\n"
            index = index + 1
            parameterIndex = index // 2
        else:
            print("Esta capa no es interesante ",name)
    return s

def ExportAllformatsMLPSKlearn(mlp,X,picklefileName,onixFileName,jsonFileName,customFileName, verbose = False):
    with open(picklefileName,'wb') as f:
        pickle.dump(mlp,f)
    
    onx = to_onnx(mlp, X[:1])
    with open(onixFileName, "wb") as f:
        f.write(onx.SerializeToString())
    
    onnx_json = convert(input_onnx_file_path=onixFileName,output_json_path=jsonFileName,json_indent=2)
    
    customFormat = ExportONNX_JSON_TO_Custom(onnx_json,mlp)
    with open(customFileName, 'w') as f:
        f.write(customFormat)

def WriteStandardScaler(file,mean,var):
    line = ""
    for i in range(0,len(mean)-1):
        line = line + str(mean[i]) + ","
    line = line + str(mean[len(mean)-1])+ "\n"
    for i in range(0,len(var)-1):
        line = line + str(var[i]) + ","
    line = line + str(var[len(var)-1])+ "\n"
    with open(file, 'w') as f:
        f.write(line) 

def one_hot_encoding(Y):
    Y = Y.reshape(-1, 1)
    encoder = OneHotEncoder(sparse_output=False)
    YEnc = encoder.fit_transform(Y)
    return YEnc

def accuracy(P,Y):
    # TP = np.sum(P == Y)
    # totalP = len(Y)
    # return TP / totalP
    P = np.array(P).flatten()
    Y = np.array(Y).flatten()
    return np.mean(P == Y)

def ExportRandomForest(rf, output_file):
    forest = []
    
    for tree in rf.estimators_:
        t = tree.tree_
        forest.append({
            "children_left": t.children_left.tolist(),
            "children_right": t.children_right.tolist(),
            "feature": t.feature.tolist(),
            "threshold": t.threshold.tolist(),
            "value": t.value.squeeze(axis=1).tolist()
        })
    
    with open(output_file, "w") as f:
        json.dump(forest, f)
        
def ExportDecisionTree(dt, output_file):
    t = dt.tree_
    tree_json = {
        "feature": t.feature.tolist(),
        "threshold": t.threshold.tolist(),
        "children_left": t.children_left.tolist(),
        "children_right": t.children_right.tolist(),
        "value": t.value.squeeze(axis=1).tolist()
    }

    with open(output_file, "w") as f:
        json.dump(tree_json, f)