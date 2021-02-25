import os
import pickle
import torch
import numpy as np
import pickle
from .voxel_vae import VAE,Encoder
import trimesh
import pytorch3d
import trimesh


class MeshRetrieval:

    def __init__(self,
        embeddings_path:str,
        class_mappings_path:str,
        model_path:str,
        shape_net_path:str,
        device):

        self.shape_net_path = shape_net_path
        self.device = device

        # load encoder via state_dict
        state_dict = torch.load(model_path)
        ae = VAE()
        ae.load_state_dict(state_dict)
        ae.eval()
        self.model = ae.encoder.to(self.device)

        with open(class_mappings_path,"rb") as f:
            self.class_mappings = pickle.load(f)
        
        self.mesh_embeddings = torch.from_numpy(np.load(embeddings_path)).to(self.device)


    def get_embeddings_for_class(self,class_name):
        if class_name in self.class_mappings:
            start,end = self.class_mappings[class_name]["indices"]
            return self.mesh_embeddings[start:end]
        else:
            return self.mesh_embeddings

    def get_mesh(self,class_name,index):
        if class_name in self.class_mappings:
            file = self.class_mappings[class_name]["files"][index.item()]
        else:
            for k,v in self.class_mappings.items():
                if v["indices"][0]<=index and v["indices"][1]>index:
                    file = v["files"][index.item()-v["indices"][0]]
        base,_ = os.path.split(file)
        mesh_file = os.path.join(self.shape_net_path,base,"models","model_normalized.obj")
        with open(mesh_file,"r") as f:
            data = trimesh.exchange.obj.load_obj(f,skip_materials=True)
        if "geometry" in data:
            mesh = None
            for model in data["geometry"]:
                m = data["geometry"][model]
                new_mesh = trimesh.Trimesh(m["vertices"],m["faces"])
                if mesh is None:
                    mesh = new_mesh
                else:
                    mesh = trimesh.util.concatenate(mesh,new_mesh)
        else:
            mesh = trimesh.Trimesh(data["vertices"],data["faces"])
        return mesh

        

    def find_closest(self,trimesh_mesh,class_name):    
        random_rotations = process_ply(trimesh_mesh)
        model_embeddings = self.model(random_rotations.to(self.device))[0]
        
        class_embeddings = self.get_embeddings_for_class(class_name)

        best = None
        best_distance = 100
        best_query = -1

        for i,sample in enumerate(model_embeddings):
            distances = torch.sqrt(torch.sum((sample-class_embeddings.to(self.device))**2,dim=1))
            sorted_indices = distances.argsort()[0]
            if distances[sorted_indices] < best_distance:
                best = sorted_indices
                best_distance = distances[sorted_indices] 
                best_query = i
                
        return self.get_mesh(class_name,best)

def normalize_mesh(mesh,max_size=32):
    size = mesh.bounds[1]-mesh.bounds[0]
    scale = (max_size-1)/size.max()
    
    # create transformation matrix to move mesh into positive space and scale to max_size
    t_mat = np.eye(4)*scale
    t_mat[0:3,3]=-mesh.bounds[0]*scale
    
    return mesh.apply_transform(t_mat)

def process_ply(mesh,random_rotations=32):

    # Reproducibility
    torch.manual_seed(0)

    normalized_mesh = normalize_mesh(mesh)

    rot_mats = pytorch3d.transforms.random_rotations(random_rotations).numpy()
    target = torch.zeros((random_rotations,1,32,32,32))
    for i in range(random_rotations):
        rot = np.eye(4)
        rot[:3,:3] = rot_mats[i]
        m_rot = normalized_mesh.apply_transform(rot)
        voxels = m_rot.voxelized(1)
        indices = torch.Tensor(voxels.sparse_indices).long().T
        target[i] = torch.sparse.FloatTensor(
                    indices= indices.clip(0,31),
                    values=torch.ones(indices.shape[1]),
                    size=[32, 32, 32]).to_dense().unsqueeze(0)

 
    return target