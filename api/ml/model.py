import logging
import os
import torch
import numpy as np
import cv2

# required so that .register() calls are executed in module scope
import meshrcnn.data  # noqa
import meshrcnn.modeling  # noqa
import meshrcnn.utils  # noqa

from meshrcnn.config import get_meshrcnn_cfg_defaults
from meshrcnn.evaluation import transform_meshes_to_camera_coord_system
from trimesh.smoothing import filter_laplacian
from trimesh.exchange.obj import export_obj
from trimesh.exchange.ply import export_ply
from trimesh.base import Trimesh
from trimesh.visual import TextureVisuals
from trimesh.resolvers import FilePathResolver
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.engine.defaults import DefaultPredictor
from pytorch3d.io import save_ply
from PIL import Image

from meshrcnn.modeling.roi_heads.mesh_head import MeshRCNNGraphConvHead, ROI_MESH_HEAD_REGISTRY
from pytorch3d.structures import Meshes

logger = logging.getLogger("model")

try:
    del ROI_MESH_HEAD_REGISTRY._obj_map['MeshRCNNGraphConvHeadWithViz']
except:
    pass


@ROI_MESH_HEAD_REGISTRY.register()
class MeshRCNNGraphConvHeadWithViz(MeshRCNNGraphConvHead):
    def forward(self, x, mesh):
        if x.numel() == 0 or mesh.isempty():
            return [Meshes(verts=[], faces=[])]
        
        save_file = os.path.join('output', 'voxel.ply')
        verts, faces = mesh.get_mesh_verts_faces(0)
        save_ply(save_file, verts.cpu(), faces.cpu())
        
        last_mesh = None
        meshes = []
        vert_feats = None
        for stage in self.stages:
            mesh, vert_feats = stage(x, mesh, vert_feats=vert_feats)
            last_vert_feats = vert_feats
            last_mesh = mesh
            meshes.append(mesh)
      
        return meshes


class MeshRCNNModel(object):
    def __init__(self, cfg, vis_highest_scoring=True, output_dir="./vis"):
        """
        Args:
            cfg (CfgNode):
            vis_highest_scoring (bool): If set to True visualizes only
                                        the highest scoring prediction
        """
        self.metadata = MetadataCatalog.get(cfg.DATASETS.TEST[0])
        self.colors = self.metadata.thing_colors
        self.cat_names = self.metadata.thing_classes

        self.cpu_device = torch.device("cpu")
        self.vis_highest_scoring = vis_highest_scoring
        self.predictor = DefaultPredictor(cfg)

        os.makedirs(output_dir, exist_ok=True)
        self.output_dir = output_dir

    def run_on_image(self, image, focal_length=20.0, highest_only=True):
        """
        Args:
            image (np.ndarray): an image of shape (H, W, C) (in BGR order).
                This is the format used by OpenCV.
            focal_length (float): the focal_length of the image

        Returns:
            predictions (dict): the output of the model.
        """
        predictions = self.predictor(image)
        # Convert image from OpenCV BGR format to Matplotlib RGB format.
        image = image[:, :, ::-1]

        # camera matrix
        imsize = [image.shape[0], image.shape[1]]
        # focal <- focal * image_width / 32
        focal_length = image.shape[1] / 32 * focal_length
        K = [focal_length, image.shape[1] / 2, image.shape[0] / 2]

        if "instances" in predictions:
            instances = predictions["instances"].to(self.cpu_device)
            scores = instances.scores
            boxes = instances.pred_boxes
            labels = instances.pred_classes
            masks = instances.pred_masks
            meshes = Meshes(
                verts=[mesh[0] for mesh in instances.pred_meshes],
                faces=[mesh[1] for mesh in instances.pred_meshes],
            )
            pred_dz = instances.pred_dz[:, 0] * (boxes.tensor[:, 3] - boxes.tensor[:, 1])
            tc = pred_dz.abs().max() + 1.0
            zranges = torch.stack(
                [
                    torch.stack(
                        [
                            tc - tc * pred_dz[i] / 2.0 / focal_length,
                            tc + tc * pred_dz[i] / 2.0 / focal_length,
                        ]
                    )
                    for i in range(len(meshes))
                ],
                dim=0,
            )

            Ks = torch.tensor(K).to(self.cpu_device).view(1, 3).expand(len(meshes), 3)
            meshes = transform_meshes_to_camera_coord_system(
                meshes, boxes.tensor, zranges, Ks, imsize
            )

            if self.vis_highest_scoring:
                det_ids = [scores.argmax().item()]
            else:
                det_ids = range(len(scores))

            for det_id in det_ids:
                self.visualize_prediction(
                    det_id,
                    image,
                    boxes.tensor[det_id],
                    labels[det_id],
                    scores[det_id],
                    masks[det_id],
                    meshes[det_id],
                    K
                )
                if highest_only:
                    break

        return predictions

    def visualize_prediction(
        self, det_id, image, box, label, score, mask, mesh, K, alpha=0.6, dpi=200, smoothing=True
    ):

        mask_color = np.array(self.colors[label], dtype=np.float32)
        cat_name = self.cat_names[label]
        thickness = max([int(np.ceil(0.001 * image.shape[0])), 1])
        box_color = (0, 255, 0)  # '#00ff00', green
        text_color = (218, 227, 218)  # gray

        composite = image.copy().astype(np.float32)

        # overlay mask
        idx = mask.nonzero()
        composite[idx[:, 0], idx[:, 1], :] *= 1.0 - alpha
        composite[idx[:, 0], idx[:, 1], :] += alpha * mask_color

        # overlay box
        (x0, y0, x1, y1) = (int(x + 0.5) for x in box)
        composite = cv2.rectangle(
            composite, (x0, y0), (x1, y1), color=box_color, thickness=thickness
        )
        composite = composite.astype(np.uint8)

        # overlay text
        font_scale = 0.001 * image.shape[0]
        font_thickness = thickness
        font = cv2.FONT_HERSHEY_TRIPLEX
        text = "%s %.3f" % (cat_name, score)
        ((text_w, text_h), _) = cv2.getTextSize(text, font, font_scale, font_thickness)
        # Place text background.
        if x0 + text_w > composite.shape[1]:
            x0 = composite.shape[1] - text_w
        if y0 - int(1.2 * text_h) < 0:
            y0 = int(1.2 * text_h)
        back_topleft = x0, y0 - int(1.3 * text_h)
        back_bottomright = x0 + text_w, y0
        cv2.rectangle(composite, back_topleft, back_bottomright, box_color, -1)
        # Show text
        text_bottomleft = x0, y0 - int(0.2 * text_h)
        cv2.putText(
            composite,
            text,
            text_bottomleft,
            font,
            font_scale,
            text_color,
            thickness=font_thickness,
            lineType=cv2.LINE_AA,
        )

        save_file = os.path.join(self.output_dir, "%d_mask_%s_%.3f.png" % (det_id, cat_name, score))
        cv2.imwrite(save_file, composite[:, :, ::-1])

        basic_texturing_mesh = self.add_texture_to_mesh(mesh, K, image)

        save_file = os.path.join(self.output_dir, "%d_mesh_%s_%.3f.ply" % (det_id, cat_name, score))
        basic_texturing_mesh.export(save_file, encoding='binary',
                                    vertex_normal=basic_texturing_mesh.vertex_normals.tolist())

        uv_texturing_mesh = self.add_uv_texture_to_mesh(mesh, K, image, mask)
        res = FilePathResolver('./output')
        uv_texturing_mesh = export_obj(uv_texturing_mesh, resolver=res)

        file_path = './output/uv_textured.obj'
        with open(file_path, "w") as f:
            f.write(uv_texturing_mesh)

        if smoothing:
            basic_texturing_mesh = filter_laplacian(basic_texturing_mesh)
            smoothing_file = os.path.join(self.output_dir, "with_smoothing.ply")
            basic_texturing_mesh.export(smoothing_file, encoding='binary',
                                        vertex_normal=basic_texturing_mesh.vertex_normals.tolist())

    @staticmethod
    def add_texture_to_mesh(mesh, K, image):
        f, ox, oy = K
        verts, faces = mesh.get_mesh_verts_faces(0)
        verts = verts.tolist()

        pix_pos = []
        for v in verts:
            x, y, z = v
            i = -x * f / z + K[1]
            j = -y * f / z + K[2]
            pix_pos.append([j, i])

        colors = []
        for i in pix_pos:
            try:
                colors.append(list(image[int(i[0])][int(i[1])]) + [1])
            except IndexError:
                print(f"Skipping index: {i}")
                pass

        textured_mesh = Trimesh(vertices=verts, faces=faces, vertex_colors=colors)

        return textured_mesh

    @staticmethod
    def add_uv_texture_to_mesh(mesh, K, image, mask):
        f, ox, oy = K
        verts, faces = mesh.get_mesh_verts_faces(0)
        verts = verts.tolist()
        mesh = Trimesh(vertices=verts, faces=faces)
        mesh = filter_laplacian(mesh)
        verts = mesh.vertices
        faces = mesh.faces

        front_view_faces = MeshRCNNModel.get_front_view_faces(mesh)
        front_view_vertices = np.unique(np.array(front_view_faces))

        uv_map = []
        for vertex_index in range(len(verts)):
            if vertex_index in front_view_vertices:
                x, y, z = verts[vertex_index]
                i = int(-x * f / z + K[1]) / image.shape[1]
                j = int(y * f / z + K[2]) / image.shape[0]
            else:
                i = 0
                j = 0
            uv_map.append([i, j])

        # Average color calculation inside the mask
        color_mask = ~np.repeat(mask[:, :, np.newaxis], 3, axis=2)
        average_color = np.ma.array(image, mask=color_mask).mean(axis=0).mean(axis=0)

        # Assign average color to pixels outside of the mask
        image[:, :, 0] = np.where(~mask, average_color[0], image[:, :, 0])
        image[:, :, 1] = np.where(~mask, average_color[1], image[:, :, 1])
        image[:, :, 2] = np.where(~mask, average_color[2], image[:, :, 2])

        image_ = Image.fromarray(image)
        texture = TextureVisuals(uv=uv_map, image=image_)
        textured_mesh = Trimesh(vertices=verts, faces=faces, visual=texture)

        return textured_mesh

    @staticmethod
    def get_front_view_faces(mesh):
        # This is an implementation of Back-face Culling
        vertices = mesh.vertices
        faces = mesh.faces
        normals = mesh.face_normals

        front_view = []

        for i in range(len(faces)):
            # -V0(.)N > 0, where (.) denotes dot product and V0 is first vertex of the face
            if np.dot(-(vertices[faces[i][0]]), normals[i]) > 1e-5:
                front_view.append(np.array(faces[i]))

        return front_view


def setup_cfg(split_idx=0):
    splits = ["../meshrcnn/meshrcnn_R50.pth", "../meshrcnn/meshrcnn_S2_R50.pth"]
    cfg = get_cfg()
    get_meshrcnn_cfg_defaults(cfg)
    cfg.merge_from_file("../meshrcnn/configs/viz.yaml")
    cfg.merge_from_list(["MODEL.WEIGHTS", splits[split_idx]])
    cfg.freeze()
    return cfg
