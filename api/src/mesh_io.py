import os

from cv2 import imwrite
from texturing import MeshTextureType


def write_object(output_path, vertices, texturing, mask, det_id=1):
    filename = "obj_{}".format(det_id)
    mask_filepath = os.path.join(output_path, filename + ".png")
    obj_filepath = os.path.join(output_path, filename + ".obj")
    mtl_filepath = os.path.join(output_path, filename + ".mtl")

    imwrite(mask_filepath, mask)

    # obj write
    with open(obj_filepath, 'w') as file:
        file.write("mtllib {}".format(mtl_filepath.split("/")[-1]) + os.linesep)

        for v in vertices:
            file.write("v {} {} {}".format(v[0], v[1], v[3]) + os.linesep)

        for i in range(len(texturing.textures)):
            file.write("usemtl material_{}".format(i) + os.linesep)
            for f in texturing.textures[i].faces:
                file.write("f {} {} {}".format(f[0], f[1], f[3]) + os.linesep)

    # mtl write
    with open(mtl_filepath, 'w') as file:
        for i in range(len(texturing.textures)):
            file.write("newmtl material_{}".format(i) + os.linesep)
            texture_data = texturing.textures[i].texture_data

            if texturing.textures[i].type == MeshTextureType.IMAGE_TEXTURE:
                # TODO: Create image mapping, append to obj file
                file.write("map_Kd {}".format(texture_data[0]) + os.linesep)
            elif texturing.textures[i].type == MeshTextureType.AVG_COLOR:
                file.write("Kd {} {} {}".format(texture_data[0][0], texture_data[0][1],
                                                texture_data[0][2]) + os.linesep)
                file.write("Ks {} {} {}".format(texture_data[1][0], texture_data[1][1],
                                                texture_data[1][2]) + os.linesep)
                file.write("Ns {}".format(texture_data[2]) + os.linesep)
            else:
                # TODO: Add a error log
                pass
