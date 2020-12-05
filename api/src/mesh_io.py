import os

from cv2 import imwrite


def write_object(output_path, vertices, faces, mask, det_id=1):
    filename = "obj_{}".format(det_id)
    mask_filepath = os.path.join(output_path, filename + ".png")
    obj_filepath = os.path.join(output_path, filename + ".obj")
    mtl_filepath = os.path.join(output_path, filename + ".mtl")

    imwrite(mask_filepath, mask)

    # Texture material
    with open(obj_filepath, 'w') as file:
        file.write("mtllib {}".format(mtl_filepath.split("/")[-1]) + os.linesep)

        for v in vertices:
            file.write("v {} {} {}".format(v[0], v[1], v[3]) + os.linesep)

        for f in faces:
            file.write("f {} {} {}".format(f[0], f[1], f[3]) + os.linesep)
