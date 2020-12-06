class Texturing:
    def __init__(self, all_faces, mask):
        self.all_faces = all_faces
        self.mask = mask
        self.textures = []

    def _create_average_color_texture(self):
        avg_color = [0, 0, 0]

        for p in self.mask:
            avg_color += p

        avg_color /= (len(self.mask) * 255)

        # TODO: Put coloring constants in a constants file
        ks = [0.50, 0.50, 0.50]
        ns = [18.00]

        texture_data = [avg_color, ks, ns]
        avg_color_texture = MeshTexture(MeshTextureType.AVG_COLOR, texture_data)

        self.textures.append(avg_color_texture)

    # TODO: Implement image projection texturing
    def _create_image_projection_texture(self):
        return NotImplementedError


class MeshTextureType:
    AVG_COLOR = "avg_color"
    IMAGE_TEXTURE = "image_texture"


class MeshTexture:
    def __init__(self, texture_type, texture_data):
        self.type = texture_type
        self.texture_data = texture_data
