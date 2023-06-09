import bpy
from PIL import Image
import tempfile
import numpy as np
import copy
import torch
import torchvision.transforms as T

# bpy.data.scenes[0].render.engine = "CYCLES"
#
# # Set the device_type
# bpy.context.preferences.addons[
#     "cycles"
# ].preferences.compute_device_type = "CUDA" # or "OPENCL"
#
# # Set the device and feature set
# bpy.context.scene.cycles.device = "GPU"
#
# # get_devices() to let Blender detects GPU device
# bpy.context.preferences.addons["cycles"].preferences.get_devices()
# # print(bpy.context.preferences.addons["cycles"].preferences.compute_device_type)
#
# bpy.context.scene.render.engine = 'CYCLES'
# bpy.context.scene.cycles.device = 'GPU'

to_pil = T.ToPILImage()


def to_image(image: torch.Tensor, name: str):
    # setup PIL image conversion
    pil_image = to_pil(image)
    # pil_image.show()
    width = pil_image.width
    height = pil_image.height
    byte_to_normalized = 1.0 / 255.0
    # create new image
    bpy_image = bpy.data.images.new(name, width=width, height=height, alpha=False)

    # convert Image 'L' to 'RGBA', normalize then flatten
    bpy_image.pixels.foreach_set((np.asarray(pil_image.convert('RGBA'),
                                             dtype=np.float32)
                                  * byte_to_normalized).ravel())
    bpy_image.pack()
    return bpy_image


def get_rendered_material(material_tensor: torch.Tensor):

    # Load the texture maps
    tensors = torch.chunk(material_tensor, 4, dim=0)

    diffuse_map = to_image(tensors[1], "diffuse")
    roughness_map = to_image(tensors[2], 'roughness')
    specular_map = to_image(tensors[3], "specular")
    normal_map = to_image(tensors[0], "normal")

    for area in bpy.context.screen.areas:
        if area.type == 'VIEW_3D':
            for space in area.spaces:
                if space.type == 'VIEW_3D':
                    space.shading.type = 'RENDERED'

    bpy.ops.object.delete(use_global=False, confirm=False)
    bpy.ops.mesh.primitive_plane_add(size=2, enter_editmode=False, align='WORLD', location=(0, 0, 0), scale=(1, 1, 1))

    material = bpy.data.materials.new(name='Material')
    material.use_nodes = True
    nodes = material.node_tree.nodes
    bsdf = nodes['Principled BSDF']
    links = material.node_tree.links

    diffuse_n = nodes.new('ShaderNodeTexImage')
    diffuse_n.image = diffuse_map

    normal_n = nodes.new('ShaderNodeTexImage')
    normal_n.image = normal_map
    normal_n.image.colorspace_settings.name = 'Non-Color'
    norm_vector = nodes.new('ShaderNodeNormalMap')
    norm_vector['default_value'] = 10.0
    specular_n = nodes.new('ShaderNodeTexImage')
    specular_n.image = specular_map
    roughness_n = nodes.new('ShaderNodeTexImage')
    roughness_n.image = roughness_map

    links.new(diffuse_n.outputs['Color'], bsdf.inputs['Base Color'])
    links.new(normal_n.outputs['Color'], norm_vector.inputs['Color'])
    links.new(norm_vector.outputs['Normal'], bsdf.inputs['Normal'])
    links.new(specular_n.outputs['Color'], bsdf.inputs['Specular'])
    links.new(roughness_n.outputs['Color'], bsdf.inputs['Roughness'])

    obj = bpy.context.active_object
    obj.data.materials.append(material)

    light_data = bpy.data.lights.new(name="light_2.80", type='POINT')
    light_data.energy = 120
    light = bpy.data.objects.new(name="light_2.80", object_data=light_data)
    bpy.context.collection.objects.link(light)
    bpy.context.view_layer.objects.active = light
    light.location = (0, 0, 1)

    camera = bpy.data.objects['Camera']
    camera.location = (0, 0, 2.75)
    camera.rotation_euler = (0, 0, 0)

    # Render the image
    render = bpy.context.scene.render
    render.resolution_x = 256
    render.resolution_y = 256
    render.image_settings.file_format = 'PNG'

    with tempfile.TemporaryDirectory() as tempdir:
        render.filepath = f'{tempdir}/output.png'
        bpy.ops.render.render(write_still=True)
        rendered_img = copy.deepcopy(Image.open(f'{tempdir}/output.png'))
        rendered_img = T.PILToTensor()(rendered_img.convert('RGB'))
    return rendered_img
