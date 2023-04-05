Hello Vulkan
===

This follows along the [Vulkan Tutorial](https://vulkan-tutorial.com/Introduction) up through generating mip-maps (but not multi-sampling). You can load whatever .obj file you want by setting `MODULE_PATH` and `TEXTURE_PATH` in `globals.odin` as there's a (very minimal) object file loader via `obj_loader.odin` which...works well enough for my purposes.

As an example I loaded this [Shiba](https://sketchfab.com/3d-models/shiba-faef9fe5ace445e7b2989d1c1ece361c) by downloading the gltf file and using blender to convert to an .obj file

I did not upload the .obj or .png file here