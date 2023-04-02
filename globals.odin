package main

import "core:sync"

import vk "vendor:vulkan"

ENABLE_VALIDATION_LAYERS :: #config(EnableValidationLayers, false)
MAX_FRAMES_IN_FLIGHT :: 2

MODEL_PATH :: "./models/shiba.obj"
TEXTURE_PATH :: "./textures/shiba.png"

when ODIN_DEBUG || ENABLE_VALIDATION_LAYERS {
	enable_validation_layers := true
} else {
	enable_validation_layers := false
}

Vec2 :: [2]f32
Vec3 :: [3]f32
Mat4 :: matrix[4, 4]f32
UniformBufferObject :: struct {
	model, view, proj: Mat4,
}

BufferMetaData :: struct {
	size: vk.DeviceSize,
	offset: vk.DeviceSize,
	element_size: vk.DeviceSize,
}

minimized_sync: sync.Atomic_Cond
minimized_sync_mut: sync.Atomic_Mutex

validation_layers := make_validation_layers()
device_extensions := make_device_extensions()

fragment_shader :: #load("./shaders/frag.spv")
vertex_shader :: #load("./shaders/vert.spv")
compute_shader :: #load("./shaders/comp.spv")

GLFW_FALSE :: 0

WIDTH :: 800
HEIGHT :: 600

GOL := [16*16]i32{
	0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
	0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
	0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
	0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
	0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
	0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
	0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0,
	0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0,
	0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
	0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
	0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
	0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
	0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
	0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
	0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
	0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
}