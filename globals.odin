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

GOL_Vertices := [4]Vec3{
	{-1, -1, 0}, {1, -1, 0}, {1, 1, 0}, {-1, 1, 0},
}

GOL_Texture_Coords := [4]Vec2{
	{0, 0}, {1, 0}, {1, 1}, {0, 1},
}

GOL_Colors := [4]Vec3{
	{1, 1, 1}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1},
}

GOL_Indices := [6]u32{0, 1, 2, 1, 2, 3}

GOL_GRID_SIZE :: 16

GOL_Single := [GOL_GRID_SIZE * GOL_GRID_SIZE]u8{
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

GOL := expand_gol(&GOL_Single)

expand_gol :: proc(g: ^[GOL_GRID_SIZE * GOL_GRID_SIZE]u8) -> (result: [GOL_GRID_SIZE * GOL_GRID_SIZE][3]i32) {
	for v, i in g {
		result[i] = i32(v)
	}
	return
}