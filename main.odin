package main

import "core:c"
import "core:fmt"
import "core:strings"
import "core:os"
import "core:mem"
import "core:mem/virtual"
import "core:slice"
import "core:time"
import "core:math"
import "core:math/linalg"
import rt "core:runtime"
import "core:thread"
import "core:sync"
import "core:testing"
import "vendor:glfw"
import stbi "vendor:stb/image"
import vk "vendor:vulkan"

main :: proc() {

	ta: mem.Tracking_Allocator
	mem.tracking_allocator_init(&ta, context.allocator)
	defer mem.tracking_allocator_destroy(&ta)
	defer fmt.printf("%#v\n", ta.allocation_map)
	defer fmt.println("total tmp space used (mb):", f32((cast(^rt.Arena)context.temp_allocator.data).total_used) / 1024.0 / 1024.0)
	alloc := mem.tracking_allocator(&ta)
	
	context.allocator = alloc

	app := init()
	defer cleanup(app)

	extension_count: u32
	result := vk.EnumerateInstanceExtensionProperties(nil, &extension_count, nil)
	fmt.println(result)
	fmt.println(extension_count, "extensions supported")
	fmt.println("extensions:")
	extension_list := make([dynamic]vk.ExtensionProperties, extension_count)
	defer delete(extension_list)

	result = vk.EnumerateInstanceExtensionProperties(
		nil,
		&extension_count,
		raw_data(extension_list),
	)
	for ext in &extension_list {
		fmt.println(
			"name",
			transmute(cstring)raw_data(ext.extensionName[:]),
			"version",
			ext.specVersion,
		)
	}

	t := thread.create(proc(t: ^thread.Thread){
		app := (^Hello_Triangle)(t.user_args[0])
		run_renderer(app)
	})
	t.user_args[0] = app
	defer thread.destroy(t)
	
	thread.start(t)
	run(app)

	fmt.println("Exiting...")
}

GLFW_FALSE :: 0

WIDTH :: 800
HEIGHT :: 600

Hello_Triangle :: struct {
	window:                  glfw.WindowHandle,
	instance:                vk.Instance,
	physical_device:         vk.PhysicalDevice,
	device:                  vk.Device,
	graphics_queue, present_queue: vk.Queue,
	dbg_msgr:                vk.DebugUtilsMessengerEXT,
	surface:                 vk.SurfaceKHR,
	swap_chain:              vk.SwapchainKHR,
	swap_chain_images:       [dynamic]vk.Image,
	swap_chain_image_format: vk.Format,
	swap_chain_extent:       vk.Extent2D,
	swap_chain_image_views:  [dynamic]vk.ImageView,
	swap_chain_frame_buffers: [dynamic]vk.Framebuffer,
	render_pass: vk.RenderPass,
	descriptor_set_layout: vk.DescriptorSetLayout,
	pipeline_layout: vk.PipelineLayout,
	graphics_pipeline: vk.Pipeline,
	mip_levels: u32,
	depth_image, texture_image: vk.Image,
	depth_image_memory, texture_memory: vk.DeviceMemory,
	depth_image_view, texture_image_view: vk.ImageView,
	texture_sampler: vk.Sampler,
	everything_buffer: vk.Buffer, // positions, colors, indices
	everything_memory: vk.DeviceMemory, // positions, colors, indicies
	uniform_buffers: [MAX_FRAMES_IN_FLIGHT]vk.Buffer,
	uniform_memories: [MAX_FRAMES_IN_FLIGHT]vk.DeviceMemory,
	uniform_buffers_mapped: [MAX_FRAMES_IN_FLIGHT]rawptr,
	descriptor_pool: vk.DescriptorPool,
	descriptor_sets: [MAX_FRAMES_IN_FLIGHT]vk.DescriptorSet,
	command_pool: vk.CommandPool,
	command_buffers: [MAX_FRAMES_IN_FLIGHT]vk.CommandBuffer,
	image_available_sems, render_finished_sems: [MAX_FRAMES_IN_FLIGHT]vk.Semaphore,
	inflight_fences: [MAX_FRAMES_IN_FLIGHT]vk.Fence,
	framebuffer_resized: bool,
	current_frame: u32,
	render_object: Obj,
	render_offsets: Vertex_Offsets,
}

Vertex_Offsets :: struct {
	positions: vk.DeviceSize,
	colors: vk.DeviceSize,
	texture_coords: vk.DeviceSize,
	indices: vk.DeviceSize,
}

// terrible but works
should_close: bool = false
out: bool = false

run_renderer :: proc(app: ^Hello_Triangle) {
	for !should_close {
		draw_frame(app)
	}
	out = true
}

run :: proc(app: ^Hello_Triangle) {
	was_minimized := false
	for !glfw.WindowShouldClose(app.window) {
		width, height := glfw.GetFramebufferSize(app.window)
		if width == 0 || height == 0 {
			was_minimized = true
		} else {
			if was_minimized {
				was_minimized = false
				sync.atomic_cond_signal(&minimized_sync)
				fmt.println("waking up renderer thread")
			}
		}
		glfw.WaitEvents()
	}
	should_close = true

	for !out {}
	vk.DeviceWaitIdle(app.device)
}

create_descriptor_set_layout :: proc(app: ^Hello_Triangle) {
	ubo_layout_binding := vk.DescriptorSetLayoutBinding{
		binding = 0,
		descriptorType = .UNIFORM_BUFFER,
		descriptorCount = 1,
		stageFlags = {.VERTEX},
	}

	sampler_layout_binding := vk.DescriptorSetLayoutBinding{
		binding = 1,
		descriptorCount = 1,
		descriptorType = .COMBINED_IMAGE_SAMPLER,
		stageFlags = {.FRAGMENT},
	}

	bindings := []vk.DescriptorSetLayoutBinding{ubo_layout_binding, sampler_layout_binding}
	
	if vk.CreateDescriptorSetLayout(app.device, &vk.DescriptorSetLayoutCreateInfo{
		sType = .DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
		bindingCount = u32(len(bindings)),
		pBindings = raw_data(bindings),
	}, nil, &app.descriptor_set_layout) != .SUCCESS {
		panic("Failed to create descriptor set layout!")
	}
}

// Can be called in a scope to create, submit, and then deallocate a temporary command buffer
// by calling this proc, begin/end does not need to be called. call begin/end separately if
// more flexibility required
@(deferred_in_out = end_single_time_commands)
scoped_single_time_commands :: proc(app: ^Hello_Triangle) -> vk.CommandBuffer {
	return begin_single_time_commands(app)
}


// Creates a temporary command buffer for one time submit / oneshot commands
// to be written to GPU
begin_single_time_commands :: proc(app: ^Hello_Triangle) -> (buffer: vk.CommandBuffer) {
	vk.AllocateCommandBuffers(app.device, &vk.CommandBufferAllocateInfo{
		sType = .COMMAND_BUFFER_ALLOCATE_INFO,
		level = .PRIMARY,
		commandPool = app.command_pool,
		commandBufferCount = 1,
	}, &buffer)

	vk.BeginCommandBuffer(buffer, &vk.CommandBufferBeginInfo{
		sType = .COMMAND_BUFFER_BEGIN_INFO,
		flags = {.ONE_TIME_SUBMIT},
	})
	
	return
}

// Ends the temporary command buffer and submits the commands
end_single_time_commands :: proc(app: ^Hello_Triangle, buffer: vk.CommandBuffer) {
	buffer := buffer

	vk.EndCommandBuffer(buffer)

	vk.QueueSubmit(app.graphics_queue, 1, &vk.SubmitInfo{
		sType = .SUBMIT_INFO,
		commandBufferCount = 1,
		pCommandBuffers = &buffer,
	}, {})
	vk.QueueWaitIdle(app.graphics_queue)

	vk.FreeCommandBuffers(app.device, app.command_pool, 1, &buffer)
}

recreate_swap_chain :: proc(app: ^Hello_Triangle) {
	
	for width, height := glfw.GetFramebufferSize(app.window); width == 0 || height == 0; width, height = glfw.GetFramebufferSize(app.window) {
		fmt.println("Minimized - renderer going to sleep")
		sync.atomic_mutex_lock(&minimized_sync_mut)
		defer sync.atomic_mutex_unlock(&minimized_sync_mut)
		sync.atomic_cond_wait(&minimized_sync, &minimized_sync_mut)
	}

	vk.DeviceWaitIdle(app.device)

	cleanup_swap_chain(app)

	create_swap_chain(app)
	create_image_views(app)
	create_depth_resources(app)
	create_frame_buffers(app)
}

cleanup_swap_chain :: proc(app: ^Hello_Triangle) {
	defer vk.DestroySwapchainKHR(app.device, app.swap_chain, nil)
	defer {
		for image_view in app.swap_chain_image_views {
			vk.DestroyImageView(app.device, image_view, nil)
		}
		clear(&app.swap_chain_image_views)
	}
	defer destroy_depth_resources(app^)
	defer {
		for fb in app.swap_chain_frame_buffers {
			vk.DestroyFramebuffer(app.device, fb, nil)
		}
		clear(&app.swap_chain_frame_buffers)
	}
}

cleanup_swap_chain_destroy :: proc(app: ^Hello_Triangle) {
	cleanup_swap_chain(app)
	delete(app.swap_chain_image_views)
	delete(app.swap_chain_images)
	delete(app.swap_chain_frame_buffers)
}

find_supported_format :: proc(app: ^Hello_Triangle, candidates: []vk.Format, tiling: vk.ImageTiling, features: vk.FormatFeatureFlags) -> vk.Format {
	for candidate in candidates {
		properties: vk.FormatProperties
		vk.GetPhysicalDeviceFormatProperties(app.physical_device, candidate, &properties)

		if tiling == .LINEAR && (properties.linearTilingFeatures & features == features) {
			return candidate
		} else if tiling == .OPTIMAL && (properties.optimalTilingFeatures & features) == features {
			return candidate
		}
	}

	panic("Could not find any supported format!")
}

has_stencil_component :: proc(format: vk.Format) -> bool {
	return format == .D32_SFLOAT_S8_UINT || format == .D24_UNORM_S8_UINT
}

find_depth_format :: proc(app: ^Hello_Triangle) -> vk.Format {
	return find_supported_format(app, []vk.Format{
		.D32_SFLOAT, .D32_SFLOAT_S8_UINT, .D24_UNORM_S8_UINT,
	}, .OPTIMAL, {.DEPTH_STENCIL_ATTACHMENT})
}

create_depth_resources :: proc(app: ^Hello_Triangle) {
	format := find_depth_format(app)
	app.depth_image, app.depth_image_memory = create_image(app, app.swap_chain_extent.width, app.swap_chain_extent.height, 1, format, .OPTIMAL, {.DEPTH_STENCIL_ATTACHMENT}, {.DEVICE_LOCAL})
	app.depth_image_view = create_image_view(app, app.depth_image, format, {.DEPTH}, 1)
}

destroy_depth_resources :: proc(app: Hello_Triangle) {
	defer destroy_image(app, app.depth_image, app.depth_image_memory)
	defer vk.DestroyImageView(app.device, app.depth_image_view, nil)
}

Image :: struct {
	pixels: []byte,
	width, height, channels, mip_levels: i32,
}

load_image :: proc(path: string) -> (ret: Image) {
	RGB_ALPHA :: 4 // from stb_image.h
	fname := strings.clone_to_cstring(path, context.temp_allocator)
	pixels := stbi.load(fname, &ret.width, &ret.height, &ret.channels, RGB_ALPHA)
	image_size := ret.width * ret.height * 4
	ret.mip_levels = i32(math.floor(math.log2(f32(max(ret.width, ret.height))))) + 1

	if pixels == nil {
		panic("Failed to load texture image!")
	}

	ret.pixels = pixels[:image_size]
	return
}

free_image :: proc(image: Image) {
	stbi.image_free(raw_data(image.pixels))
}

generate_mipmaps :: proc(app: ^Hello_Triangle, image: vk.Image, format: vk.Format, tex_width, tex_height: i32, mip_levels: u32) {
	command_buffer := scoped_single_time_commands(app)

	format_properties: vk.FormatProperties
	vk.GetPhysicalDeviceFormatProperties(app.physical_device, format, &format_properties)

	if format_properties.optimalTilingFeatures & {.SAMPLED_IMAGE_FILTER_LINEAR} == nil {
		panic("Texture image format does not support linear blitting!")
	}

	barrier := vk.ImageMemoryBarrier{
		sType = .IMAGE_MEMORY_BARRIER,
		image = image,
		srcQueueFamilyIndex = vk.QUEUE_FAMILY_IGNORED,
		dstQueueFamilyIndex = vk.QUEUE_FAMILY_IGNORED,
		subresourceRange = {
			aspectMask = {.COLOR},
			baseArrayLayer = 0,
			layerCount = 1,
			levelCount = 1,
		},
	}

	mip_width, mip_height := tex_width, tex_height
	
	for i in 1..<mip_levels {
		barrier.subresourceRange.baseMipLevel = i - 1
		barrier.oldLayout = .TRANSFER_DST_OPTIMAL
		barrier.newLayout = .TRANSFER_SRC_OPTIMAL
		barrier.srcAccessMask = {.TRANSFER_WRITE}
		barrier.dstAccessMask = {.TRANSFER_READ}
		vk.CmdPipelineBarrier(command_buffer, {.TRANSFER}, {.TRANSFER}, nil, 0, nil, 0, nil, 1, &barrier)

		vk.CmdBlitImage(command_buffer, image, .TRANSFER_SRC_OPTIMAL, image, .TRANSFER_DST_OPTIMAL, 1, &vk.ImageBlit{
			srcOffsets = {0 = {0, 0, 0}, 1 = {mip_width, mip_height, 1},},
			srcSubresource = {
				aspectMask = {.COLOR},
				mipLevel = i - 1,
				baseArrayLayer = 0,
				layerCount = 1,
			},
			dstOffsets = {0 = {0, 0, 0}, 1 = {mip_width / 2 if mip_width > 1 else 1, mip_height / 2 if mip_height > 1 else 1, 1}},
			dstSubresource = {
				aspectMask = {.COLOR},
				mipLevel = i,
				baseArrayLayer = 0,
				layerCount = 1,
			},
		}, .LINEAR)

		barrier.oldLayout = .TRANSFER_SRC_OPTIMAL
		barrier.newLayout = .SHADER_READ_ONLY_OPTIMAL
		barrier.srcAccessMask = {.TRANSFER_READ}
		barrier.dstAccessMask = {.SHADER_READ}

		vk.CmdPipelineBarrier(command_buffer, {.TRANSFER}, {.FRAGMENT_SHADER}, nil, 0, nil, 0, nil, 1, &barrier)

		if mip_width > 1 do mip_width /= 2
		if mip_height > 1 do mip_height /= 2

	}
	barrier.subresourceRange.baseMipLevel = mip_levels - 1
	barrier.oldLayout = .TRANSFER_DST_OPTIMAL
	barrier.newLayout = .SHADER_READ_ONLY_OPTIMAL
	barrier.srcAccessMask = {.TRANSFER_WRITE}
	barrier.dstAccessMask = {.SHADER_READ}

	vk.CmdPipelineBarrier(command_buffer, {.TRANSFER}, {.FRAGMENT_SHADER}, nil, 0, nil, 0, nil, 1, &barrier)

}

create_texture_image :: proc(app: ^Hello_Triangle) {
	image := load_image(TEXTURE_PATH)
	app.mip_levels = u32(image.mip_levels)
	defer free_image(image)

	staging_buffer, staging_memory := create_buffer(app, vk.DeviceSize(len(image.pixels)), {.TRANSFER_SRC}, {.HOST_VISIBLE, .HOST_COHERENT})
	defer destroy_buffer(app^, staging_buffer, staging_memory)

	data: rawptr
	vk.MapMemory(app.device, staging_memory, 0, vk.DeviceSize(len(image.pixels)), nil, &data)
	mem.copy(data, raw_data(image.pixels), len(image.pixels))
	vk.UnmapMemory(app.device, staging_memory)

	app.texture_image, app.texture_memory = create_image(app, u32(image.width), u32(image.height), app.mip_levels, .R8G8B8A8_SRGB, .OPTIMAL, {.TRANSFER_SRC, .TRANSFER_DST, .SAMPLED}, {.DEVICE_LOCAL})

	transition_image_layout(app, app.texture_image, .R8G8B8A8_SRGB, .UNDEFINED, .TRANSFER_DST_OPTIMAL, app.mip_levels)
	copy_buffer_to_image(app, staging_buffer, app.texture_image, u32(image.width), u32(image.height))
	// transition_image_layout(app, app.texture_image, .R8G8B8A8_SRGB, .TRANSFER_DST_OPTIMAL, .SHADER_READ_ONLY_OPTIMAL, app.mip_levels)
	generate_mipmaps(app, app.texture_image, .R8G8B8A8_SRGB, image.width, image.height, u32(image.mip_levels))

}

create_image :: proc(app: ^Hello_Triangle, width, height, mip_levels: u32, format: vk.Format, tiling: vk.ImageTiling, usage: vk.ImageUsageFlags, properties: vk.MemoryPropertyFlags) -> (image: vk.Image, memory: vk.DeviceMemory) {
	if vk.CreateImage(app.device, &vk.ImageCreateInfo{
		sType = .IMAGE_CREATE_INFO,
		imageType = .D2,
		extent = vk.Extent3D{width = width, height = height, depth = 1},
		mipLevels = mip_levels,
		arrayLayers = 1,
		format = format,
		tiling = tiling,
		initialLayout = .UNDEFINED,
		usage = usage,
		sharingMode = .EXCLUSIVE,
		samples = {._1},
		flags = nil,
	}, nil, &image) != .SUCCESS {
		panic("Failed to create image!")
	}

	mem_requirements: vk.MemoryRequirements
	vk.GetImageMemoryRequirements(app.device, image, &mem_requirements)

	if vk.AllocateMemory(app.device, &vk.MemoryAllocateInfo{
		sType = .MEMORY_ALLOCATE_INFO,
		allocationSize = mem_requirements.size,
		memoryTypeIndex = find_memory_type(app, mem_requirements.memoryTypeBits, properties),
	}, nil, &memory) != .SUCCESS {
		panic("failed to allocate image memory!")
	}

	vk.BindImageMemory(app.device, image, memory, 0)
	return
}

destroy_image :: proc(app: Hello_Triangle, image: vk.Image, image_mem: vk.DeviceMemory) {
	vk.DestroyImage(app.device, image, nil)
	vk.FreeMemory(app.device, image_mem, nil)
}

create_texture_image_view :: proc(app: ^Hello_Triangle) {
	app.texture_image_view = create_image_view(app, app.texture_image, .R8G8B8A8_SRGB, {.COLOR}, app.mip_levels)
}

create_image_view :: proc(app: ^Hello_Triangle, image: vk.Image, format: vk.Format, aspect_flags: vk.ImageAspectFlags, mip_levels: u32) -> (view: vk.ImageView) {
	if vk.CreateImageView(app.device, &vk.ImageViewCreateInfo{
		sType = .IMAGE_VIEW_CREATE_INFO,
		image = image,
		viewType = .D2,
		format = format,
		subresourceRange = {
			aspectMask = aspect_flags,
			baseMipLevel = 0,
			levelCount = mip_levels,
			baseArrayLayer = 0,
			layerCount = 1,
		},
	}, nil, &view) != .SUCCESS {
		panic("Failed to create texture image")
	}
	return
}

create_image_views :: proc(app: ^Hello_Triangle) {
	resize(&app.swap_chain_image_views, len(app.swap_chain_images))
	for image, i in app.swap_chain_images {
		app.swap_chain_image_views[i] = create_image_view(app, image, app.swap_chain_image_format, {.COLOR}, 1)
	}
}

create_texture_sampler :: proc(app: ^Hello_Triangle) {
	props: vk.PhysicalDeviceProperties
	vk.GetPhysicalDeviceProperties(app.physical_device, &props)
	
	sampler_info := vk.SamplerCreateInfo{
		sType = .SAMPLER_CREATE_INFO,
		magFilter = .LINEAR,
		minFilter = .LINEAR,
		addressModeU = .REPEAT,
		addressModeV = .REPEAT,
		addressModeW = .REPEAT,
		anisotropyEnable = true,
		maxAnisotropy = props.limits.maxSamplerAnisotropy,
		borderColor = .INT_OPAQUE_BLACK,
		unnormalizedCoordinates = false,
		compareEnable = false,
		compareOp = .ALWAYS,
		mipmapMode = .LINEAR,
		mipLodBias = 0,
		minLod = 0,
		maxLod = 0,
	}

	if vk.CreateSampler(app.device, &sampler_info, nil, &app.texture_sampler) != .SUCCESS {
		panic("Failed to create texture sampler!")
	}
}

transition_image_layout :: proc(app: ^Hello_Triangle, image: vk.Image, format: vk.Format, old_layout, new_layout: vk.ImageLayout, mip_levels: u32) {
	command_buffer := scoped_single_time_commands(app)
	barrier := vk.ImageMemoryBarrier{
		sType = .IMAGE_MEMORY_BARRIER,
		oldLayout = old_layout,
		newLayout = new_layout,
		srcQueueFamilyIndex = vk.QUEUE_FAMILY_IGNORED,
		dstQueueFamilyIndex = vk.QUEUE_FAMILY_IGNORED,
		image = image,
		subresourceRange = {
			aspectMask = {.COLOR},
			baseMipLevel = 0,
			levelCount = mip_levels,
			baseArrayLayer = 0,
			layerCount = 1,
		},
		srcAccessMask = nil,
		dstAccessMask = nil,
	}

	source_stage, destination_stage: vk.PipelineStageFlags

	if old_layout == .UNDEFINED && new_layout == .TRANSFER_DST_OPTIMAL {
		barrier.srcAccessMask = nil
		barrier.dstAccessMask = nil
		
		source_stage = {.TOP_OF_PIPE}
		destination_stage = {.TRANSFER}
	} else if old_layout == .TRANSFER_DST_OPTIMAL && new_layout == .SHADER_READ_ONLY_OPTIMAL {
		barrier.srcAccessMask = {.TRANSFER_WRITE}
		barrier.dstAccessMask = {.SHADER_READ}

		source_stage = {.TRANSFER}
		destination_stage = {.FRAGMENT_SHADER}
	} else {
		panic("unsupported layout transition!")
	}
	vk.CmdPipelineBarrier(command_buffer, source_stage, destination_stage, {}, 0, nil, 0, nil, 1, &barrier)
}

copy_buffer_to_image :: proc(app: ^Hello_Triangle, buffer: vk.Buffer, image: vk.Image, width, height: u32) {
	command_buffer := scoped_single_time_commands(app)

	region := vk.BufferImageCopy{
		bufferOffset = 0,
		bufferRowLength = 0,
		bufferImageHeight = 0,

		imageSubresource = {
			aspectMask = {.COLOR},
			mipLevel = 0,
			baseArrayLayer = 0,
			layerCount = 1,
		},

		imageOffset = {0, 0, 0},
		imageExtent = {width, height, 1},

	}

	vk.CmdCopyBufferToImage(command_buffer, buffer, image, .TRANSFER_DST_OPTIMAL, 1, &region)
}

create_buffer :: proc(app: ^Hello_Triangle, size: vk.DeviceSize, usage: vk.BufferUsageFlags, properties: vk.MemoryPropertyFlags) -> (buffer: vk.Buffer, memory: vk.DeviceMemory) {
	buffer_info := vk.BufferCreateInfo{
		sType = .BUFFER_CREATE_INFO,
		size = size,
		usage = usage,
		sharingMode = .EXCLUSIVE,
	}

	if vk.CreateBuffer(app.device, &buffer_info, nil, &buffer) != .SUCCESS {
		fmt.panicf("Failed to create buffer: {%v, %v, %v}\n", size, usage, properties)
	}

	mem_requirements: vk.MemoryRequirements
	vk.GetBufferMemoryRequirements(app.device, buffer, &mem_requirements)

	alloc_info := vk.MemoryAllocateInfo{
		sType = .MEMORY_ALLOCATE_INFO,
		allocationSize = mem_requirements.size,
		memoryTypeIndex = find_memory_type(app, mem_requirements.memoryTypeBits, properties),
	}

	if vk.AllocateMemory(app.device, &alloc_info, nil, &memory) != .SUCCESS {
		fmt.panicf("failed to allocate memory for the buffer: {%v, %v, %v}\n", size, usage, properties)
	}

	vk.BindBufferMemory(app.device, buffer, memory, 0)

	return
}

destroy_buffer :: proc(app: Hello_Triangle, buffer: vk.Buffer, memory: vk.DeviceMemory) {
	defer vk.DestroyBuffer(app.device, buffer, nil)
	defer  vk.FreeMemory(app.device, memory, nil)
}

// Generates a CmdCopyBuffer for a single vk.Buffer to a single vk.Buffer. HOWEVER
// it allows multiple regions within each buffer to be copied from/to
// e.g. two vertex buffers can be copied into a single staging buffer
// which is then split out into two regions within a larger gpu local buffer
copy_buffer :: proc(app: ^Hello_Triangle, src, dst: vk.Buffer, copy_infos: []vk.BufferCopy) {
	temp_command_buffer := scoped_single_time_commands(app)

	vk.CmdCopyBuffer(temp_command_buffer, src, dst, u32(len(copy_infos)), raw_data(copy_infos))
}

create_full_buffer :: proc(app: ^Hello_Triangle) {
	// this should be fine since they all have the same alignment
	position_size, color_size, index_size, texture_size := slice_size(app.render_object.vertices[:]), slice_size(app.render_object.colors[:]), slice_size(app.render_object.indices[:]), slice_size(app.render_object.texture_coords[:])
	total_allocation_size := position_size + color_size + index_size + texture_size

	fmt.println("FULL ALLOC:", total_allocation_size)
	app.everything_buffer, app.everything_memory = create_buffer(app, vk.DeviceSize(total_allocation_size), {.TRANSFER_DST, .VERTEX_BUFFER, .INDEX_BUFFER}, {.DEVICE_LOCAL})
}

slice_size :: proc(s: $T/[]$E) -> int {
	return len(s) * size_of(E)
}

/// Note this is just because we can, but the staging buffer has a layout: positions, colors, indices, texture_coords
/// but the everything_buffer has a layout: positions, colors, texture_coords, indices
/// and that works totally fine because we just copy to different offsets in the staging buffer and then just copy
/// from staging offset to everything buffer offset
initialize_buffers :: proc(app: ^Hello_Triangle) {

	position_size, color_size, index_size, texture_size := slice_size(app.render_object.vertices[:]), slice_size(app.render_object.colors[:]), slice_size(app.render_object.indices[:]), slice_size(app.render_object.texture_coords[:])
	staging_position_offset, staging_color_offset, staging_index_offset, staging_texture_offset := 0, position_size, position_size + color_size, position_size + color_size + index_size
	staging_memory_size := position_size + color_size + texture_size + index_size

	staging_buffer, staging_memory := create_buffer(app, vk.DeviceSize(staging_memory_size), {.TRANSFER_SRC}, {.HOST_VISIBLE, .HOST_COHERENT})
	defer destroy_buffer(app^, staging_buffer, staging_memory)

	staging_data: rawptr // full data
	position_data, color_data, index_data, texture_data: rawptr // views

	vk.MapMemory(app.device, staging_memory, 0, vk.DeviceSize(staging_memory_size), nil, &staging_data)
	
	position_data, color_data, index_data, texture_data = 
		rawptr(uintptr(staging_data) + uintptr(staging_position_offset)),
		rawptr(uintptr(staging_data) + uintptr(staging_color_offset)),
		rawptr(uintptr(staging_data) + uintptr(staging_index_offset)),
		rawptr(uintptr(staging_data) + uintptr(staging_texture_offset))
	
	raw_vertices, raw_colors, raw_indices, raw_textures := slice.to_bytes(app.render_object.vertices[:]), slice.to_bytes(app.render_object.colors[:]), slice.to_bytes(app.render_object.indices[:]), slice.to_bytes(app.render_object.texture_coords[:])
	
	mem.copy(position_data, raw_data(raw_vertices), len(raw_vertices))
	mem.copy(color_data, raw_data(raw_colors), len(raw_colors))
	mem.copy(index_data, raw_data(raw_indices), len(raw_indices))
	mem.copy(texture_data, raw_data(raw_textures), len(raw_textures))
	
	vk.UnmapMemory(app.device, staging_memory)

	copy_buffer(app, staging_buffer, app.everything_buffer, []vk.BufferCopy{
		{
			size = vk.DeviceSize(position_size), 
			srcOffset = vk.DeviceSize(staging_position_offset),
			dstOffset = app.render_offsets.positions,
		},
		{
			size = vk.DeviceSize(color_size), 
			srcOffset = vk.DeviceSize(staging_color_offset), 
			dstOffset = app.render_offsets.colors,
		},
		{
			size = vk.DeviceSize(index_size),
			srcOffset = vk.DeviceSize(staging_index_offset),
			dstOffset = app.render_offsets.indices,
		},
		{
			size = vk.DeviceSize(texture_size),
			srcOffset = vk.DeviceSize(staging_texture_offset),
			dstOffset = app.render_offsets.texture_coords,
		},
	})
}

create_uniform_buffers :: proc(app: ^Hello_Triangle) {
	buffer_size := vk.DeviceSize(size_of(UniformBufferObject))

	for i in 0..<MAX_FRAMES_IN_FLIGHT {
		app.uniform_buffers[i], app.uniform_memories[i] = create_buffer(app, buffer_size, {.UNIFORM_BUFFER}, {.HOST_VISIBLE, .HOST_COHERENT})
		vk.MapMemory(app.device, app.uniform_memories[i], 0, buffer_size, nil, &app.uniform_buffers_mapped[i])
	}
}

create_descriptor_pool :: proc(app: ^Hello_Triangle) {

	pool_sizes := []vk.DescriptorPoolSize{
		{
			type = .UNIFORM_BUFFER,
			descriptorCount = u32(MAX_FRAMES_IN_FLIGHT),
		},
		{
			type = .COMBINED_IMAGE_SAMPLER,
			descriptorCount = u32(MAX_FRAMES_IN_FLIGHT),
		},
	}

	if vk.CreateDescriptorPool(app.device, &vk.DescriptorPoolCreateInfo{
		sType = .DESCRIPTOR_POOL_CREATE_INFO,
		poolSizeCount = u32(len(pool_sizes)),
		pPoolSizes = raw_data(pool_sizes),
		maxSets = u32(MAX_FRAMES_IN_FLIGHT),
	}, nil, &app.descriptor_pool) != .SUCCESS {
		panic("Failed to create descriptor pool!")
	}
}

create_descriptor_sets :: proc(app: ^Hello_Triangle) {
	layouts := make([dynamic]vk.DescriptorSetLayout, MAX_FRAMES_IN_FLIGHT, context.temp_allocator)
	layouts[0] = app.descriptor_set_layout
	layouts[1] = app.descriptor_set_layout

	if vk.AllocateDescriptorSets(app.device, &vk.DescriptorSetAllocateInfo{
		sType = .DESCRIPTOR_SET_ALLOCATE_INFO,
		descriptorPool = app.descriptor_pool,
		descriptorSetCount = u32(MAX_FRAMES_IN_FLIGHT),
		pSetLayouts = raw_data(layouts),
	}, raw_data(app.descriptor_sets[:])) != .SUCCESS {
		panic("Failed to allocate descriptor sets!")
	}

	for i in 0..<MAX_FRAMES_IN_FLIGHT {

		descriptor_sets := []vk.WriteDescriptorSet{
			{
				sType = .WRITE_DESCRIPTOR_SET,
				dstSet = app.descriptor_sets[i],
				dstBinding = 0,
				dstArrayElement = 0,
				descriptorType = .UNIFORM_BUFFER,
				descriptorCount = 1,
				pBufferInfo = &vk.DescriptorBufferInfo{
					buffer = app.uniform_buffers[i],
					offset = 0,
					range = size_of(UniformBufferObject),
				},
			},
			{
				sType = .WRITE_DESCRIPTOR_SET,
				dstSet = app.descriptor_sets[i],
				dstBinding = 1,
				dstArrayElement = 0,
				descriptorType = .COMBINED_IMAGE_SAMPLER,
				descriptorCount = 1,
				pImageInfo = &vk.DescriptorImageInfo{
					imageLayout = .SHADER_READ_ONLY_OPTIMAL,
					imageView = app.texture_image_view,
					sampler = app.texture_sampler,
				},
			},
		}
		vk.UpdateDescriptorSets(app.device, u32(len(descriptor_sets)), raw_data(descriptor_sets), 0, nil)
	}
}


find_memory_type :: proc(app: ^Hello_Triangle, type_filter: u32, properties: vk.MemoryPropertyFlags) -> u32 {
	mem_properties: vk.PhysicalDeviceMemoryProperties
	vk.GetPhysicalDeviceMemoryProperties(app.physical_device, &mem_properties)

	for i in 0..<mem_properties.memoryTypeCount {
		if type_filter & (1 << i) != 0 && (mem_properties.memoryTypes[i].propertyFlags & properties == properties) {
			return i
		}
	}

	panic("Failed to find suitable memory type!")

}

get_binding_descriptions :: proc() -> (ret: [3]vk.VertexInputBindingDescription) {
	position_description := &ret[0]
	position_description^ = vk.VertexInputBindingDescription{
		stride = size_of(Vec3),
		inputRate = .VERTEX,
	}

	color_description := &ret[1]
	color_description^ = vk.VertexInputBindingDescription{
		binding = 1,
		stride = size_of(Vec3),
		inputRate = .VERTEX,
	}

	texture_description := &ret[2]
	texture_description^ = vk.VertexInputBindingDescription{
		binding = 2,
		stride = size_of(Vec2),
		inputRate = .VERTEX,
	}

	return
}

get_attribute_descriptions :: proc() -> (ret: [3]vk.VertexInputAttributeDescription) {
	position_attribute, color_attribute, texture_attribute := &ret[0], &ret[1], &ret[2]

	position_attribute^ = vk.VertexInputAttributeDescription{
		binding = 0,
		location = 0,
		format = .R32G32B32_SFLOAT,
	}

	color_attribute^ = vk.VertexInputAttributeDescription{
		binding = 1,
		location = 1,
		format = .R32G32B32_SFLOAT,
	}

	texture_attribute^ = vk.VertexInputAttributeDescription{
		binding = 2,
		location = 2,
		format = .R32G32_SFLOAT,
	}

	return
}

START_TIME := time.now()

// copy of linalg.matrix4_perspective_f32 but with depth range [0, 1] which is required
// for vulkan. (as opposed to linalg's [-1, 1] whihc is how GL works)
matrix4_perspective_f32 :: proc(fovy, aspect, near, far: f32, flip_z_axis := true) -> (m: linalg.Matrix4f32) {
	tan_half_fovy := math.tan(0.5 * fovy)
	m[0, 0] = 1 / (aspect*tan_half_fovy)
	m[1, 1] = 1 / (tan_half_fovy)
	m[2, 2] = far / (far - near)
	m[3, 2] = +1
	m[2, 3] = -(far*near) / (far - near)

	if flip_z_axis {
		m[2] = -m[2]
	}

	return
}

matrix4_perspective :: proc{matrix4_perspective_f32}

update_uniform_buffer :: proc(app: ^Hello_Triangle, current_image: u32) {
	current_time := time.now()
	t := time.diff(START_TIME, current_time)
	ubo := UniformBufferObject{
		model = linalg.matrix4_rotate(f32(time.duration_seconds(t)) * f32(linalg.radians(90.0)), linalg.Vector3f32{0, 0, 1.0}) * linalg.matrix4_rotate(f32(linalg.radians(90.0)), linalg.Vector3f32{1, 0, 0}),
		view = linalg.matrix4_look_at(linalg.Vector3f32{2.0, 2.0, 2.0}, linalg.Vector3f32{0.0, 0.0, 0.0}, linalg.Vector3f32{0.0, 0.0, 1.0}),
		proj = matrix4_perspective(linalg.radians(f32(45.0)), f32(app.swap_chain_extent.width) / f32(app.swap_chain_extent.height), 0.1, 10.0),
	}
	ubo.proj[1, 1] *= -1
	mem.copy(app.uniform_buffers_mapped[current_image], &ubo, size_of(ubo))
}

draw_frame :: proc(app: ^Hello_Triangle) {
	vk.WaitForFences(app.device, 1, &app.inflight_fences[app.current_frame], true, max(u64))

	if app.framebuffer_resized {
		app.framebuffer_resized = false
		recreate_swap_chain(app)
		return
	}

	image_index: u32
	result := vk.AcquireNextImageKHR(app.device, app.swap_chain, max(u64), app.image_available_sems[app.current_frame], {}, &image_index)
	if result == .ERROR_OUT_OF_DATE_KHR {
		recreate_swap_chain(app)
		return
	} else if result != .SUCCESS && result != .SUBOPTIMAL_KHR { // .SUBOPTIMAL_KHR is considered a "success" return
		panic("Failed to acquire swap chain image!")
	}

	vk.ResetFences(app.device, 1, &app.inflight_fences[app.current_frame])

	vk.ResetCommandBuffer(app.command_buffers[app.current_frame], {})

	record_command_buffer(app, app.command_buffers[app.current_frame], int(image_index))

	dst_stage_mask := [1]vk.PipelineStageFlags{{.COLOR_ATTACHMENT_OUTPUT}}

	update_uniform_buffer(app, app.current_frame)

	if result = vk.QueueSubmit(app.graphics_queue, 1, &vk.SubmitInfo{
		sType = .SUBMIT_INFO,
		waitSemaphoreCount = 1,
		pWaitSemaphores = &app.image_available_sems[app.current_frame],
		pWaitDstStageMask = raw_data(dst_stage_mask[:]),
		commandBufferCount = 1,
		pCommandBuffers = &app.command_buffers[app.current_frame],
		signalSemaphoreCount = 1,
		pSignalSemaphores = &app.render_finished_sems[app.current_frame],
	}, app.inflight_fences[app.current_frame]); result != .SUCCESS {
		fmt.panicf("failed to submit draw command buffer! got=%v", result)
	}

	result = vk.QueuePresentKHR(app.present_queue, &vk.PresentInfoKHR{
		sType = .PRESENT_INFO_KHR,
		waitSemaphoreCount = 1,
		pWaitSemaphores = &app.render_finished_sems[app.current_frame],
		swapchainCount = 1,
		pSwapchains = &app.swap_chain,
		pImageIndices = &image_index,
		pResults = nil,
	})

	if result == .ERROR_OUT_OF_DATE_KHR || result == .SUBOPTIMAL_KHR || app.framebuffer_resized {
		app.framebuffer_resized = false
		recreate_swap_chain(app)
	} else if result != .SUCCESS {
		panic("Failed to present the swap chain image!")
	}

	app.current_frame = (app.current_frame + 1) % MAX_FRAMES_IN_FLIGHT

}

framebuffer_resize_callback :: proc "c" (window: glfw.WindowHandle, width, height: i32) {
	app := (^Hello_Triangle)(glfw.GetWindowUserPointer(window))
	app.framebuffer_resized = true
}

global_ctx: rt.Context

init :: proc() -> (app: ^Hello_Triangle) {
	global_ctx = context
	context.user_ptr = &global_ctx

	app = new(Hello_Triangle)

	glfw.Init()
	// NOTE: this has to be called after glfw.Init()
	vk.load_proc_addresses_global(rawptr(glfw.GetInstanceProcAddress))

	glfw.WindowHint(glfw.CLIENT_API, glfw.NO_API)
	window := glfw.CreateWindow(WIDTH, HEIGHT, "Hello Vulkan", nil, nil)
	assert(window != nil, "Window could not be crated")
	app.window = window

	glfw.SetWindowUserPointer(window, app)
	glfw.SetFramebufferSizeCallback(app.window, framebuffer_resize_callback)

	instance, ok := create_instance()
	assert(ok, "Couldn't create vulkan window")
	app.instance = instance

	get_instance_proc_addr :: proc "system" (
		instance: vk.Instance,
		name: cstring,
	) -> vk.ProcVoidFunction {
		f := glfw.GetInstanceProcAddress(instance, name)
		return (vk.ProcVoidFunction)(f)
	}
	vk.GetInstanceProcAddr = get_instance_proc_addr
	vk.load_proc_addresses(app.instance)

	dbg, res := create_debug_messenger(app.instance)
	if res == .ERROR {
		fmt.println("Error creating debug messenger. exiting...")
		os.exit(1)
	} else if res != .SUCCESS {
		fmt.println("Something else happened:", res)
	} else {
		app.dbg_msgr = dbg
	}

	if glfw.CreateWindowSurface(app.instance, app.window, nil, &app.surface) != .SUCCESS {
		panic("Couldn't create surface")
	}

	app.physical_device = pick_physical_device(app^)
	app.device, app.graphics_queue, app.present_queue = create_logical_device(app^)
	create_swap_chain(app)
	create_image_views(app)

	create_render_pass(app)
	create_descriptor_set_layout(app)
	app.pipeline_layout, app.graphics_pipeline = create_graphics_pipeline(app)

	
	app.command_pool = create_command_pool(app)
	create_command_buffers(app)
	
	create_depth_resources(app)
	create_frame_buffers(app)

	load_model(app)
	fmt.printf("%#v\n", app.render_object.indices[:30])
	create_texture_image(app)
	create_texture_image_view(app)
	create_texture_sampler(app)

	create_full_buffer(app)
	initialize_buffers(app)
	create_uniform_buffers(app)
	create_descriptor_pool(app)
	create_descriptor_sets(app)

	create_sync_objects(app)
	return
}

cleanup :: proc(app: ^Hello_Triangle) {
	defer free(app)
	defer glfw.Terminate()
	defer glfw.DestroyWindow(app.window)
	defer vk.DestroyInstance(app.instance, nil)
	defer destroy_debug_messenger(app.instance, app.dbg_msgr, nil)
	defer vk.DestroyDevice(app.device, nil)
	defer vk.DestroySurfaceKHR(app.instance, app.surface, nil)

	defer vk.DestroyRenderPass(app.device, app.render_pass, nil)
	defer vk.DestroyPipelineLayout(app.device, app.pipeline_layout, nil)
	defer vk.DestroyDescriptorSetLayout(app.device, app.descriptor_set_layout, nil)
	defer {
		for i in 0..<MAX_FRAMES_IN_FLIGHT {
			defer vk.DestroyBuffer(app.device, app.uniform_buffers[i], nil)
			defer vk.FreeMemory(app.device, app.uniform_memories[i], nil)
			defer vk.UnmapMemory(app.device, app.uniform_memories[i])
		}
	}
	defer vk.DestroyPipeline(app.device, app.graphics_pipeline, nil)
	defer unload_object(app.render_object)
	defer destroy_image(app^, app.texture_image, app.texture_memory)
	defer vk.DestroyImageView(app.device, app.texture_image_view, nil)
	defer vk.DestroySampler(app.device, app.texture_sampler, nil)
	defer destroy_buffer(app^, app.everything_buffer, app.everything_memory)
	defer vk.DestroyCommandPool(app.device, app.command_pool, nil)
	defer vk.DestroyDescriptorPool(app.device, app.descriptor_pool, nil)
	defer {
		for i in 0..<MAX_FRAMES_IN_FLIGHT {
			defer vk.DestroySemaphore(app.device, app.image_available_sems[i], nil)
			defer vk.DestroySemaphore(app.device, app.render_finished_sems[i], nil)
			defer vk.DestroyFence(app.device, app.inflight_fences[i], nil)
		}
	}
	defer cleanup_swap_chain_destroy(app)
}

load_model :: proc(app: ^Hello_Triangle) {
	model, load_ok := load_object(MODEL_PATH)
	if !load_ok {
		panic("Could not load object model")
	}
	app.render_object = model
	app.render_offsets = {
		positions = 0,
		colors = vk.DeviceSize(slice_size(model.vertices[:])),
		texture_coords = vk.DeviceSize(slice_size(model.vertices[:])) + vk.DeviceSize(slice_size(model.colors[:])),
		indices = vk.DeviceSize(slice_size(model.vertices[:])) + vk.DeviceSize(slice_size(model.colors[:])) + vk.DeviceSize(slice_size(model.texture_coords[:])),
	}
}

create_sync_objects :: proc(app: ^Hello_Triangle) {
	for i in 0..<MAX_FRAMES_IN_FLIGHT {
		s1 := vk.CreateSemaphore(app.device, &vk.SemaphoreCreateInfo{
			sType = .SEMAPHORE_CREATE_INFO,
		}, nil, &app.image_available_sems[i])
		s2 := vk.CreateSemaphore(app.device, &vk.SemaphoreCreateInfo{
			sType = .SEMAPHORE_CREATE_INFO,
		}, nil, &app.render_finished_sems[i])
		fen := vk.CreateFence(app.device, &vk.FenceCreateInfo{
			sType = .FENCE_CREATE_INFO,
			flags = {.SIGNALED},
		}, nil, &app.inflight_fences[i])
	
		if s1 != .SUCCESS || s2 != .SUCCESS || fen != .SUCCESS {
			panic("failed to create sync objects")
		}
	}
}

record_command_buffer :: proc(app: ^Hello_Triangle, buffer: vk.CommandBuffer, image_index: int) {
	if vk.BeginCommandBuffer(buffer, &vk.CommandBufferBeginInfo{
		sType = .COMMAND_BUFFER_BEGIN_INFO,
	}) != .SUCCESS {
		panic("Could not begin recording command buffer!")
	}
	defer {
		if vk.EndCommandBuffer(buffer) != .SUCCESS {
			panic("failed to record command buffer!")
		}
	}

	clear_values := []vk.ClearValue{
		{
			color = vk.ClearColorValue{float32 = {0.0, 0.0, 0.0, 1.0}},
		},
		{
			depthStencil = vk.ClearDepthStencilValue{1.0, 0},
		},
	}

	vk.CmdBeginRenderPass(buffer, &vk.RenderPassBeginInfo{
		sType = .RENDER_PASS_BEGIN_INFO,
		renderPass = app.render_pass,
		framebuffer = app.swap_chain_frame_buffers[image_index],
		renderArea = {
			extent = app.swap_chain_extent,
		},
		clearValueCount = u32(len(clear_values)),
		pClearValues = raw_data(clear_values),
	}, .INLINE)
	defer vk.CmdEndRenderPass(buffer)

	vk.CmdSetViewport(buffer, 0, 1, &vk.Viewport{
		width = f32(app.swap_chain_extent.width),
		height = f32(app.swap_chain_extent.height),
		maxDepth = 1.0,
	})
	vk.CmdSetScissor(buffer, 0, 1, &vk.Rect2D{
		extent = app.swap_chain_extent,
	})


	vk.CmdBindPipeline(buffer, .GRAPHICS, app.graphics_pipeline)
	vertex_buffers := []vk.Buffer{app.everything_buffer, app.everything_buffer, app.everything_buffer}
	offsets := []vk.DeviceSize{app.render_offsets.positions, app.render_offsets.colors, app.render_offsets.texture_coords}
	index_buffer := app.everything_buffer
	vk.CmdBindVertexBuffers(buffer, 0, u32(len(vertex_buffers)), raw_data(vertex_buffers), raw_data(offsets))
	vk.CmdBindIndexBuffer(buffer, index_buffer, app.render_offsets.indices, .UINT32)
	vk.CmdBindDescriptorSets(buffer, .GRAPHICS, app.pipeline_layout, 0, 1, &app.descriptor_sets[app.current_frame], 0, nil)
	vk.CmdDrawIndexed(buffer, u32(len(app.render_object.indices)), 1, 0, 0, 0)

}

create_command_buffers :: proc(app: ^Hello_Triangle) {
	if result := vk.AllocateCommandBuffers(app.device, &vk.CommandBufferAllocateInfo{
		sType = .COMMAND_BUFFER_ALLOCATE_INFO,
		commandPool = app.command_pool,
		level = .PRIMARY,
		commandBufferCount = MAX_FRAMES_IN_FLIGHT,
	}, raw_data(app.command_buffers[:])); result != .SUCCESS {
		panic("Failed to create command buffer!")
	}
	return
}

create_command_pool :: proc(app: ^Hello_Triangle) -> (pool: vk.CommandPool) {
	indices := find_queue_families(app^, app.physical_device)

	if result := vk.CreateCommandPool(app.device, &vk.CommandPoolCreateInfo{
		sType = .COMMAND_POOL_CREATE_INFO,
		flags = {.RESET_COMMAND_BUFFER},
		queueFamilyIndex = indices.graphics_family.(u32),
	}, nil, &pool); result != .SUCCESS {
		panic("Failed to create command pool!")
	}
	return
}

create_frame_buffers :: proc(app: ^Hello_Triangle) {
	resize(&app.swap_chain_frame_buffers, len(app.swap_chain_image_views))
	for view, i in app.swap_chain_image_views {
		attachments := [?]vk.ImageView{view, app.depth_image_view}
		if result := vk.CreateFramebuffer(app.device, &vk.FramebufferCreateInfo{
			sType = .FRAMEBUFFER_CREATE_INFO,
			renderPass = app.render_pass,
			attachmentCount = u32(len(attachments)),
			pAttachments = raw_data(attachments[:]),
			width = app.swap_chain_extent.width,
			height = app.swap_chain_extent.height,
			layers = 1,
		}, nil, &app.swap_chain_frame_buffers[i]); result != .SUCCESS {
			panic("failed to create framebuffer!")
		}
	}
	return
}

create_render_pass :: proc(app: ^Hello_Triangle) {
	color_attachment := vk.AttachmentDescription{
		format = app.swap_chain_image_format,
		samples = {._1},
		loadOp = .CLEAR,
		storeOp = .STORE,
		stencilLoadOp = .DONT_CARE,
		stencilStoreOp = .DONT_CARE,
		initialLayout = .UNDEFINED,
		finalLayout = .PRESENT_SRC_KHR,
	}

	depth_attachment := vk.AttachmentDescription{
		format = find_depth_format(app),
		samples = {._1},
		loadOp = .CLEAR,
		storeOp = .DONT_CARE,
		stencilLoadOp = .DONT_CARE,
		stencilStoreOp = .DONT_CARE,
		initialLayout = .UNDEFINED,
		finalLayout = .DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
	}

	ca_ref := vk.AttachmentReference{
		attachment = 0,
		layout = .ATTACHMENT_OPTIMAL,
	}

	depth_ref := vk.AttachmentReference{
		attachment = 1,
		layout = .DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
	}

	subpass := vk.SubpassDescription{
		pipelineBindPoint = .GRAPHICS,
		colorAttachmentCount = 1,
		pColorAttachments = &ca_ref,
		pDepthStencilAttachment = &depth_ref,
	}

	attachments := []vk.AttachmentDescription{color_attachment, depth_attachment}

	if result := vk.CreateRenderPass(app.device, &vk.RenderPassCreateInfo{
		sType = .RENDER_PASS_CREATE_INFO,
		attachmentCount = u32(len(attachments)),
		pAttachments = raw_data(attachments),
		subpassCount = 1,
		pSubpasses = &subpass,
		dependencyCount = 1,
		pDependencies = &vk.SubpassDependency{
			srcSubpass = vk.SUBPASS_EXTERNAL,
			dstSubpass = 0,
			srcStageMask = {.COLOR_ATTACHMENT_OUTPUT, .EARLY_FRAGMENT_TESTS},
			dstStageMask = {.COLOR_ATTACHMENT_OUTPUT, .EARLY_FRAGMENT_TESTS},
			dstAccessMask = {.COLOR_ATTACHMENT_WRITE, .DEPTH_STENCIL_ATTACHMENT_WRITE},
		},
	}, nil, &app.render_pass); result != .SUCCESS {
		panic("Could not create render pass")
	}
}

create_graphics_pipeline :: proc(app: ^Hello_Triangle) -> (pl: vk.PipelineLayout, pipeline: vk.Pipeline) {
	vert_shader_module := create_shader_module(app, vertex_shader)
	defer vk.DestroyShaderModule(app.device, vert_shader_module, nil)
	frag_shader_module := create_shader_module(app, fragment_shader)
	defer vk.DestroyShaderModule(app.device, frag_shader_module, nil)

	shader_stages := [?]vk.PipelineShaderStageCreateInfo{
		{
			sType = .PIPELINE_SHADER_STAGE_CREATE_INFO,
			stage = {.VERTEX},
			module = vert_shader_module,
			pName = "main",
		},
		{
			sType = .PIPELINE_SHADER_STAGE_CREATE_INFO,
			stage = {.FRAGMENT},
			module = frag_shader_module,
			pName = "main",
		},
	}

	dynamic_states := [?]vk.DynamicState{.VIEWPORT, .SCISSOR}
	dynamic_state := vk.PipelineDynamicStateCreateInfo {
		sType             = .PIPELINE_DYNAMIC_STATE_CREATE_INFO,
		dynamicStateCount = u32(len(dynamic_states)),
		pDynamicStates    = raw_data(dynamic_states[:]),
	}

	binding_descriptions := get_binding_descriptions()
	attribute_descriptions := get_attribute_descriptions()
	
	vertex_input_info := vk.PipelineVertexInputStateCreateInfo {
		sType = .PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
		vertexBindingDescriptionCount = u32(len(binding_descriptions)),
		vertexAttributeDescriptionCount = u32(len(attribute_descriptions)),
		pVertexBindingDescriptions = raw_data(binding_descriptions[:]),
		pVertexAttributeDescriptions = raw_data(attribute_descriptions[:]),
	}

	input_assembly_info := vk.PipelineInputAssemblyStateCreateInfo {
		sType                  = .PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
		topology               = .TRIANGLE_LIST,
		primitiveRestartEnable = false,
	}

	viewport := vk.Viewport {
		width    = f32(app.swap_chain_extent.width),
		height   = f32(app.swap_chain_extent.height),
		maxDepth = 1.0,
	}

	scissor := vk.Rect2D {
		extent = app.swap_chain_extent,
	}

	viewport_state := vk.PipelineViewportStateCreateInfo {
		sType         = .PIPELINE_VIEWPORT_STATE_CREATE_INFO,
		viewportCount = 1,
		scissorCount  = 1,
	}

	rasterizer := vk.PipelineRasterizationStateCreateInfo {
		sType = .PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
		polygonMode = .FILL,
		lineWidth = 1.0,
		cullMode = {.BACK},
		frontFace = .COUNTER_CLOCKWISE,
	}

	multisampling := vk.PipelineMultisampleStateCreateInfo {
		sType = .PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
		rasterizationSamples = {._1},
		minSampleShading = 1.0,
	}

	color_blend_attachment := vk.PipelineColorBlendAttachmentState {
		colorWriteMask = {.R, .G, .B, .A},
		srcColorBlendFactor = .ONE,
		dstColorBlendFactor = .ZERO,
		colorBlendOp = .ADD,
		srcAlphaBlendFactor = .ONE,
		dstAlphaBlendFactor = .ZERO,
		alphaBlendOp = .ADD,
	}

	color_blending := vk.PipelineColorBlendStateCreateInfo {
		sType           = .PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
		logicOp         = .COPY,
		attachmentCount = 1,
		pAttachments    = &color_blend_attachment,
	}

	if result := vk.CreatePipelineLayout(app.device, &vk.PipelineLayoutCreateInfo{
		sType = .PIPELINE_LAYOUT_CREATE_INFO,
		setLayoutCount = 1,
		pSetLayouts = &app.descriptor_set_layout,
	}, nil, &pl); result != .SUCCESS {
		panic("failed ot create pipeline layout!")
	}

	if result := vk.CreateGraphicsPipelines(app.device, {}, 1, &vk.GraphicsPipelineCreateInfo{
		sType = .GRAPHICS_PIPELINE_CREATE_INFO,
		stageCount = 2,
		pStages = raw_data(shader_stages[:]),
		pVertexInputState = &vertex_input_info,
		pInputAssemblyState = &input_assembly_info,
		pViewportState = &viewport_state,
		pRasterizationState = &rasterizer,
		pMultisampleState = &multisampling,
		pColorBlendState = &color_blending,
		pDynamicState = &dynamic_state,
		layout = pl,
		renderPass = app.render_pass,
		subpass = 0,
		basePipelineHandle = {},
		basePipelineIndex = -1,
		pDepthStencilState = &vk.PipelineDepthStencilStateCreateInfo{
			sType = .PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
			depthTestEnable = true,
			depthWriteEnable = true,
			depthCompareOp = .LESS,
			depthBoundsTestEnable = false,
			minDepthBounds = 0.0,
			maxDepthBounds = 1.0,
			stencilTestEnable = false,
			front = {},
			back = {},
		},
	}, nil, &pipeline); result != .SUCCESS {
		panic("failed to create graphics pipeline")
	}

	return
}

create_shader_module :: proc(app: ^Hello_Triangle, code: []byte) -> (sm: vk.ShaderModule) {
	if result := vk.CreateShaderModule(
		   app.device,
		   &vk.ShaderModuleCreateInfo{
			   sType = .SHADER_MODULE_CREATE_INFO,
			   codeSize = len(code),
			   pCode = (^u32)(raw_data(code)),
		   },
		   nil,
		   &sm,
	   ); result != .SUCCESS {
		panic("Failed to create shader module")
	}
	return
}

create_swap_chain :: proc(app: ^Hello_Triangle) {
	swapchain_support := query_swap_chain_support(app^, app.physical_device)
	surface_format := choose_swap_surface_format(swapchain_support.formats[:])
	present_mode := choose_swap_present_mode(swapchain_support.present_modes[:])
	extent := choose_swap_extent(app^, swapchain_support.capabilities)

	image_count := swapchain_support.capabilities.minImageCount + 1
	if swapchain_support.capabilities.maxImageCount > 0 &&
	   image_count > swapchain_support.capabilities.maxImageCount {
		image_count = swapchain_support.capabilities.maxImageCount
	}

	create_info := vk.SwapchainCreateInfoKHR {
		sType = .SWAPCHAIN_CREATE_INFO_KHR,
		surface = app.surface,
		minImageCount = image_count,
		imageFormat = surface_format.format,
		imageColorSpace = surface_format.colorSpace,
		imageExtent = extent,
		imageArrayLayers = 1,
		imageUsage = {.COLOR_ATTACHMENT},
		preTransform = swapchain_support.capabilities.currentTransform,
		compositeAlpha = {.OPAQUE},
		presentMode = present_mode,
		clipped = true,
		oldSwapchain = {},
	}

	indices := find_queue_families(app^, app.physical_device)
	queue_family_indices := [?]u32{indices.graphics_family.(u32), indices.present_family.(u32)}

	if queue_family_indices[0] != queue_family_indices[1] {
		create_info.imageSharingMode = .CONCURRENT
		create_info.queueFamilyIndexCount = 2
		create_info.pQueueFamilyIndices = raw_data(queue_family_indices[:])
	} else {
		create_info.imageSharingMode = .EXCLUSIVE
	}

	if vk.CreateSwapchainKHR(app.device, &create_info, nil, &app.swap_chain) != .SUCCESS {
		panic("Could not create swap chain!")
	}

	sc_image_count: u32
	vk.GetSwapchainImagesKHR(app.device, app.swap_chain, &sc_image_count, nil)
	resize(&app.swap_chain_images, int(sc_image_count))
	vk.GetSwapchainImagesKHR(
		app.device,
		app.swap_chain,
		&sc_image_count,
		raw_data(app.swap_chain_images),
	)

	app.swap_chain_image_format = surface_format.format
	app.swap_chain_extent = extent
}

choose_swap_surface_format :: proc(
	available_formats: []vk.SurfaceFormatKHR,
) -> vk.SurfaceFormatKHR {
	for available_format in available_formats {
		if available_format.format == .B8G8R8A8_SRGB &&
		   available_format.colorSpace == .SRGB_NONLINEAR {
			return available_format
		}
	}
	return available_formats[0]
}

// MAILBOX == go go go go go (burn that cpu)
// FIFO = vsync == go to sleep when frame buffers are full (wake up after vsync clock)
// FIFO for LOW (in this program almost 0) CPU usage (caps 1.4-2%) (depends how much work needs to be done between frames)
// MAILBOX for latency (up to 14% cpu usage in this program now that glfw.WaitEvents() is used)
choose_swap_present_mode :: proc(
	available_present_modes: []vk.PresentModeKHR,
) -> vk.PresentModeKHR {
	// for present_mode in available_present_modes {
		// if present_mode == .MAILBOX do return present_mode
	// }

	return .FIFO
}

choose_swap_extent :: proc(
	app: Hello_Triangle,
	capabilities: vk.SurfaceCapabilitiesKHR,
) -> vk.Extent2D {
	// if capabilities.currentExtent.width != max(u32) {
		// return capabilities.currentExtent
	// } else {
		width, height := glfw.GetFramebufferSize(app.window)

		actual_extent := vk.Extent2D{u32(width), u32(height)}
		actual_extent.width = clamp(
			actual_extent.width,
			capabilities.minImageExtent.width,
			capabilities.maxImageExtent.width,
		)
		actual_extent.height = clamp(
			actual_extent.height,
			capabilities.minImageExtent.height,
			capabilities.maxImageExtent.height,
		)
		return actual_extent
	// }
}

create_logical_device :: proc(
	app: Hello_Triangle,
) -> (
	dev: vk.Device,
	graphics,
	present: vk.Queue,
) {
	indices := find_queue_families(app, app.physical_device)

	queue_set := make(map[u32]u32)
	defer delete(queue_set)
	queue_set[indices.graphics_family.(u32)] = 1
	queue_set[indices.present_family.(u32)] = 1

	queue_priority: f32 = 1.0

	queue_create_infos := make([dynamic]vk.DeviceQueueCreateInfo)
	defer delete(queue_create_infos)

	for queue_family, _ in queue_set {
		queue_create_info: vk.DeviceQueueCreateInfo
		append(
			&queue_create_infos,
			vk.DeviceQueueCreateInfo{
				sType = .DEVICE_QUEUE_CREATE_INFO,
				queueFamilyIndex = queue_family,
				queueCount = 1,
				pQueuePriorities = &queue_priority,
			},
		)
	}

	device_features: vk.PhysicalDeviceFeatures
	device_features.samplerAnisotropy = true

	create_info := vk.DeviceCreateInfo {
		sType                   = .DEVICE_CREATE_INFO,
		pQueueCreateInfos       = raw_data(queue_create_infos),
		queueCreateInfoCount    = u32(len(queue_create_infos)),
		pEnabledFeatures        = &device_features,
		enabledExtensionCount   = u32(len(device_extensions)),
		ppEnabledExtensionNames = raw_data(device_extensions),
	}

	if enable_validation_layers {
		create_info.enabledLayerCount = u32(len(validation_layers))
		create_info.ppEnabledLayerNames = raw_data(validation_layers)
	} else {
		create_info.enabledLayerCount = 0
	}

	if vk.CreateDevice(app.physical_device, &create_info, nil, &dev) != .SUCCESS {
		panic("failed to create logical device!")
	}

	vk.GetDeviceQueue(dev, indices.graphics_family.(u32), 0, &graphics)
	vk.GetDeviceQueue(dev, indices.present_family.(u32), 0, &present)

	return
}

pick_physical_device :: proc(app: Hello_Triangle) -> (pd: vk.PhysicalDevice) {
	device_count: u32
	vk.EnumeratePhysicalDevices(app.instance, &device_count, nil)

	if device_count == 0 {
		panic("Failed to find GPUs with Vulkan Support!")
	}

	devices := make([dynamic]vk.PhysicalDevice, device_count)
	defer delete(devices)
	vk.EnumeratePhysicalDevices(app.instance, &device_count, raw_data(devices))
	for device in devices {
		if is_device_suitable(app, device) {
			pd = device
			break
		}
	}

	if pd == nil {
		panic("Failed to find a suitable GPU!")
	}

	return
}

is_device_suitable :: proc(app: Hello_Triangle, device: vk.PhysicalDevice) -> bool {
	device_props: vk.PhysicalDeviceProperties
	vk.GetPhysicalDeviceProperties(device, &device_props)

	device_features: vk.PhysicalDeviceFeatures
	vk.GetPhysicalDeviceFeatures(device, &device_features)

	indices := find_queue_families(app, device)

	extensions_supported := check_device_extension_support(device)

	swapchain_adequate: bool
	if extensions_supported {
		swapchain_support := query_swap_chain_support(app, device)
		swapchain_adequate =
			len(swapchain_support.formats) != 0 && len(swapchain_support.present_modes) != 0
	}

	return(
		bool(device_features.geometryShader) &&
		is_complete(indices) &&
		extensions_supported &&
		swapchain_adequate &&
		device_features.samplerAnisotropy \
	)
}

Queue_Family_Indices :: struct {
	graphics_family: Maybe(u32),
	present_family:  Maybe(u32),
}

is_complete :: proc(indices: Queue_Family_Indices) -> bool {
	_, has_graphics := indices.graphics_family.(u32)
	_, has_present := indices.present_family.(u32)
	return has_graphics && has_present
}

find_queue_families :: proc(
	app: Hello_Triangle,
	device: vk.PhysicalDevice,
) -> (
	indices: Queue_Family_Indices,
) {
	queue_family_count: u32
	vk.GetPhysicalDeviceQueueFamilyProperties(device, &queue_family_count, nil)

	queue_families := make([dynamic]vk.QueueFamilyProperties, queue_family_count)
	defer delete(queue_families)
	vk.GetPhysicalDeviceQueueFamilyProperties(
		device,
		&queue_family_count,
		raw_data(queue_families),
	)

	for queue_family, i in queue_families {
		if .GRAPHICS in queue_family.queueFlags {
			indices.graphics_family = u32(i)
		}
		present_support: b32
		vk.GetPhysicalDeviceSurfaceSupportKHR(device, u32(i), app.surface, &present_support)
		if (present_support) {
			indices.present_family = u32(i)
		}
	}

	return
}

Swap_Chain_Support_Details :: struct {
	capabilities:  vk.SurfaceCapabilitiesKHR,
	formats:       [dynamic]vk.SurfaceFormatKHR,
	present_modes: [dynamic]vk.PresentModeKHR,
}

query_swap_chain_support :: proc(
	app: Hello_Triangle,
	device: vk.PhysicalDevice,
) -> (
	result: Swap_Chain_Support_Details,
) {
	vk.GetPhysicalDeviceSurfaceCapabilitiesKHR(device, app.surface, &result.capabilities)

	format_count: u32
	vk.GetPhysicalDeviceSurfaceFormatsKHR(device, app.surface, &format_count, nil)

	if format_count != 0 {
		result.formats = make([dynamic]vk.SurfaceFormatKHR, format_count, context.temp_allocator)
		vk.GetPhysicalDeviceSurfaceFormatsKHR(
			device,
			app.surface,
			&format_count,
			raw_data(result.formats),
		)
	}

	present_mode_count: u32
	vk.GetPhysicalDeviceSurfacePresentModesKHR(device, app.surface, &present_mode_count, nil)

	if present_mode_count != 0 {
		result.present_modes = make([dynamic]vk.PresentModeKHR, present_mode_count, context.temp_allocator)
		vk.GetPhysicalDeviceSurfacePresentModesKHR(
			device,
			app.surface,
			&present_mode_count,
			raw_data(result.present_modes),
		)
	}
	return
}

create_instance :: proc() -> (instance: vk.Instance, ok := true) {
	if enable_validation_layers && !check_validation_layer_support() {
		return nil, false
	}

	// Technically ApplicationInfo is OPTIONAL
	app_info := vk.ApplicationInfo {
		sType              = .APPLICATION_INFO,
		pApplicationName   = "Hello Triangle",
		applicationVersion = vk.MAKE_VERSION(1, 0, 0),
		pEngineName        = "No Engine",
		engineVersion      = vk.MAKE_VERSION(1, 0, 0),
		apiVersion         = vk.API_VERSION_1_0,
	}

	// specifies required extensions. Vulkan requires
	// an extension to interface with windowing system since the
	// core is platform agnostic
	// will add the glfw required basics and then any additional
	// we specify
	extensions := get_required_extensions()
	defer delete(extensions)

	// InstanceCreateInfo is REQUIRED
	create_info := vk.InstanceCreateInfo {
		sType                   = .INSTANCE_CREATE_INFO,
		pApplicationInfo        = &app_info,
		enabledExtensionCount   = u32(len(extensions)),
		ppEnabledExtensionNames = raw_data(extensions),
	}

	debug_info: vk.DebugUtilsMessengerCreateInfoEXT
	if enable_validation_layers {
		create_info.enabledLayerCount = u32(len(validation_layers))
		create_info.ppEnabledLayerNames = raw_data(validation_layers)

		// debug_info = create_debug_messenger_info()
		// create_info.pNext = &debug_info
	} else {
		create_info.enabledLayerCount = 0
	}

	if vk.CreateInstance(&create_info, nil, &instance) != .SUCCESS {
		return nil, false
	}
	return
}

Dbg_Result :: enum {
	SUCCESS,
	DISABLED,
	ERROR,
}

create_debug_messenger_info :: proc() -> vk.DebugUtilsMessengerCreateInfoEXT {
	return(
		vk.DebugUtilsMessengerCreateInfoEXT{
			sType = .DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT,
			messageSeverity = {.VERBOSE, .INFO, .WARNING, .ERROR},
			messageType = {.GENERAL, .VALIDATION, .PERFORMANCE},
			pfnUserCallback = debug_callback,
			pUserData = context.user_ptr,
		} \
	)
}

create_debug_messenger :: proc(
	instance: vk.Instance,
) -> (
	dm: vk.DebugUtilsMessengerEXT,
	ok: Dbg_Result = .SUCCESS,
) {
	if !enable_validation_layers {
		ok = .DISABLED
		return
	}

	info := create_debug_messenger_info()
	if vk.CreateDebugUtilsMessengerEXT(instance, &info, nil, &dm) != .SUCCESS {
		ok = .ERROR
		return
	}
	return
}

destroy_debug_messenger :: proc(
	instance: vk.Instance,
	dm: vk.DebugUtilsMessengerEXT,
	alloc_callback: ^vk.AllocationCallbacks,
) {
	if !enable_validation_layers {
		return
	}

	vk.DestroyDebugUtilsMessengerEXT(instance, dm, alloc_callback)
}

as_cstr :: proc(d: $T/[]u8) -> cstring {
	return transmute(cstring)raw_data(d)
}

make_validation_layers :: proc() -> (result: [dynamic]cstring) {
	append(&result, "VK_LAYER_KHRONOS_validation")
	return
}

make_device_extensions :: proc() -> (result: [dynamic]cstring) {
	append(&result, vk.KHR_SWAPCHAIN_EXTENSION_NAME)
	return
}

check_validation_layer_support :: proc() -> bool {
	layer_count: u32
	vk.EnumerateInstanceLayerProperties(&layer_count, nil)

	available_layers := make([dynamic]vk.LayerProperties, layer_count)
	defer delete(available_layers)
	vk.EnumerateInstanceLayerProperties(&layer_count, raw_data(available_layers))

	outer: for layer in validation_layers {
		for available_layer in &available_layers {
			if layer == as_cstr(available_layer.layerName[:]) do continue outer
		}
		return false
	}

	return true
}

check_device_extension_support :: proc(device: vk.PhysicalDevice) -> bool {
	extension_count: u32
	vk.EnumerateDeviceExtensionProperties(device, nil, &extension_count, nil)

	available_extension := make([dynamic]vk.ExtensionProperties, extension_count)
	defer delete(available_extension)
	vk.EnumerateDeviceExtensionProperties(
		device,
		nil,
		&extension_count,
		raw_data(available_extension),
	)

	required_extensions := make(map[string]bool)
	defer delete(required_extensions)

	for ext in device_extensions {
		required_extensions[strings.clone_from_cstring(ext, context.temp_allocator)] = true
	}

	for ext in &available_extension {
		key := transmute(cstring)raw_data(ext.extensionName[:])
		delete_key(&required_extensions, strings.clone_from_cstring(key, context.temp_allocator))
	}

	return len(required_extensions) == 0
}

get_required_extensions :: proc() -> (result: [dynamic]cstring) {

	extensions := glfw.GetRequiredInstanceExtensions()
	append(&result, ..extensions)

	if enable_validation_layers {
		append(&result, vk.EXT_DEBUG_UTILS_EXTENSION_NAME)
	}
	return
}

debug_callback :: proc "system" (
	message_severity: vk.DebugUtilsMessageSeverityFlagsEXT,
	message_type: vk.DebugUtilsMessageTypeFlagsEXT,
	p_callback_data: ^vk.DebugUtilsMessengerCallbackDataEXT,
	p_user_data: rawptr,
) -> b32 {
	// context = (^rt.Context)(p_user_data)^
	context = rt.default_context()
	if message_severity & {.WARNING, .ERROR} != nil {
		fmt.println(message_severity, message_type)
		fmt.println("MESSAGE:", p_callback_data.pMessage)
	}

	return false
}
