package main

import "core:c"
import "core:fmt"
import "core:strings"
import "core:os"
import "core:mem"
import rt "core:runtime"
import "vendor:glfw"
import vk "vendor:vulkan"

when ODIN_DEBUG || #config(EnableValidationLayers, false) {
	enable_validation_layers := true
} else {
	enable_validation_layers := false
}

validation_layers := make_validation_layers()
device_extensions := make_device_extensions()

fragment_shader :: #load("./shaders/frag.spv")
vertex_shader :: #load("./shaders/vert.spv")

Vec2 :: [2]f32
Vec3 :: [3]f32

positions := []Vec2{
	{0.0, -0.5},
	{0.5, 0.5},
	{-0.5, 0.5},
}

colors := []Vec3{
	{1.0, 0.0, 0.0},
	{0.0, 1.0, 0.0},
	{0.0, 0.0, 1.0},
}

main :: proc() {

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

	run(&app)

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
	pipeline_layout: vk.PipelineLayout,
	graphics_pipeline: vk.Pipeline,
	position_buffer, color_buffer: vk.Buffer,
	position_buffer_memory, color_buffer_memory: vk.DeviceMemory,
	command_pool: vk.CommandPool,
	command_buffer: vk.CommandBuffer,
	image_available_sem, render_finished_sem: vk.Semaphore,
	inflight_fence: vk.Fence,
}

run :: proc(app: ^Hello_Triangle) {
	for !glfw.WindowShouldClose(app.window) {
		glfw.PollEvents()
		draw_frame(app)
	}

	vk.DeviceWaitIdle(app.device)
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
		memoryTypeIndex = find_memory_type(app, mem_requirements.memoryTypeBits, {.HOST_VISIBLE, .HOST_COHERENT}),
	}

	if vk.AllocateMemory(app.device, &alloc_info, nil, &memory) != .SUCCESS {
		fmt.panicf("failed to allocate memory for the buffer: {%v, %v, %v}\n", size, usage, properties)
	}

	vk.BindBufferMemory(app.device, buffer, memory, 0)

	return
}

copy_buffer :: proc(app: ^Hello_Triangle, src, dest: vk.Buffer, size: vk.DeviceSize) {
	temp_command_buffer: vk.CommandBuffer
	vk.AllocateCommandBuffers(app.device, &vk.CommandBufferAllocateInfo{
		sType = .COMMAND_BUFFER_ALLOCATE_INFO,
		level = .PRIMARY,
		commandPool = app.command_pool,
		commandBufferCount = 1,
	}, &temp_command_buffer)
	defer vk.FreeCommandBuffers(app.device, app.command_pool, 1, &temp_command_buffer)

	{
		vk.BeginCommandBuffer(temp_command_buffer, &vk.CommandBufferBeginInfo{
			sType = .COMMAND_BUFFER_BEGIN_INFO,
			flags = {.ONE_TIME_SUBMIT},
		})
		defer vk.EndCommandBuffer(temp_command_buffer)

		vk.CmdCopyBuffer(temp_command_buffer, src, dest, 1, &vk.BufferCopy{
			srcOffset = 0,
			dstOffset = 0,
			size = size,
		})
	}

	vk.QueueSubmit(app.graphics_queue, 1, &vk.SubmitInfo{
		sType = .SUBMIT_INFO,
		commandBufferCount = 1,
		pCommandBuffers = &temp_command_buffer,
	}, {})
	vk.QueueWaitIdle(app.graphics_queue)
}

create_vertex_buffer :: proc(app: ^Hello_Triangle) {

	position_size, color_size := vk.DeviceSize(size_of(Vec2) * len(positions)), vk.DeviceSize(size_of(Vec3) * len(colors))

	staging_position_buffer, staging_position_memory := create_buffer(app, position_size, {.TRANSFER_SRC}, {.HOST_VISIBLE, .HOST_COHERENT})
	defer vk.DestroyBuffer(app.device, staging_position_buffer, nil)
	defer  vk.FreeMemory(app.device, staging_position_memory, nil)

	staging_color_buffer, staging_color_memory := create_buffer(app, color_size, {.TRANSFER_SRC}, {.HOST_VISIBLE, .HOST_COHERENT})
	defer vk.DestroyBuffer(app.device, staging_color_buffer, nil)
	defer vk.FreeMemory(app.device, staging_color_memory, nil)

	position_data, color_data: rawptr
	vk.MapMemory(app.device, staging_position_memory, 0, position_size, nil, &position_data)
	vk.MapMemory(app.device, staging_color_memory, 0, color_size, nil, &color_data)

	mem.copy(position_data, raw_data(positions), int(position_size))
	mem.copy(color_data, raw_data(colors), int(color_size))

	vk.UnmapMemory(app.device, staging_position_memory)
	vk.UnmapMemory(app.device, staging_color_memory)

	app.position_buffer, app.position_buffer_memory = create_buffer(app, position_size, {.VERTEX_BUFFER, .TRANSFER_DST}, {.DEVICE_LOCAL})
	app.color_buffer, app.color_buffer_memory = create_buffer(app, color_size, {.VERTEX_BUFFER, .TRANSFER_DST}, {.DEVICE_LOCAL})

	copy_buffer(app, staging_position_buffer, app.position_buffer, position_size)
	copy_buffer(app, staging_color_buffer, app.color_buffer, color_size)
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

get_binding_descriptions :: proc() -> (ret: [2]vk.VertexInputBindingDescription) {
	position_description := &ret[0]
	position_description^ = vk.VertexInputBindingDescription{
		stride = size_of(Vec2),
		inputRate = .VERTEX,
	}

	color_description := &ret[1]
	color_description^ = vk.VertexInputBindingDescription{
		binding = 1,
		stride = size_of(Vec3),
		inputRate = .VERTEX,
	}
	return
}

get_attribute_descriptions :: proc() -> (ret: [2]vk.VertexInputAttributeDescription) {
	position_attribute, color_attribute := &ret[0], &ret[1]

	position_attribute^ = vk.VertexInputAttributeDescription{
		binding = 0,
		location = 0,
		format = .R32G32_SFLOAT,
	}

	color_attribute^ = vk.VertexInputAttributeDescription{
		binding = 1,
		location = 1,
		format = .R32G32B32_SFLOAT,
	}

	return
}

draw_frame :: proc(app: ^Hello_Triangle) {
	vk.WaitForFences(app.device, 1, &app.inflight_fence, true, max(u64))
	vk.ResetFences(app.device, 1, &app.inflight_fence)

	image_index: u32
	vk.AcquireNextImageKHR(app.device, app.swap_chain, max(u64), app.image_available_sem, {}, &image_index)
	vk.ResetCommandBuffer(app.command_buffer, {})

	record_command_buffer(app, app.command_buffer, int(image_index))

	dst_stage_mask := [1]vk.PipelineStageFlags{{.COLOR_ATTACHMENT_OUTPUT}}
	if vk.QueueSubmit(app.graphics_queue, 1, &vk.SubmitInfo{
		sType = .SUBMIT_INFO,
		waitSemaphoreCount = 1,
		pWaitSemaphores = &app.image_available_sem,
		pWaitDstStageMask = raw_data(dst_stage_mask[:]),
		commandBufferCount = 1,
		pCommandBuffers = &app.command_buffer,
		signalSemaphoreCount = 1,
		pSignalSemaphores = &app.render_finished_sem,
	}, app.inflight_fence) != .SUCCESS {
		panic("failed to submit draw command buffer!")
	}

	vk.QueuePresentKHR(app.present_queue, &vk.PresentInfoKHR{
		sType = .PRESENT_INFO_KHR,
		waitSemaphoreCount = 1,
		pWaitSemaphores = &app.render_finished_sem,
		swapchainCount = 1,
		pSwapchains = &app.swap_chain,
		pImageIndices = &image_index,
		pResults = nil,
	})

}

init :: proc() -> (app: Hello_Triangle) {
	global_ctx := context
	context.user_ptr = &global_ctx

	glfw.Init()
	// NOTE: this has to be called after glfw.Init()
	vk.load_proc_addresses_global(rawptr(glfw.GetInstanceProcAddress))

	glfw.WindowHint(glfw.CLIENT_API, glfw.NO_API)
	glfw.WindowHint(glfw.RESIZABLE, GLFW_FALSE)
	window := glfw.CreateWindow(WIDTH, HEIGHT, "Hello Vulkan", nil, nil)
	assert(window != nil, "Window could not be crated")
	app.window = window

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

	app.physical_device = pick_physical_device(app)
	app.device, app.graphics_queue, app.present_queue = create_logical_device(app)
	app.swap_chain = create_swap_chain(&app)

	app.swap_chain_image_views = make([dynamic]vk.ImageView, len(app.swap_chain_images))
	for image, i in app.swap_chain_images {
		if result := vk.CreateImageView(
			   app.device,
			   &vk.ImageViewCreateInfo{
				   sType = .IMAGE_VIEW_CREATE_INFO,
				   image = image,
				   viewType = .D2,
				   format = app.swap_chain_image_format,
				   components = {r = .IDENTITY, g = .IDENTITY, b = .IDENTITY, a = .IDENTITY},
				   subresourceRange = {
					   aspectMask = {.COLOR},
					   baseMipLevel = 0,
					   levelCount = 1,
					   baseArrayLayer = 0,
					   layerCount = 1,
				   },
			   },
			   nil,
			   &app.swap_chain_image_views[i],
		   ); result != .SUCCESS {
			panic("Could not create image view!")
		}
	}

	create_render_pass(&app)
	app.pipeline_layout, app.graphics_pipeline = create_graphics_pipeline(&app)

	app.swap_chain_frame_buffers = create_frame_buffers(&app)

	app.command_pool = create_command_pool(&app)
	app.command_buffer = create_command_buffer(&app)

	create_vertex_buffer(&app)

	app.image_available_sem, app.render_finished_sem, app.inflight_fence = create_sync_objects(&app)
	return
}

cleanup :: proc(app: Hello_Triangle) {
	defer glfw.Terminate()
	defer glfw.DestroyWindow(app.window)
	defer vk.DestroyInstance(app.instance, nil)
	defer destroy_debug_messenger(app.instance, app.dbg_msgr, nil)
	defer vk.DestroyDevice(app.device, nil)
	defer vk.DestroySurfaceKHR(app.instance, app.surface, nil)
	defer vk.DestroySwapchainKHR(app.device, app.swap_chain, nil)
	defer {
		for image_view in app.swap_chain_image_views {
			vk.DestroyImageView(app.device, image_view, nil)
		}
	}
	defer vk.DestroyRenderPass(app.device, app.render_pass, nil)
	defer vk.DestroyPipelineLayout(app.device, app.pipeline_layout, nil)
	defer vk.DestroyPipeline(app.device, app.graphics_pipeline, nil)
	defer {
		for fb in app.swap_chain_frame_buffers {
			vk.DestroyFramebuffer(app.device, fb, nil)
		}
	}
	defer vk.DestroyBuffer(app.device, app.position_buffer, nil)
	defer vk.DestroyBuffer(app.device, app.color_buffer, nil)
	defer vk.FreeMemory(app.device, app.position_buffer_memory, nil)
	defer vk.FreeMemory(app.device, app.color_buffer_memory, nil)
	defer vk.DestroyCommandPool(app.device, app.command_pool, nil)
	defer vk.DestroySemaphore(app.device, app.image_available_sem, nil)
	defer vk.DestroySemaphore(app.device, app.render_finished_sem, nil)
	defer vk.DestroyFence(app.device, app.inflight_fence, nil)
}

create_sync_objects :: proc(app: ^Hello_Triangle) -> (ias, rfs: vk.Semaphore, iff: vk.Fence) {
	s1 := vk.CreateSemaphore(app.device, &vk.SemaphoreCreateInfo{
		sType = .SEMAPHORE_CREATE_INFO,
	}, nil, &ias)
	s2 := vk.CreateSemaphore(app.device, &vk.SemaphoreCreateInfo{
		sType = .SEMAPHORE_CREATE_INFO,
	}, nil, &rfs)
	fen := vk.CreateFence(app.device, &vk.FenceCreateInfo{
		sType = .FENCE_CREATE_INFO,
		flags = {.SIGNALED},
	}, nil, &iff)

	if s1 != .SUCCESS || s2 != .SUCCESS || fen != .SUCCESS {
		panic("failed to create sync objects")
	}
	return
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

	vk.CmdBeginRenderPass(buffer, &vk.RenderPassBeginInfo{
		sType = .RENDER_PASS_BEGIN_INFO,
		renderPass = app.render_pass,
		framebuffer = app.swap_chain_frame_buffers[image_index],
		renderArea = {
			extent = app.swap_chain_extent,
		},
		clearValueCount = 1,
		pClearValues = &vk.ClearValue{
			color = {float32 = {0.0, 0.0, 0.0, 1.0,}},
		},
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
	vertex_buffers := []vk.Buffer{app.position_buffer, app.color_buffer}
	offsets := []vk.DeviceSize{0, 0}
	vk.CmdBindVertexBuffers(buffer, 0, 2, raw_data(vertex_buffers), raw_data(offsets))	
	vk.CmdDraw(buffer, u32(len(positions)), 1, 0, 0)

}

create_command_buffer :: proc(app: ^Hello_Triangle) -> (buffer: vk.CommandBuffer) {
	if result := vk.AllocateCommandBuffers(app.device, &vk.CommandBufferAllocateInfo{
		sType = .COMMAND_BUFFER_ALLOCATE_INFO,
		commandPool = app.command_pool,
		level = .PRIMARY,
		commandBufferCount = 1,
	}, &buffer); result != .SUCCESS {
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

create_frame_buffers :: proc(app: ^Hello_Triangle) -> (fb: [dynamic]vk.Framebuffer) {
	resize(&fb, len(app.swap_chain_image_views))
	for view, i in app.swap_chain_image_views {
		attachments := [?]vk.ImageView{view}
		if result := vk.CreateFramebuffer(app.device, &vk.FramebufferCreateInfo{
			sType = .FRAMEBUFFER_CREATE_INFO,
			renderPass = app.render_pass,
			attachmentCount = 1,
			pAttachments = raw_data(attachments[:]),
			width = app.swap_chain_extent.width,
			height = app.swap_chain_extent.height,
			layers = 1,
		}, nil, &fb[i]); result != .SUCCESS {
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

	ca_ref := vk.AttachmentReference{
		attachment = 0,
		layout = .ATTACHMENT_OPTIMAL,
	}

	subpass := vk.SubpassDescription{
		pipelineBindPoint = .GRAPHICS,
		colorAttachmentCount = 1,
		pColorAttachments = &ca_ref,
	}

	if result := vk.CreateRenderPass(app.device, &vk.RenderPassCreateInfo{
		sType = .RENDER_PASS_CREATE_INFO,
		attachmentCount = 1,
		pAttachments = &color_attachment,
		subpassCount = 1,
		pSubpasses = &subpass,
		dependencyCount = 1,
		pDependencies = &vk.SubpassDependency{
			srcSubpass = vk.SUBPASS_EXTERNAL,
			dstSubpass = 0,
			srcStageMask = {.COLOR_ATTACHMENT_OUTPUT},
			dstStageMask = {.COLOR_ATTACHMENT_OUTPUT},
			dstAccessMask = {.COLOR_ATTACHMENT_WRITE},
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
		frontFace = .CLOCKWISE,
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

create_swap_chain :: proc(app: ^Hello_Triangle) -> vk.SwapchainKHR {
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

	swap_chain: vk.SwapchainKHR
	if vk.CreateSwapchainKHR(app.device, &create_info, nil, &swap_chain) != .SUCCESS {
		panic("Could not create swap chain!")
	}

	sc_image_count: u32
	vk.GetSwapchainImagesKHR(app.device, swap_chain, &sc_image_count, nil)
	app.swap_chain_images = make([dynamic]vk.Image, sc_image_count)
	vk.GetSwapchainImagesKHR(
		app.device,
		swap_chain,
		&sc_image_count,
		raw_data(app.swap_chain_images),
	)

	app.swap_chain_image_format = surface_format.format
	app.swap_chain_extent = extent
	return swap_chain
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

choose_swap_present_mode :: proc(
	available_present_modes: []vk.PresentModeKHR,
) -> vk.PresentModeKHR {
	for present_mode in available_present_modes {
		if present_mode == .MAILBOX do return present_mode
	}

	return .FIFO
}

choose_swap_extent :: proc(
	app: Hello_Triangle,
	capabilities: vk.SurfaceCapabilitiesKHR,
) -> vk.Extent2D {
	if capabilities.currentExtent.width != max(u32) {
		return capabilities.currentExtent
	} else {
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
	}
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
		swapchain_adequate \
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
		result.formats = make([dynamic]vk.SurfaceFormatKHR, format_count)
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
		result.present_modes = make([dynamic]vk.PresentModeKHR, present_mode_count)
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
