package main

import "core:c"
import "core:fmt"
import "core:strings"
import "core:os"
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


	mat: matrix[4, 4]f32
	vec: [4]f32
	test := mat * vec

	fmt.println(test)

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
	graphics_queue:          vk.Queue,
	present_queue:           vk.Queue,
	dbg_msgr:                vk.DebugUtilsMessengerEXT,
	surface:                 vk.SurfaceKHR,
	swap_chain:              vk.SwapchainKHR,
	swap_chain_images:       [dynamic]vk.Image,
	swap_chain_image_format: vk.Format,
	swap_chain_extent:       vk.Extent2D,
	swap_chain_image_views:  [dynamic]vk.ImageView,
}

run :: proc(app: ^Hello_Triangle) {
	for !glfw.WindowShouldClose(app.window) {
		glfw.PollEvents()
	}
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
	}

	if glfw.CreateWindowSurface(app.instance, app.window, nil, &app.surface) != .SUCCESS {
		panic("Couldn't create surface")
	}

	app.physical_device = pick_physical_device(app)
	fmt.println("physical device:", app.physical_device)

	app.device, app.graphics_queue, app.present_queue = create_logical_device(app)
	app.swap_chain = create_swap_chain(&app)

	fmt.println()
	fmt.println("graphics queue:", app.graphics_queue, "present_queue:", app.present_queue)
	fmt.println()

	app.swap_chain_image_views = make([dynamic]vk.ImageView, len(app.swap_chain_images))
	for image, i in app.swap_chain_images {
		result := vk.CreateImageView(
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
		)

		if result != .SUCCESS {
			panic("Could not create image view!")
		}
	}

	return
}

cleanup :: proc(app: Hello_Triangle) {
	defer glfw.Terminate()
	defer glfw.DestroyWindow(app.window)
	defer vk.DestroyInstance(app.instance, nil)
	defer vk.DestroyDevice(app.device, nil)
	defer destroy_debug_messenger(app.instance, app.dbg_msgr, nil)
	defer vk.DestroySurfaceKHR(app.instance, app.surface, nil)
	defer vk.DestroySwapchainKHR(app.device, app.swap_chain, nil)
	defer {
		for image_view in app.swap_chain_image_views {
			vk.DestroyImageView(app.device, image_view, nil)
		}
	}
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

	create_info := vk.SwapchainCreateInfoKHR{
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
		append(&queue_create_infos, vk.DeviceQueueCreateInfo{
            sType = .DEVICE_QUEUE_CREATE_INFO,
            queueFamilyIndex = queue_family,
            queueCount = 1,
            pQueuePriorities = &queue_priority,
        })
	}

	device_features: vk.PhysicalDeviceFeatures

	create_info := vk.DeviceCreateInfo{
	    sType = .DEVICE_CREATE_INFO,
	    pQueueCreateInfos = raw_data(queue_create_infos),
	    queueCreateInfoCount = u32(len(queue_create_infos)),
	    pEnabledFeatures = &device_features,
	    enabledExtensionCount = u32(len(device_extensions)),
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
	// Technically ApplicationInfo is OPTIONAL
	app_info := vk.ApplicationInfo{
	    sType = .APPLICATION_INFO,
	    pApplicationName = "Hello Triangle",
	    applicationVersion = vk.MAKE_VERSION(1, 0, 0),
	    pEngineName = "No Engine",
	    engineVersion = vk.MAKE_VERSION(1, 0, 0),
	    apiVersion = vk.API_VERSION_1_0,
    }

    // specifies required extensions. Vulkan requires
	// an extension to interface with windowing system since the
	// core is platform agnostic
	// will add the glfw required basics and then any additional
	// we specify
	extensions := get_required_extensions()

	// InstanceCreateInfo is REQUIRED
	create_info := vk.InstanceCreateInfo{
	    sType = .INSTANCE_CREATE_INFO,
	    pApplicationInfo = &app_info,
	    enabledExtensionCount = u32(len(extensions)),
	    ppEnabledExtensionNames = raw_data(extensions),
    }

	if enable_validation_layers && !check_validation_layer_support() {
		return nil, false
	}

    debug_info: vk.DebugUtilsMessengerCreateInfoEXT
	// for now we'll just not enable any validation layers
	if enable_validation_layers {
		create_info.enabledLayerCount = u32(len(validation_layers))
		create_info.ppEnabledLayerNames = raw_data(validation_layers)

		debug_info = create_debug_messenger_info()
		create_info.pNext = &debug_info
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
	return vk.DebugUtilsMessengerCreateInfoEXT{
	    sType = .DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT,
	    messageSeverity = {.VERBOSE, .INFO, .WARNING, .ERROR},
	    messageType = {.GENERAL, .VALIDATION, .PERFORMANCE},
	    pfnUserCallback = debug_callback,
	    pUserData = context.user_ptr,
    }
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
