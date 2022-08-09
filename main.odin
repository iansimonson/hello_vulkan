package main

import "core:c"
import "core:fmt"
import "vendor:glfw"
import vk "vendor:vulkan"

main :: proc() {
    glfw.Init()
    defer glfw.Terminate()

    vk.load_proc_addresses_global(rawptr(glfw.GetInstanceProcAddress))

    fmt.println("vulkan supported?", glfw.VulkanSupported())

    glfw.WindowHint(glfw.CLIENT_API, glfw.NO_API)
    window := glfw.CreateWindow(800, 600, "Vulkan Window", nil, nil)
    defer glfw.DestroyWindow(window)

    extension_count: u32
    result := vk.EnumerateInstanceExtensionProperties(nil, &extension_count, nil)
    fmt.println(result)
    fmt.println(extension_count, "extensions supported")

    mat: matrix[4, 4]f32
    vec: [4]f32
    test := mat * vec

    fmt.println(test)

    for !glfw.WindowShouldClose(window) {
        glfw.PollEvents()
    }
}