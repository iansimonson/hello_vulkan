Steps to draw a triangle:
1. Initiate a Vulkan API through a `VkInstance`
2. Select one or more `VkPhysicalDevice` (properties like VRAM size etc)
    e.g. you can ask if this device is a dedicated graphics card or integrated
3. Create a VkDevice (logical device)
4. Specify the types of queues you will be using (commands submitted via VkQueue)
    queues have families e.g. graphics, compute, memory transfer
5. Create a window surface (VkSurfaceKHR)
6. Create a swap chain (VkSwapchainKHR)

to draw image we have to wrap in VkImageView and VkFramebuffer
    frame buffer is group of image views

VkPipeline - describes configurable state on graphics card
VkShaderModule - compiled shader code

****All configurations will have to be set in advanced.***
Otherwise requires tearing down and recreating pipelines (slow?)

Create commands in a VkCommandBuffer which are then sent to the VkQueues
Command buffers allocated from a VkCommandPool (easy enough)
    * there are specific command pools for each queue family

Drawing a triangle requires the following commands:
    * Begin Render Pass
    * Bind Graphics Pipeline
    * Draw 3 vertices
    * End render pass

We can record a command buffer for each possible image ahead of time. You could record command buffer on each frame but that's not as efficient (TRIANGLE ONLY)

Main Loop:
=========
1. vkAcquireNextImageKHR (get next image in swapchain)
2. vkQueueSubmit (send command buffer to queue)
3. vkQueuePresentKHR (return image to swapchain / commit)
    - necessary because everything is async so won't draw to screen
    until the image is returned and semaphore is released


GLM? there is the linalg library in odin

