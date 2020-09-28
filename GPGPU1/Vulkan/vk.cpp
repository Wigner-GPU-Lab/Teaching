#include <vulkan.h>
#include <fstream>
#include <iostream>
#include <vector>

VkResult CreateDebugUtilsMessengerEXT(VkInstance instance, const VkDebugUtilsMessengerCreateInfoEXT* pCreateInfo, const VkAllocationCallbacks* pAllocator, VkDebugUtilsMessengerEXT* pDebugMessenger)
{
    auto func = reinterpret_cast<PFN_vkCreateDebugUtilsMessengerEXT>(vkGetInstanceProcAddr(instance, "vkCreateDebugUtilsMessengerEXT"));
    if (func != nullptr) { return func(instance, pCreateInfo, pAllocator, pDebugMessenger); } else { return VK_ERROR_EXTENSION_NOT_PRESENT; }
}

VkResult DestroyDebugUtilsMessengerEXT(VkInstance instance, VkDebugUtilsMessengerEXT debugMessenger, const VkAllocationCallbacks* pAllocator)
{
    auto func = reinterpret_cast<PFN_vkDestroyDebugUtilsMessengerEXT>(vkGetInstanceProcAddr(instance, "vkDestroyDebugUtilsMessengerEXT"));
    if (func != nullptr) { func(instance, debugMessenger, pAllocator); return VK_SUCCESS; } else { return VK_ERROR_EXTENSION_NOT_PRESENT; }
}

static VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
                                                    VkDebugUtilsMessageTypeFlagsEXT        messageType,
                                              const VkDebugUtilsMessengerCallbackDataEXT*  pCallbackData,
                                                    void*                                  pUserData)
{
    std::cout << "Error from validation layer: " << pCallbackData->pMessage << "\n";
    return VK_FALSE;
}

int main()
{
    const std::vector<const char*> exts             = { VK_EXT_DEBUG_UTILS_EXTENSION_NAME };
    const std::vector<const char*> validationLayers = { "VK_LAYER_KHRONOS_validation" };
    VkResult err = VK_SUCCESS;

    VkInstance instance;
    VkDebugUtilsMessengerEXT debugCallbackFn;
    VkDevice device;
    VkQueue  graphicsQueue;

    VkBuffer buffer1, buffer2, buffer3;
    VkDeviceMemory bufferMemory1, bufferMemory2, bufferMemory3;

    VkPipeline pipeline;
    VkPipelineLayout pipelineLayout;
    VkShaderModule computeShaderModule;

    VkCommandPool commandPool;
    VkCommandBuffer commandBuffer;

    VkDescriptorPool descriptorPool;
    VkDescriptorSet descriptorSet;
    VkDescriptorSetLayout descriptorSetLayout;

    size_t N = 1024;

    float factor = 10.0f;
    std::vector<float> data1(N);
    std::vector<float> data2(N);
    for(size_t i=0; i<N; ++i){ data1[i] = i/100.0f; }
    for(size_t i=0; i<N; ++i){ data2[i] = (N-i)/100.0f; }

    size_t bufferSize = N * sizeof(float);

    VkApplicationInfo appInfo{};
    appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    appInfo.pApplicationName = "Test";
    appInfo.applicationVersion = VK_MAKE_VERSION(1, 1, 0);
    appInfo.pEngineName = "No Engine";
    appInfo.engineVersion = VK_MAKE_VERSION(1, 1, 0);
    appInfo.apiVersion = VK_API_VERSION_1_1;

    VkInstanceCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    createInfo.pApplicationInfo = &appInfo;
    createInfo.enabledExtensionCount   = static_cast<uint32_t>(exts.size());
    createInfo.ppEnabledExtensionNames = exts.data();

    uint32_t layerCount;
    err = vkEnumerateInstanceLayerProperties(&layerCount, nullptr);
    if(err != VK_SUCCESS){ std::cout << "Failed to get instance layer count\n"; return -1; }

    std::vector<VkLayerProperties> availableLayers(layerCount);
    err = vkEnumerateInstanceLayerProperties(&layerCount, availableLayers.data());
    if(err != VK_SUCCESS){ std::cout << "Failed to get instance layers\n"; return -1; }

    auto missing = validationLayers.size();
    for(const char* layerName : validationLayers)
    {
        bool layerFound = false;
        for(const auto& layerProperties : availableLayers)
        {
            //std::cout << "   " << layerProperties.layerName << "\n";
            if(strcmp(layerName, layerProperties.layerName) == 0){ missing -= 1; layerFound = true; }
        }
        //std::cout << layerName << "\n";
        if(!layerFound){ std::cout << layerName << " not found.\n"; }
    }
    if(!missing){ std::cout << "All required validation layers are found.\n"; }
    else{ std::cout << "Some validation layers are missing.\n"; }

    VkDebugUtilsMessengerCreateInfoEXT debugCreateInfo;
    if(validationLayers.size() > 0)
    {
        createInfo.enabledLayerCount = static_cast<uint32_t>(validationLayers.size());
        createInfo.ppEnabledLayerNames = validationLayers.data();
        createInfo.pNext = (VkDebugUtilsMessengerCreateInfoEXT*) &debugCreateInfo;
        debugCreateInfo = {};
        debugCreateInfo.sType           = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
        debugCreateInfo.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
        debugCreateInfo.messageType     = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
        debugCreateInfo.pfnUserCallback = debugCallback;
    }
    else
    {
        createInfo.enabledLayerCount = 0;
        createInfo.pNext = nullptr;
    }

    err = vkCreateInstance(&createInfo, nullptr, &instance);
    if(err != VK_SUCCESS){ std::cout << "Failed to create instance\n"; return -1; }

    //setup debug:
    err = CreateDebugUtilsMessengerEXT(instance, &debugCreateInfo, nullptr, &debugCallbackFn);
    if(err != VK_SUCCESS){ std::cout << "Failed to create debug callback " << err << "\n"; return -1; }

    //devices:

    uint32_t deviceCount = 0;
    err = vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr);
    if(err != VK_SUCCESS || deviceCount == 0){ std::cout << "Failed to get device count " << err << "\n"; return -1; }
        
    std::vector<VkPhysicalDevice> devices(deviceCount);
    err = vkEnumeratePhysicalDevices(instance, &deviceCount, devices.data());
    if(err != VK_SUCCESS ){ std::cout << "Failed to get devices " << err << "\n"; return -1; }

    for(int d=0; d<deviceCount; ++d)
    {
        VkPhysicalDeviceProperties deviceProps{};
        vkGetPhysicalDeviceProperties(devices[d], &deviceProps);
        std::cout << "Device " << d << " : " << deviceProps.deviceName << "\n";
    
        VkPhysicalDeviceMemoryBudgetPropertiesEXT memBudget{};
        memBudget.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_BUDGET_PROPERTIES_EXT;
        VkPhysicalDeviceMemoryProperties2 deviceProps2{};
        deviceProps2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_PROPERTIES_2;
        deviceProps2.pNext = &memBudget;
        vkGetPhysicalDeviceMemoryProperties2(devices[d], &deviceProps2);

        auto hc = deviceProps2.memoryProperties.memoryHeapCount;
        std::cout << "Device " << d << " : heap count: " << hc << "\n";
        for(auto i=0*hc; i<hc; ++i)
        {
            bool isLocal = deviceProps2.memoryProperties.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT;
            std::cout << "Device " << d << " : heap " << i << " size:   " << deviceProps2.memoryProperties.memoryHeaps[i].size << (isLocal ? " (local)\n" : "\n");
            std::cout << "Device " << d << " : heap " << i << " budget: " << memBudget.heapBudget[i] << "\n";
            std::cout << "Device " << d << " : heap " << i << " usage:  " << memBudget.heapUsage[i]  << "\n";
        }

        uint32_t queueFamilyCount = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(devices[d], &queueFamilyCount, nullptr);

        std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
        vkGetPhysicalDeviceQueueFamilyProperties(devices[d], &queueFamilyCount, queueFamilies.data());

        for(auto i=0*queueFamilies.size(); i<queueFamilies.size(); ++i)
        {
            const auto& f = queueFamilies[i];
            std::cout << "Queue " << i << " with count " << f.queueCount << " supports:";
            if(f.queueFlags & VK_QUEUE_GRAPHICS_BIT){ std::cout << " graphics"; }
            if(f.queueFlags & VK_QUEUE_COMPUTE_BIT ){ std::cout << " compute" ; }
            if(f.queueFlags & VK_QUEUE_TRANSFER_BIT){ std::cout << " transfer"; }
            std::cout << "\n";
        }
    }

    //logical device and queue(s)

    auto devSel = devices[0];
    int deviceQueueFamilyIndex = 0;
    VkDeviceQueueCreateInfo queueCreateInfo{};
    queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    queueCreateInfo.queueFamilyIndex = deviceQueueFamilyIndex;
    queueCreateInfo.queueCount = 1;
    float queuePriority = 1.0f;
    queueCreateInfo.pQueuePriorities = &queuePriority;

    VkPhysicalDeviceFeatures deviceFeatures{};

    VkDeviceCreateInfo devCreateInfo{};
    devCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    devCreateInfo.pQueueCreateInfos = &queueCreateInfo;
    devCreateInfo.queueCreateInfoCount = 1;
    devCreateInfo.pEnabledFeatures = &deviceFeatures;
    devCreateInfo.enabledExtensionCount = 0;

    if(validationLayers.size() > 0)
    {
        devCreateInfo.enabledLayerCount = static_cast<uint32_t>(validationLayers.size());
        devCreateInfo.ppEnabledLayerNames = validationLayers.data();
    }
    else{ devCreateInfo.enabledLayerCount = 0; }

    err = vkCreateDevice(devSel, &devCreateInfo, nullptr, &device);
    if(err != VK_SUCCESS){ std::cout << "Device creation failed: " << err << "\n"; return -1; }

    vkGetDeviceQueue(device, deviceQueueFamilyIndex, 0, &graphicsQueue);

    //buffer
    VkBufferCreateInfo bufferCreateInfo{};
    bufferCreateInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferCreateInfo.size = bufferSize; // buffer size in bytes. 
    bufferCreateInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT; // buffer is used as a storage buffer.
    bufferCreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE; // buffer is exclusive to a single queue family at a time. 
    err = vkCreateBuffer(device, &bufferCreateInfo, nullptr, &buffer1);
    if(err != VK_SUCCESS){ std::cout << "Buffer 1 creation failed: " << err << "\n"; return -1; }
    err = vkCreateBuffer(device, &bufferCreateInfo, nullptr, &buffer2);
    if(err != VK_SUCCESS){ std::cout << "Buffer 2 creation failed: " << err << "\n"; return -1; }
    err = vkCreateBuffer(device, &bufferCreateInfo, nullptr, &buffer3);
    if(err != VK_SUCCESS){ std::cout << "Buffer 3 creation failed: " << err << "\n"; return -1; }

    //allocate memory
    VkMemoryRequirements memoryRequirements{};
    vkGetBufferMemoryRequirements(device, buffer1, &memoryRequirements);

    VkPhysicalDeviceMemoryProperties memoryProperties;
    vkGetPhysicalDeviceMemoryProperties(devSel, &memoryProperties);

    VkMemoryAllocateInfo allocateInfo{};
    allocateInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocateInfo.allocationSize = memoryRequirements.size;

    //find memory type:
    uint32_t res = -1;
    auto rq = VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT;
    for(uint32_t i = 0; i < memoryProperties.memoryTypeCount; ++i)
    {
        if((memoryRequirements.memoryTypeBits & (1 << i)) && ((memoryProperties.memoryTypes[i].propertyFlags & rq) == rq)){ res = i; break; }
    }
    allocateInfo.memoryTypeIndex = res;
    //findMemoryType(memoryRequirements.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);

    err = vkAllocateMemory(device, &allocateInfo, nullptr, &bufferMemory1);
    if(err != VK_SUCCESS){ std::cout << "Allocate memory 1 failed: " << err << "\n"; return -1; }

    err = vkAllocateMemory(device, &allocateInfo, nullptr, &bufferMemory2);
    if(err != VK_SUCCESS){ std::cout << "Allocate memory 2 failed: " << err << "\n"; return -1; }

    err = vkAllocateMemory(device, &allocateInfo, nullptr, &bufferMemory3);
    if(err != VK_SUCCESS){ std::cout << "Allocate memory 3 failed: " << err << "\n"; return -1; }

    {
        void* mapped = nullptr;
        res = vkMapMemory(device, bufferMemory1, 0, bufferCreateInfo.size, 0, &mapped);
        if(err != VK_SUCCESS){ std::cout << "Failed to map memory 1: " << err << "\n"; return -1; }
        memcpy(mapped, data1.data(), bufferCreateInfo.size);
        vkUnmapMemory(device, bufferMemory1);
    }

    {
        void* mapped = nullptr;
        res = vkMapMemory(device, bufferMemory2, 0, bufferCreateInfo.size, 0, &mapped);
        if(err != VK_SUCCESS){ std::cout << "Failed to map memory 2: " << err << "\n"; return -1; }
        memcpy(mapped, data2.data(), bufferCreateInfo.size);
        vkUnmapMemory(device, bufferMemory2);
    }

    err = vkBindBufferMemory(device, buffer1, bufferMemory1, 0);
    if(err != VK_SUCCESS){ std::cout << "Bind buffer memory 1 failed: " << err << "\n"; return -1; }
    err = vkBindBufferMemory(device, buffer2, bufferMemory2, 0);
    if(err != VK_SUCCESS){ std::cout << "Bind buffer memory 2 failed: " << err << "\n"; return -1; }
    err = vkBindBufferMemory(device, buffer3, bufferMemory3, 0);
    if(err != VK_SUCCESS){ std::cout << "Bind buffer memory 3 failed: " << err << "\n"; return -1; }

    //descriptor set layout
    VkDescriptorSetLayoutBinding descriptorSetLayoutBinding[3] = {{}, {}, {}};
    descriptorSetLayoutBinding[0].binding = 0; // binding = 0
    descriptorSetLayoutBinding[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    descriptorSetLayoutBinding[0].descriptorCount = 1;
    descriptorSetLayoutBinding[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    descriptorSetLayoutBinding[1].binding = 1; // binding = 1
    descriptorSetLayoutBinding[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    descriptorSetLayoutBinding[1].descriptorCount = 1;
    descriptorSetLayoutBinding[1].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    descriptorSetLayoutBinding[2].binding = 2; // binding = 2
    descriptorSetLayoutBinding[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    descriptorSetLayoutBinding[2].descriptorCount = 1;
    descriptorSetLayoutBinding[2].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    VkDescriptorSetLayoutCreateInfo descriptorSetLayoutCreateInfo{};
    descriptorSetLayoutCreateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    descriptorSetLayoutCreateInfo.bindingCount = 3;
    descriptorSetLayoutCreateInfo.pBindings = descriptorSetLayoutBinding; 

    err = vkCreateDescriptorSetLayout(device, &descriptorSetLayoutCreateInfo, nullptr, &descriptorSetLayout);
    if(err != VK_SUCCESS){ std::cout << "Failed to create descriptor set layout: " << err << "\n"; return -1; }

    VkDescriptorPoolSize descriptorPoolSize{};
    descriptorPoolSize.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    descriptorPoolSize.descriptorCount = 1;

    VkDescriptorPoolCreateInfo descriptorPoolCreateInfo{};
    descriptorPoolCreateInfo.sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    descriptorPoolCreateInfo.maxSets       = 1; // we only need to allocate one descriptor set from the pool.
    descriptorPoolCreateInfo.poolSizeCount = 1;
    descriptorPoolCreateInfo.pPoolSizes    = &descriptorPoolSize;
    res = vkCreateDescriptorPool(device, &descriptorPoolCreateInfo, nullptr, &descriptorPool);
    if(err != VK_SUCCESS){ std::cout << "Failed to create descriptor pool: " << err << "\n"; return -1; }

    VkDescriptorSetAllocateInfo descriptorSetAllocateInfo{};
    descriptorSetAllocateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO; 
    descriptorSetAllocateInfo.descriptorPool = descriptorPool; // pool to allocate from.
    descriptorSetAllocateInfo.descriptorSetCount = 1; // allocate a single descriptor set.
    descriptorSetAllocateInfo.pSetLayouts = &descriptorSetLayout;

    err = vkAllocateDescriptorSets(device, &descriptorSetAllocateInfo, &descriptorSet);
    if(err != VK_SUCCESS){ std::cout << "Failed to allocate descriptor set: " << err << "\n"; return -1; }

    //bind buffer to descriptor set:
    VkDescriptorBufferInfo descriptorBufferInfo[3] = {{}, {}, {}};
    descriptorBufferInfo[0].buffer = buffer1;
    descriptorBufferInfo[0].offset = 0;
    descriptorBufferInfo[0].range = bufferSize;
    descriptorBufferInfo[1].buffer = buffer2;
    descriptorBufferInfo[1].offset = 0;
    descriptorBufferInfo[1].range = bufferSize;
    descriptorBufferInfo[2].buffer = buffer3;
    descriptorBufferInfo[2].offset = 0;
    descriptorBufferInfo[2].range = bufferSize;

    VkWriteDescriptorSet writeDescriptorSet{};
    writeDescriptorSet.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writeDescriptorSet.dstSet = descriptorSet; // write to this descriptor set.
    writeDescriptorSet.dstBinding = 0; // write to the first, and only binding.
    writeDescriptorSet.descriptorCount = 3; // update all descriptors.
    writeDescriptorSet.pBufferInfo = descriptorBufferInfo;
    writeDescriptorSet.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; // storage buffer.
    vkUpdateDescriptorSets(device, 1, &writeDescriptorSet, 0, nullptr);

    //load shader
    std::ifstream input_file( "comp.spv", std::ios::binary );
    std::vector<char> shader_data{std::istreambuf_iterator<char>(input_file), std::istreambuf_iterator<char>()};

    VkShaderModuleCreateInfo computeShaderModuleInfo{};
    computeShaderModuleInfo.flags = 0;
    computeShaderModuleInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    computeShaderModuleInfo.pNext = nullptr;
    computeShaderModuleInfo.pCode = (uint32_t*)shader_data.data();
    computeShaderModuleInfo.codeSize = shader_data.size();

	res = vkCreateShaderModule(device, &computeShaderModuleInfo, nullptr, &computeShaderModule);
    if(err != VK_SUCCESS){ std::cout << "Failed create shader module: " << err << "\n"; return -1; }

    //pipeline
    VkPipelineShaderStageCreateInfo shaderStageCreateInfo{};
    shaderStageCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    shaderStageCreateInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    shaderStageCreateInfo.module = computeShaderModule;
    shaderStageCreateInfo.pName = "main";

    VkPushConstantRange vpcr[1];
    vpcr[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT ;
    vpcr[0].offset = 0;
    vpcr[0].size = sizeof( float );

    VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo{};
    pipelineLayoutCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipelineLayoutCreateInfo.setLayoutCount = 1;
    pipelineLayoutCreateInfo.pSetLayouts = &descriptorSetLayout;
    pipelineLayoutCreateInfo.pushConstantRangeCount = 1;
    pipelineLayoutCreateInfo.pPushConstantRanges = vpcr;
    err = vkCreatePipelineLayout(device, &pipelineLayoutCreateInfo, nullptr, &pipelineLayout);
    if(err != VK_SUCCESS){ std::cout << "Failed create pipeline layout: " << err << "\n"; return -1; }

    VkComputePipelineCreateInfo pipelineCreateInfo{};
    pipelineCreateInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipelineCreateInfo.stage = shaderStageCreateInfo;
    pipelineCreateInfo.layout = pipelineLayout;

    err = vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipelineCreateInfo, nullptr, &pipeline);
    if(err != VK_SUCCESS){ std::cout << "Failed create pipeline: " << err << "\n"; return -1; }

    //command buffer
    VkCommandPoolCreateInfo commandPoolCreateInfo{};
    commandPoolCreateInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    commandPoolCreateInfo.flags = 0;
    commandPoolCreateInfo.queueFamilyIndex = deviceQueueFamilyIndex;
    err = vkCreateCommandPool(device, &commandPoolCreateInfo, nullptr, &commandPool);
    if(err != VK_SUCCESS){ std::cout << "Failed create command pool: " << err << "\n"; return -1; }

    VkCommandBufferAllocateInfo commandBufferAllocateInfo{};
    commandBufferAllocateInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    commandBufferAllocateInfo.commandPool = commandPool;
    commandBufferAllocateInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    commandBufferAllocateInfo.commandBufferCount = 1;
    err = vkAllocateCommandBuffers(device, &commandBufferAllocateInfo, &commandBuffer);
    if(err != VK_SUCCESS){ std::cout << "Failed allocate command buffers: " << err << "\n"; return -1; }

    //command recording
    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT; // the buffer is only submitted and used once.
    err = vkBeginCommandBuffer(commandBuffer, &beginInfo); // start recording commands.
    if(err != VK_SUCCESS){ std::cout << "Failed begin command buffer: " << err << "\n"; return -1; }

    vkCmdBindPipeline      (commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
    vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout, 0, 1, &descriptorSet, 0, nullptr);
    vkCmdPushConstants     (commandBuffer, pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT , 0, sizeof(float), (void*)&factor );
    vkCmdDispatch          (commandBuffer, N, 1, 1);
    err = vkEndCommandBuffer(commandBuffer);
    if(err != VK_SUCCESS){ std::cout << "Failed to end command buffer recording: " << err << "\n"; return -1; }

    //submit
    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1; // submit a single command buffer
    submitInfo.pCommandBuffers = &commandBuffer; // the command buffer to submit.

    VkFence fence;
    VkFenceCreateInfo fenceCreateInfo{};
    fenceCreateInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    fenceCreateInfo.flags = 0;
    err = vkCreateFence(device, &fenceCreateInfo, nullptr, &fence);
    if(err != VK_SUCCESS){ std::cout << "Failed to create fence: " << err << "\n"; return -1; }

    err = vkQueueSubmit(graphicsQueue, 1, &submitInfo, fence);
    if(err != VK_SUCCESS){ std::cout << "Failed to submit: " << err << "\n"; return -1; }

    err = vkWaitForFences(device, 1, &fence, VK_TRUE, 100000000000);
    if(err != VK_SUCCESS){ std::cout << "Failed to wait for fence: " << err << "\n"; return -1; }

    vkDestroyFence(device, fence, nullptr);

    //read data
    void* mappedMemory = nullptr;
    err = vkMapMemory(device, bufferMemory3, 0, bufferSize, 0, &mappedMemory);
    if(err != VK_SUCCESS){ std::cout << "Failed to map memory: " << err << "\n"; return -1; }

    {
        float* ptr = (float*)mappedMemory;
        for(int i=0; i<N; ++i)
        {
            std::cout << ptr[i] << " expected: " << factor * data1[i] + data2[i] << "\n";
        }
    }

    vkUnmapMemory(device, bufferMemory3);

    std::cout << "Success\n";

    vkFreeCommandBuffers(device, commandPool, 1, &commandBuffer);
    vkDestroyCommandPool(device, commandPool, nullptr);
    vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
    vkDestroyPipeline(device, pipeline, nullptr);
    vkDestroyShaderModule(device, computeShaderModule, nullptr);
    vkDestroyDescriptorPool(device, descriptorPool, nullptr);
    vkDestroyDescriptorSetLayout(device, descriptorSetLayout, nullptr);
    vkFreeMemory(device, bufferMemory1, nullptr);
    vkFreeMemory(device, bufferMemory2, nullptr);
    vkFreeMemory(device, bufferMemory3, nullptr);
    vkDestroyBuffer(device, buffer1, nullptr);
    vkDestroyBuffer(device, buffer2, nullptr);
    vkDestroyBuffer(device, buffer3, nullptr);
    vkDestroyDevice(device, nullptr);

    err = DestroyDebugUtilsMessengerEXT(instance, debugCallbackFn, nullptr);
    if(err != VK_SUCCESS){ std::cout << "Failed to destroy debug callback\n"; return -1; }

    vkDestroyInstance(instance, nullptr);
    
    return 0;
}
