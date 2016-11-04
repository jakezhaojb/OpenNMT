require 'torch'
require 'nn'
require 'nngraph'

local Cuda = {
  nn = nn,
  activated = false,
  gpuid = 0
}

function Cuda.init(opt)
  Cuda.activated = opt.gpuid > 0
  Cuda.gpuid = opt.gpuid

  if Cuda.activated then
    local _, err = pcall(function()
      require 'cutorch'
      require 'cunn'
      if opt.cudnn then
        require 'cudnn'
        Cuda.nn = cudnn
      end
      -- allow memory access between devices
      cutorch.getKernelPeerToPeerAccess(true)
      cutorch.manualSeedAll(opt.seed)
    end)

    if err then
      if opt.fallback_to_cpu then
        print('Info: Failed to initialize Cuda on device ' .. opt.gpuid .. ', falling back to CPU.')
        Cuda.activated = false
      else
        error(err)
      end
    else
       print('Using GPU ' .. opt.gpuid .. '.')
    end
  end
end

function Cuda.convert(obj)
  if Cuda.activated then
    if torch.typename(obj) == nil and type(obj) == 'table' then
      for i = 1, #obj do
        obj[i] = Cuda.convert(obj[i])
      end
    elseif obj.cuda ~= nil then
        return obj:cuda()
    end
  end
  return obj
end

function Cuda.getGPUs(ngpu)
  local gpus = {}
  if Cuda.activated then
    if ngpu > cutorch.getDeviceCount() then
      error("not enough available GPU - " .. ngpu .. " requested, " .. cutorch.getDeviceCount() .. " available")
    end
    gpus[1] = Cuda.gpuid
    local i = 1
    while #gpus ~= ngpu do
      if i ~= gpus[1] then
        table.insert(gpus, i)
      end
      i = i + 1
    end
  else
    for _ = 1, ngpu do
      table.insert(gpus, 0)
    end
  end
  return gpus
end

function Cuda.setDevice(gpuIdx)
  if Cuda.activated and gpuIdx > 0 then
    cutorch.setDevice(gpuIdx)
  end
end

return Cuda
