--[[
   This file provides generic parallel class - allowing to run functions
   in different threads and on different GPU
]]--

local cuda = require 's2sa.cuda'

local Parallel = {
   gpus = {0},
   _pool = nil,
   count = 1
}

function Parallel.init(args)
   if cuda.activated then
      Parallel.count = args.nparallel
      Parallel.gpus = cuda.getGPUs(args.nparallel)
      if Parallel.count > 1 then
         local threads = require 'threads'
         print('launching threads on gpus')
         Parallel._pool = threads.Threads(
            Parallel.count,
            function(threadid)
               cutorch = require 'cutorch'
               print('starting thread ', threadid, 'on GPU ' .. Parallel.gpus[threadid])
               -- require 's2sa.cuda'
               -- cuda.setDevice(Parallel.gpus[threadid])
            end
         )
         print('done...')
      end
   end
end

function Parallel.getGPU(i)
   if cuda.activated and Parallel.gpus[i] ~= 0 then
      return Parallel.gpus[i]
   end
   return 0
end

function Parallel.launch(j, closure)
   if Parallel._pool == nil then
      closure()
   else
      Parallel._pool:addjob(j, closure)
   end
end

return Parallel
