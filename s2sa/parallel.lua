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
      if Parallel.count >= 1 then
         local threads = require 'threads'
         threads.Threads.serialization('threads.sharedserialize')
         local thegpus = Parallel.gpus
         Parallel._pool = threads.Threads(
            Parallel.count,
            function(threadid)
               require 'cunn'
               require 'nngraph'
               _G.Decoder = require 's2sa.decoder'
               _G.Encoder = require 's2sa.encoder'
               _G.Generator = require 's2sa.generator'
               require 's2sa.dict'
               require 's2sa.data'
               print('starting thread ', threadid, 'on GPU ' .. thegpus[threadid])
               cudathread = require 's2sa.cuda'
               cudathread.init(args, thegpus[threadid])
            end
         ) -- dedicate threads to GPUs
         Parallel._pool:specific(true)
      end
   end
end

function Parallel.getGPU(i)
   if cuda.activated and Parallel.gpus[i] ~= 0 then
      return Parallel.gpus[i]
   end
   return 0
end

function Parallel.launch(label, closure, endcallback)
   if label ~= nil then
      print("START",label)
   end
   for j = 1, Parallel.count do
      if Parallel._pool == nil then
         closure(j)
         if endcallback ~= nil then
            endcallback()
         end
      else
         if endcallback == nil then
            Parallel._pool:addjob(j, function() closure(j) end)
         else
            Parallel._pool:addjob(j, function() return closure(j) end, endcallback)
         end
      end
   end
   Parallel._pool:synchronize()
   if label ~= nil then
      print("DONE",label)
   end
end

return Parallel
