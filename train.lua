require 's2sa.dict'

local lfs = require 'lfs'
local path = require 'pl.path'
local cuda = require 's2sa.cuda'
local Bookkeeper = require 's2sa.bookkeeper'
local Checkpoint = require 's2sa.checkpoint'
local Data = require 's2sa.data'
local Evaluator = require 's2sa.evaluator'
local Optim = require 's2sa.optim'
local table_utils = require 's2sa.table_utils'
local parallel = require 's2sa.parallel'

local cmd = torch.CmdLine()

cmd:text("")
cmd:text("**Data options**")
cmd:text("")
cmd:option('-data','data/demo.t7', [[Path to the training *-train.t7 file from preprocess.lua]])
cmd:option('-savefile', 'seq2seq_lstm_attn', [[Savefile name (model will be saved as
                                             savefile_epochX_PPL.t7 where X is the X-th epoch and PPL is
                                             the validation perplexity]])
cmd:option('-train_from', '', [[If training from a checkpoint then this is the path to the pretrained model.]])

cmd:text("")
cmd:text("**Model options**")
cmd:text("")

cmd:option('-num_layers', 2, [[Number of layers in the LSTM encoder/decoder]])
cmd:option('-rnn_size', 500, [[Size of LSTM hidden states]])
cmd:option('-word_vec_size', 500, [[Word embedding sizes]])

cmd:text("")
cmd:text("**Optimization options**")
cmd:text("")

cmd:option('-max_batch_size', 64, [[Maximum batch size]])
cmd:option('-epochs', 13, [[Number of training epochs]])
cmd:option('-start_epoch', 1, [[If loading from a checkpoint, the epoch from which to start]])
cmd:option('-param_init', 0.1, [[Parameters are initialized over uniform distribution with support (-param_init, param_init)]])
cmd:option('-learning_rate', 1, [[Starting learning rate. If adagrad/adadelta/adam is used,
                                then this is the global learning rate. Recommended settings: sgd =1,
                                adagrad = 0.1, adadelta = 1, adam = 0.1]])
cmd:option('-max_grad_norm', 5, [[If the norm of the gradient vector exceeds this renormalize it to have the norm equal to max_grad_norm]])
cmd:option('-dropout', 0.3, [[Dropout probability. Dropout is applied between vertical LSTM stacks.]])
cmd:option('-lr_decay', 0.5, [[Decay learning rate by this much if (i) perplexity does not decrease
                             on the validation set or (ii) epoch has gone past the start_decay_at_limit]])
cmd:option('-start_decay_at', 9, [[Start decay after this epoch]])
cmd:option('-pre_word_vecs_enc', '', [[If a valid path is specified, then this will load
                                     pretrained word embeddings (hdf5 file) on the encoder side.
                                     See README for specific formatting instructions.]])
cmd:option('-pre_word_vecs_dec', '', [[If a valid path is specified, then this will load
                                     pretrained word embeddings (hdf5 file) on the decoder side.
                                     See README for specific formatting instructions.]])
cmd:option('-fix_word_vecs_enc', 0, [[If = 1, fix word embeddings on the encoder side]])
cmd:option('-fix_word_vecs_dec', 0, [[If = 1, fix word embeddings on the decoder side]])

cmd:text("")
cmd:text("**Other options**")
cmd:text("")

-- GPU and parallelism
cmd:option('-gpuid', -1, [[Which gpu to use (1-indexed). < 1 = use CPU]])
cmd:option('-nparallel', 1, [[How many parallel process]])
cmd:option('-fallback_to_cpu', false, [[Fallback to CPU if no GPU available or can not use cuda/cudnn]])
cmd:option('-cudnn', false, [[Whether to use cudnn or not]])

-- bookkeeping
cmd:option('-save_every', 1, [[Save every this many epochs]])
cmd:option('-intermediate_save', 0, [[Save intermediate models every this many iterations within an epoch]])
cmd:option('-print_every', 50, [[Print stats after this many batches]])
cmd:option('-seed', 3435, [[Seed for random initialization]])

local opt = cmd:parse(arg)


local function train(train_data, valid_data, encoders, decoders, generators)
  local num_params = 0
  local params = {}
  local grad_params = {}

  parallel.launch('Initializing parameters', function()
    _G.layers = { _G.encoder, _G.decoder, _G.generator}
    _G.params = {}
    _G.grad_params = {}
    for i = 1, #_G.layers do
      local p, gp = _G.layers[i].network:getParameters()
      if opt.train_from:len() == 0 then
        p:uniform(-opt.param_init, opt.param_init)
      end
      _G.params[i] = p
      _G.grad_params[i] = gp
    end
    return __threadid, _G.params, _G.grad_params
  end, function(id, theparams, thegrad_params)
    params[id]=theparams
    grad_params[id]=thegrad_params
  end)

  for i = 1, #params[1] do
    num_params = num_params + params[1][i]:size(1)
  end
  print("Number of parameters: " .. num_params)

  local function train_batch(data, epoch, optim, checkpoint)
    local bookkeeper = Bookkeeper.new({
      learning_rate = optim:get_rate(),
      data_size = #data,
      epoch = epoch
    })

    parallel.launch(nil, function()
      _G.encoder:training()
      _G.decoder:training()
      _G.generator:training()
      return __threadid
    end)

    local batch_order = torch.randperm(#data) -- shuffle mini batch order

    for i = 1, #data, parallel.count do
      local total_size=0

      local batchs = {}
      local losses = {}
      for j = 1, parallel.count do
        local batch
        if i+j-1 <= #data then
          batch = data:get_batch(batch_order[i+j-1], true)
        end
        table.insert(batchs, batch)
        total_size = total_size + batch.size
      end

      parallel.launch(nil, function()
        _G.batch = batchs[__threadid]
        if _G.batch == nil then
          return __threadid, 0
        end
        cuda.convert(_G.batch)
        _G.batch.total_size = total_size
        _G.encoder:forget()
        _G.decoder:forget()
        table_utils.zero(_G.grad_params)

        -- forward encoder
        local encoder_states, context = _G.encoder:forward(_G.batch)

        -- forward decoder
        local decoder_states, decoder_out = _G.decoder:forward(_G.batch, encoder_states)

        -- forward and backward attention and generator
        local decoder_grad_output, grad_context, loss = _G.generator:process(_G.batch, context, decoder_states, decoder_out)

        -- backward decoder
        local decoder_grad_input = _G.decoder:backward(decoder_grad_output)

        -- backward encoder
        local encoder_grad_output = decoder_grad_input
        encoder_grad_output[#encoder_grad_output] = grad_context
        _G.encoder:backward(encoder_grad_output)
        return __threadid, loss
      end,
      function(id, loss) losses[id]=loss end)

      for j = 2, parallel.count do
        for h = 1, #grad_params[1] do
          local remote_grad_params=grad_params[j][h]:clone()
          grad_params[1][h]:add(remote_grad_params)
        end
      end

      optim:update_params(params[1], grad_params[1], opt.max_grad_norm)
      for j = 2, parallel.count do
        for h = 1, #params[1] do
          params[j][h]:copy(params[1][h])
        end
      end

      -- Bookkeeping
      bookkeeper:update(batchs, losses)

      if i % opt.print_every == 0 then
        bookkeeper:log(i)
      end

      checkpoint:save_iteration(i, bookkeeper)
    end

    return bookkeeper
  end

  local evaluator = Evaluator.new()
  local optim = Optim.new(opt.learning_rate, opt.lr_decay, opt.start_decay_at)
  local checkpoint = Checkpoint.new({
    layers = layers,
    options = opt,
    optim = optim,
    script_path = lfs.currentdir()
  })

  for epoch = opt.start_epoch, opt.epochs do

    local bookkeeper = train_batch(train_data, epoch, optim, checkpoint)

    local score = evaluator:process({
      encoder = encoders[1],
      decoder = decoders[1],
      generator = generators[1]
    }, valid_data)

    optim:update_rate(score, epoch)

    checkpoint:save_epoch(score, bookkeeper)
  end

  checkpoint:save_final()
end

local function main()
  torch.manualSeed(opt.seed)

  cuda.init(opt)
  parallel.init(opt)

  -- Create the data loader class.
  print('Loading data from ' .. opt.data .. '...')
  local dataset = torch.load(opt.data)

  local train_data = Data.new(dataset.train, opt.max_batch_size)
  local valid_data = Data.new(dataset.valid, opt.max_batch_size)

  print(string.format('Source vocab size: %d, Target vocab size: %d',
                      #dataset.src_dict, #dataset.targ_dict))
  print(string.format('Source max sent len: %d, Target max sent len: %d',
                      train_data.max_source_length, train_data.max_target_length))

  -- Build model
  if opt.train_from:len() == 0 then
    parallel.launch("creating model", function(j)
      _G.encoder = _G.Encoder.new({
                  pre_word_vecs = opt.pre_word_vecs_enc,
                  fix_word_vecs = opt.fix_word_vecs_enc,
                  vocab_size = #dataset.src_dict
                }, opt, i)

      _G.decoder = _G.Decoder.new({
                  pre_word_vecs = opt.pre_word_vecs_dec,
                  fix_word_vecs = opt.fix_word_vec,
                  vocab_size = #dataset.targ_dict
                }, opt, i)

      _G.generator = _G.Generator.new({
                    vocab_size = #dataset.targ_dict
                  }, opt)
    end)
  else
    assert(path.exists(opt.train_from), 'checkpoint path invalid')
    print('loading ' .. opt.train_from .. '...')
    local checkpoint = torch.load(opt.train_from)
    local model, model_opt = checkpoint[1], checkpoint[2]
    opt.num_layers = model_opt.num_layers
    opt.rnn_size = model_opt.rnn_size
    encoder = model[1]
    decoder = model[2]
    generator = model[3]
  end

  parallel.launch("model finalization", function(j)
    _G.generator:build_criterion(#dataset.targ_dict)
    cuda.convert({_G.encoder.network, _G.decoder.network, _G.generator.network, _G.generator.criterion})
  end)

  train(train_data, valid_data)
end

main()
