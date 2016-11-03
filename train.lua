require 's2sa.dict'

local lfs = require 'lfs'
local path = require 'pl.path'
local cuda = require 's2sa.cuda'
local Bookkeeper = require 's2sa.bookkeeper'
local Checkpoint = require 's2sa.checkpoint'
local Data = require 's2sa.data'
local Decoder = require 's2sa.decoder'
local Encoder = require 's2sa.encoder'
local Evaluator = require 's2sa.evaluator'
local Generator = require 's2sa.generator'
local Optim = require 's2sa.optim'
local table_utils = require 's2sa.table_utils'

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

-- GPU
cmd:option('-gpuid', -1, [[Which gpu to use (1-indexed). < 1 = use CPU]])
cmd:option('-ngpu', 1, [[How many parallel GPU]])
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

  local layers = {encoders[1], decoders[1], generators[1]}

  cutorch.setDevice(1)
  print('Initializing parameters...')
  for i = 1, #layers do
    local p, gp = layers[i].network:getParameters()
    if opt.train_from:len() == 0 then
      p:uniform(-opt.param_init, opt.param_init)
    end
    num_params = num_params + p:size(1)
    params[i] = p
    grad_params[i] = gp
  end
  print("Number of parameters: " .. num_params)

  local num_params2 = 0
  local params2 = {}
  local grad_params2 = {}

  if opt.ngpu == 2 then
    cutorch.setDevice(1)
    local layers2 = {encoders[2] , decoders[2], generators[2]}

    for i = 1, #layers2 do
      local p, gp = layers2[i].network:getParameters()
      num_params2 = num_params2 + p:size(1)
      params2[i] = p
      grad_params2[i] = gp
    end
    print("Number of parameters: " .. num_params2)
    cutorch.setDevice(1)
  end

  local function train_batch(data, epoch, optim, checkpoint)
    local bookkeeper = Bookkeeper.new({
      learning_rate = optim:get_rate(),
      data_size = #data,
      epoch = epoch
    })

    for j = 1, opt.ngpu do
      encoders[j]:training()
      decoders[j]:training()
      generators[j]:training()
    end

    local batch_order = torch.randperm(#data) -- shuffle mini batch order

    for i = 1, #data, opt.ngpu do
      batchs = {}

      cutorch.setDevice(1)
      encoders[1]:forget()
      decoders[1]:forget()
      batchs[1] = data:get_batch(batch_order[i], 1)
      table_utils.zero(grad_params)

      -- forward encoder
      local encoder_states, context = encoders[1]:forward(batchs[1])

      -- forward decoder
      local decoder_states, decoder_out = decoders[1]:forward(batchs[1], encoder_states)

      -- forward and backward attention and generator
      local decoder_grad_output, grad_context, loss = generators[1]:process(batchs[1], context, decoder_states, decoder_out)

      -- backward decoder
      local decoder_grad_input = decoders[1]:backward(decoder_grad_output)

      -- backward encoder
      local encoder_grad_output = decoder_grad_input
      encoder_grad_output[#encoder_grad_output] = grad_context
      encoders[1]:backward(encoder_grad_output)

      if opt.ngpu == 2 then
        cutorch.setDevice(2)

        encoders[2]:forget()
        decoders[2]:forget()
        batchs[2] = data:get_batch(batch_order[i+1], 2)
        table_utils.zero(grad_params2)

        -- forward encoder
        local encoder_states2, context2 = encoders[2]:forward(batchs[2])

        -- forward decoder
        local decoder_states2, decoder_out2 = decoders[2]:forward(batchs[2], encoder_states2)

        -- forward and backward attention and generator
        local decoder_grad_output2, grad_context2, loss2 = generators[2]:process(batchs[2], context2, decoder_states2, decoder_out2)

        -- backward decoder
        local decoder_grad_input2 = decoders[2]:backward(decoder_grad_output2)

        -- backward encoder
        local encoder_grad_output2 = decoder_grad_input2
        encoder_grad_output2[#encoder_grad_output2] = grad_context2
        encoders[2]:backward(encoder_grad_output2)
      end

      if opt.ngpu == 3 then
        cutorch.setDevice(1)
        for j = 1, #grad_params do
          local remote_grad_params=grad_params2[j]:clone()
          grad_params[j]=(grad_params[j]*batchs[1].size+remote_grad_params[j]*batchs[2].size)/(batchs[1].size+batchs[2].size)
        end
      end

      optim:update_params(params, grad_params, opt.max_grad_norm)
      if opt.ngpu == 2 then
        cutorch.setDevice(2)
        for j = 1, #params do
          params2[j]:copy(params[j])
        end
      end

      -- Bookkeeping
      bookkeeper:update(batchs[1], loss)

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
  local encoders = {}
  local decoders = {}
  local generators = {}

  if opt.train_from:len() == 0 then
    for i = 1, opt.ngpu do
      cutorch.setDevice(i)
      table.insert(encoders, Encoder.new({
                                pre_word_vecs = opt.pre_word_vecs_enc,
                                fix_word_vecs = opt.fix_word_vecs_enc,
                                vocab_size = #dataset.src_dict
                             }, opt, i))

      table.insert(decoders, Decoder.new({
                                pre_word_vecs = opt.pre_word_vecs_dec,
                                fix_word_vecs = opt.fix_word_vec,
                                vocab_size = #dataset.targ_dict
                             }, opt, i))

      table.insert(generators, Generator.new({
                                  vocab_size = #dataset.targ_dict
                                }, opt))
    end
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

  for i = 1, opt.ngpu do
    generators[i]:build_criterion(#dataset.targ_dict)
    cuda.convert({encoders[i].network, decoders[i].network, generators[i].network, generators[i].criterion}, i)
  end

  train(train_data, valid_data, encoders, decoders, generators)
end

main()
