local model_utils = require 's2sa.model_utils'
require 's2sa.sequencer'

local Decoder, Sequencer = torch.class('Decoder', 'Sequencer')

function Decoder:__init(args, opt, gpuIdx)
  Sequencer.__init(self, args, opt, gpuIdx)
end

function Decoder:forward(batch, encoder_states)
  local hidden_states = model_utils.reset_state(self.init_states, batch.size)
  for i = 1, #encoder_states do
    hidden_states[i]:copy(encoder_states[i])
  end

  return Sequencer.forward(self, hidden_states, batch.target_input)
end

return Decoder
