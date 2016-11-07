local MM=nn.MM

function MM:updateOutput(input)
  assert(#input == 2, 'input must be a pair of minibatch matrices')
  local a, b = table.unpack(input)
  assert(a:nDimension() == 2 or a:nDimension() == 3, 'input tensors must be 2D or 3D')

  if a:nDimension() == 2 then
    assert(b:nDimension() == 2, 'second input tensor must be 2D')

    if self.transA then a = a:t() end
    if self.transB then b = b:t() end
    assert(a:size(2) == b:size(1), 'matrix sizes do not match')

    self.output:resize(a:size(1), b:size(2))
    self.output:mm(a, b)
  else
    assert(b:nDimension() == 3, 'second input tensor must be 3D')
    assert(a:size(1) == b:size(1), 'inputs must contain the same number of minibatches')

    if self.transA then a = a:transpose(2, 3) end
    if self.transB then b = b:transpose(2, 3) end
    assert(a:size(3) == b:size(2), 'matrix sizes do not match')

    self.output:resize(a:size(1), a:size(2), b:size(3))
    for i = 1,a:size(1) do
      self.output:select(1,i):mm(a:select(1,i), b:select(1,i))
    end
  end

  return self.output
end

function MM:updateGradInput(input, gradOutput)
  self.gradInput[1] = self.gradInput[1] or input[1].new()
  self.gradInput[2] = self.gradInput[2] or input[2].new()

  assert(#input == 2, 'input must be a pair of tensors')
  local a, b = table.unpack(input)
  self.gradInput[1]:resizeAs(a)
  self.gradInput[2]:resizeAs(b)

  assert(gradOutput:nDimension() == 2 or gradOutput:nDimension() == 3, 'arguments must be a 2D or 3D Tensor')

  local h_dim, w_dim, f
  if gradOutput:nDimension() == 2 then
    assert(a:nDimension() == 2, 'first input tensor must be 2D')
    assert(b:nDimension() == 2, 'second input tensor must be 2D')

    h_dim, w_dim = 1, 2
    f = "mm"
  else
    assert(a:nDimension() == 3, 'first input tensor must be 3D')
    assert(b:nDimension() == 3, 'second input tensor must be 3D')

    h_dim, w_dim = 2, 3
    f = "bmm"
  end

  if self.transA == self.transB then
    a = a:transpose(h_dim, w_dim)
    b = b:transpose(h_dim, w_dim)
  end

  if self.transA then
    self.gradInput[1][f](self.gradInput[1], b, gradOutput:transpose(h_dim, w_dim))
  else
    if f == "bmm" then
      for i = 1, a:size(1) do
        self.gradInput[1]["mm"](self.gradInput[1]:select(1,i), gradOutput:select(1,i), b:select(1,i))
      end
    else
      self.gradInput[1]["mm"](self.gradInput[1], gradOutput, b)
    end
  end

  if self.transB then
    self.gradInput[2][f](self.gradInput[2], gradOutput:transpose(h_dim, w_dim), a)
  else
    if f == "bmm" then
      for i = 1, a:size(1) do
        self.gradInput[2]["mm"](self.gradInput[2]:select(1,i), a:select(1,i), gradOutput:select(1,i))
      end
    else
      self.gradInput[2]["mm"](self.gradInput[2], a, gradOutput)
    end
  end

  return self.gradInput
end
