//! GPT2 in Zig, Inspired by Karpathy
//! An implementation of the LLMs in Zig based on Andrej Karpathy's minGPT
//!
const std = @import("std");
const math = @import("std").math;
const builtin = @import("builtin");
const LlmFloat = f32; // To easily change the precision of the model
const FloatList = std.ArrayList(LlmFloat);
pub const UIntList = std.ArrayList(u32);

extern fn exit() noreturn;
const VectorSize: usize = 8; // This seems the best choice for the vectorization on M2 macbook pro
const RndGen = std.rand.DefaultPrng;

pub const VecType = @Vector(VectorSize, LlmFloat);
const Errors = error{InvalidModelHeader, InvalidModelVersion, InvalidTokensFile};
const program_name = "LLM GPT-2";
const NumParameterTensors = 16;
const NumActivationTensors = 23;
const FILE_HEADER = 20240326;
const FILE_VERSION = 1;
const FILE_HEADER_SIZE = 256;


const cfuncs = @cImport({
    @cInclude("math.h");
    });
const time = std.time;
const Instant = time.Instant;
const Timer = time.Timer;


const ParameterTensors = struct {
wte:        []LlmFloat, // (V, C)
wpe:        []LlmFloat, // (maxT, C)
ln1w:       []LlmFloat, // (L, C)
ln1b:       []LlmFloat, // (L, C)
qkvw:       []LlmFloat, // (L, 3*C, C)
qkvb:       []LlmFloat, // (L, 3*C)
attprojw:   []LlmFloat, // (L, C, C)
attprojb:   []LlmFloat, // (L, C)
ln2w:       []LlmFloat, // (L, C)
ln2b:       []LlmFloat, // (L, C)
fcw:        []LlmFloat, // (L, 4*C, C)
fcb:        []LlmFloat, // (L, 4*C)
fcprojw:    []LlmFloat, // (L, C, 4*C)
fcprojb:    []LlmFloat, // (L, C)
lnfw:       []LlmFloat, // (L, C)
lnfb:       []LlmFloat, // (L, C)
};

const ActivationTensors = struct {
encoded:    []LlmFloat, // (B, T, C)
ln1:        []LlmFloat, // (L, B, T, C)
ln1_mean:   []LlmFloat, // (L, B, T)
ln1_rstd:   []LlmFloat, // (L, B, T)
qkv:        []LlmFloat, // (L, B, T, 3*C)
atty:       []LlmFloat, // (L, B, T, C)
preatt:     []LlmFloat, // (L, B, NH, T, T)
att:        []LlmFloat, // (L, B, NH, T, T)
attproj:    []LlmFloat, // (L, B, T, C)
residual2:  []LlmFloat, // (L, B, T, C)
ln2:        []LlmFloat, // (L, B, T, C)
ln2_mean:   []LlmFloat, // (L, B, T)
ln2_rstd:   []LlmFloat, // (L, B, T)
fch:        []LlmFloat, // (L, B, T, 4*C)
fch_gelu:   []LlmFloat, // (L, B, T, 4*C)
fcproj:     []LlmFloat, // (L, B, T, C)
residual3:  []LlmFloat, // (L, B, T, C)
lnf:        []LlmFloat, // (B, T, C)
lnf_mean:   []LlmFloat, // (B, T)
lnf_rstd:   []LlmFloat, // (B, T)
logits:     []LlmFloat, // (B, T, V)
probs:      []LlmFloat, // (B, T, V)
losses:     []LlmFloat, // (B, T)
};

const GPT2Config = struct {
model_header:   u32,  // header
model_version:  u32,  // version
max_seq_len:    u32, // max sequence length, e.g. 1024
vocab_size:     u32,  // vocab size, e.g. 50257
num_layers:     u32,  // number of layers, e.g. 12
num_heads:      u32,  // number of heads in attention, e.g. 12
channels:       u32,  // number of channels, e.g. 768
};

const GPT2 = struct {
init_params:        bool,
init_grads:         bool,
init_grads_acts:    bool,
config:             GPT2Config,
params:             ParameterTensors,
params_sizes:       [NumParameterTensors]u32, // Change to zigtype isize
params_memory:      []LlmFloat,
num_parameters:     u32,
// gradients of the weights
grads:              ParameterTensors,
grads_memory:       []LlmFloat,
// buffers for the AdamW optimizer
m_memory:           []LlmFloat,
v_memory:           []LlmFloat,
init_adam:         bool,
// the activations of the model, and their sizes
acts:               ActivationTensors,
act_sizes:          [NumActivationTensors]u32,
acts_memory:        []LlmFloat,
num_activations:    u32,
// gradients of the activations
grads_acts:         ActivationTensors,
grads_acts_memory:  []LlmFloat,
// other run state configuration
batch_size:         u32, // the batch size (B) of current forward pass
seq_len:            u32, // the sequence length (T) of current forward pass
inputs:             []u32,  // the input tokens for the current forward pass
targets:            []u32, // the target tokens for the current forward pass
mean_loss:          LlmFloat, // after a forward pass with targets, will be populated with the mean loss
};

const DataLoader = struct {
B:                  u32, // batch size
T:                  u32, // sequence length
tokens_file:        std.fs.File,
file_size:          u64,
current_position:   u64,
batch:              []u32,
inputs:             []u32,
targets:            []u32,
num_batches:        u32,
raw_data:           []u8,
};

pub const Token = []u8;
pub const Tokenizer = struct {
vocab_size: u32,
vocab_map: [*][]u8,
init_ok: bool,
};

pub fn read_n_parameters_from_file(comptime T:type, file_name: [] const u8, N :usize, offset:usize) !std.ArrayList(T) {
    var file = try std.fs.cwd().openFile(file_name, .{ .mode = .read_only, });
    defer file.close();

    var file_size = try file.getEndPos();
    if (file_size < N * @sizeOf(T)) {
        return error.NotEnoughParameters;
        }
    if (file_size == 0) {
        return error.FileEmpty;
        }

    try file.seekTo(offset * @sizeOf(T));

    // Create an ArrayList to hold the data read.
    var data = try std.ArrayList(T).initCapacity(std.heap.page_allocator, N);
    try std.ArrayList(T).resize(&data, N);

    var bytes = std.mem.sliceAsBytes(data.items);
    _ = try file.read(bytes);
    return data;
    }

pub fn tokenizer_free(allocator: std.mem.Allocator, tokenizer:*Tokenizer) void{
    if(tokenizer.init_ok){
        for(0..tokenizer.vocab_size) |i| {
            allocator.free(tokenizer.vocab_map[i]);
            }
        }
    }

pub fn tokenizer_init(allocator: std.mem.Allocator, tokenizer:*Tokenizer, filename:[] const u8) !void{
    var model_header: UIntList = try read_n_parameters_from_file(u32,filename, FILE_HEADER_SIZE, 0);
    if (model_header.items[0] != 20240328){
        return Errors.InvalidModelHeader;
        }
    if (model_header.items[1] != 1){
        return Errors.InvalidModelVersion;
        }
    var file = try std.fs.cwd().openFile(filename, .{});
    defer file.close();
    try file.seekTo(FILE_HEADER_SIZE * @sizeOf(u32));

    tokenizer.vocab_size = model_header.items[2];
    var val = try allocator.alloc([]u8, tokenizer.vocab_size);
    tokenizer.vocab_map = val.ptr;

    for(0..tokenizer.vocab_size) |i| {
        var token_length: [1]u8 = undefined;
        //var token_data: [64]u8 = undefined;
        var read_token_length = try file.read(&token_length);

        if (@as(u32, @intCast(read_token_length)) == 0){
            return Errors.InvalidTokensFile;
            }
        var dyn_buffer = try allocator.alloc(u8, @as(u32, @intCast(token_length[0])));
        var read_token_bytes = try file.read(dyn_buffer);
        if (@as(u32, @intCast(read_token_bytes)) == 0){
            return Errors.InvalidTokensFile;
            }
        tokenizer.vocab_map[i] = dyn_buffer;
        }
    var sample_token = tokenizer.vocab_map[50001];
    std.debug.print("sample token: {s}\n", .{sample_token});
    tokenizer.init_ok = true;
    }


pub fn tokenizer_decode(tokenizer:Tokenizer, input:u32) []u8{
    if(tokenizer.init_ok){
        if (input < tokenizer.vocab_size){
            return tokenizer.vocab_map[input];
            }
        }
    return undefined;
    }
/// Print some the first and last elements of the parameters and the size of the parameters mostly for debugging
pub fn printParams(model: GPT2) void {
    std.debug.print("params.wte size        {}      first {d:.3}        last {d:.3}\n", .{model.params.wte.len, model.params.wte[0], model.params.wte[model.params.wte.len - 1]});
    std.debug.print("params.wpe size        {}      first {d:.3}        last {d:.3}\n", .{model.params.wpe.len, model.params.wpe[0], model.params.wpe[model.params.wpe.len - 1]});
    std.debug.print("params.ln1w size       {}      first {d:.3}        last {d:.3}\n", .{model.params.ln1w.len, model.params.ln1w[0], model.params.ln1w[model.params.ln1w.len - 1]});
    std.debug.print("params.ln1b size       {}      first {d:.3}        last {d:.3}\n", .{model.params.ln1b.len, model.params.ln1b[0], model.params.ln1b[model.params.ln1b.len - 1]});
    std.debug.print("params.qkvw size       {}      first {d:.3}        last {d:.3}\n", .{model.params.qkvw.len, model.params.qkvw[0], model.params.qkvw[model.params.qkvw.len - 1]});
    std.debug.print("params.qkvb size       {}      first {d:.3}        last {d:.3}\n", .{model.params.qkvb.len, model.params.qkvb[0], model.params.qkvb[model.params.qkvb.len - 1]});
    std.debug.print("params.attprojw size   {}      first {d:.3}        last {d:.3}\n", .{model.params.attprojw.len, model.params.attprojw[0], model.params.attprojw[model.params.attprojw.len - 1]});
    std.debug.print("params.attprojb size   {}      first {d:.3}        last {d:.3}\n", .{model.params.attprojb.len, model.params.attprojb[0], model.params.attprojb[model.params.attprojb.len - 1]});
    std.debug.print("params.ln2w size       {}      first {d:.3}        last {d:.3}\n", .{model.params.ln2w.len, model.params.ln2w[0], model.params.ln2w[model.params.ln2w.len - 1]});
    std.debug.print("params.ln2b size       {}      first {d:.3}        last {d:.3}\n", .{model.params.ln2b.len, model.params.ln2b[0], model.params.ln2b[model.params.ln2b.len - 1]});
    std.debug.print("params.fcw size        {}      first {d:.3}        last {d:.3}\n", .{model.params.fcw.len, model.params.fcw[0], model.params.fcw[model.params.fcw.len - 1]});
    std.debug.print("params.fcb size        {}      first {d:.3}        last {d:.3}\n", .{model.params.fcb.len, model.params.fcb[0], model.params.fcb[model.params.fcb.len - 1]});
    std.debug.print("params.fcprojw size    {}      first {d:.3}        last {d:.3}\n", .{model.params.fcprojw.len, model.params.fcprojw[0], model.params.fcprojw[model.params.fcprojw.len - 1]});
    std.debug.print("params.fcprojb size    {}      first {d:.3}        last {d:.3}\n", .{model.params.fcprojb.len, model.params.fcprojb[0], model.params.fcprojb[model.params.fcprojb.len - 1]});
    std.debug.print("params.lnfw size       {}      first {d:.3}        last {d:.3}\n", .{model.params.lnfw.len, model.params.lnfw[0], model.params.lnfw[model.params.lnfw.len - 1]});
    std.debug.print("params.lnfb size       {}      first {d:.3}        last {d:.3}\n", .{model.params.lnfb.len, model.params.lnfb[0], model.params.lnfb[model.params.lnfb.len - 1]});
    }

/// Encodes the input tokens into the model's input tensor by combining token embeddings and position embeddings.
/// This is often the first step in transformer models like GPT, where the input sequence is converted into
/// a more richly represented format for further processing.
/// @param output: The output buffer where the encoded tensor will be stored.
/// @param input: An array of token indices representing the input sequence.
/// @param wte: The token embedding weights (Vocabulary size x Embedding dimension).
/// @param wpe: The positional embedding weights (Max sequence length x Embedding dimension).
/// @param B: The batch size, indicating the number of sequences being processed simultaneously.
/// @param T: The sequence length of the input.
/// @param C: The embedding dimension or channel size of the embeddings.
pub fn encoder_forward( output : []LlmFloat, input: []u32, wte: []LlmFloat, wpe: []LlmFloat, B:u32, T:u32, C:u32) void
    {
    for(0..B) |b| {
        for(0..T) |t| {
            var out_bt = output[b * T * C + t * C..];
            // Get the index of the token at inp[b, t]
            var ix: u32 = input[b * T + t];
            // Seek to the position in wte corresponding to the token
            var wte_ix = wte[ix * C..];
            // Seek to the position in wpe corresponding to the position
            var wpe_t = wpe[t * C..];
            for(0..C) |c| {
                out_bt[c] = wte_ix[c] + wpe_t[c];
                }
            }
        }
    }

/// Vectorized
/// Encodes the input tokens into the model's input tensor by combining token embeddings and position embeddings.
/// This is often the first step in transformer models like GPT, where the input sequence is converted into
/// a more richly represented format for further processing.
/// @param output: The output buffer where the encoded tensor will be stored.
/// @param input: An array of token indices representing the input sequence.
/// @param wte: The token embedding weights (Vocabulary size x Embedding dimension).
/// @param wpe: The positional embedding weights (Max sequence length x Embedding dimension).
/// @param B: The batch size, indicating the number of sequences being processed simultaneously.
/// @param T: The sequence length of the input.
/// @param C: The embedding dimension or channel size of the embeddings.
pub fn encoder_forward_vec( comptime N: usize, output : []LlmFloat, input: []u32, wte: []LlmFloat, wpe: []LlmFloat,
                B:u32, T:u32, C:u32) void
    {
    for(0..B) |b| {
        for(0..T) |t| {
            for(0..C/N) |i| {
                var ix: u32 = input[b * T + t];
                const wte_ix : @Vector(N , f32)  = wte[ix * C + i*N..][0..N].*;
                const wpe_t : @Vector(N , f32)  = wpe[t * C + i*N..][0..N].*;
                const res: [N]f32 = wte_ix + wpe_t;
                var start = b * T * C + t * C + i * N;
                var end = start + N;
                @memcpy(output[start..end], &res);
                }
            }
        }
    }
/// Computes gradients for the encoder's forward pass. This is crucial for training, as it helps in optimizing
/// the token and position embeddings by backpropagating errors from the output towards the inputs.
/// @param dwte: Gradient with respect to the token embeddings.
/// @param dwpe: Gradient with respect to the positional embeddings.
/// @param dout: Gradient coming from the next layer (upstream gradients).
/// @param inp: Array of token indices (input to the forward function).
/// @param B: Batch size, as defined in the forward function.
/// @param T: Sequence length, as defined in the forward function.
/// @param C: Embedding dimension, as defined in the forward function.
pub fn encoder_backward( dwte: []LlmFloat, dwpe: []LlmFloat, dout: []LlmFloat, inp: []u32, B: u32, T: u32, C: u32) void
    {
    for (0..B) |b| {
        for (0..T) |t| {
            var dout_bt = dout[b * T * C + t * C ..];
            var ix: u32 = inp[b * T + t];
            var dwte_ix = dwte[ix * C ..];
            var dwpe_t = dwpe[t * C ..];
            for (0..C) |c| {
                var d = dout_bt[c];
                dwte_ix[c] += d;
                dwpe_t[c] += d;
                }
            }
        }
    }
/// Vectorized:
/// Computes gradients for the encoder's forward pass. This is crucial for training, as it helps in optimizing
/// the token and position embeddings by backpropagating errors from the output towards the inputs.
/// @param dwte: Gradient with respect to the token embeddings.
/// @param dwpe: Gradient with respect to the positional embeddings.
/// @param dout: Gradient coming from the next layer (upstream gradients).
/// @param inp: Array of token indices (input to the forward function).
/// @param B: Batch size, as defined in the forward function.
/// @param T: Sequence length, as defined in the forward function.
/// @param C: Embedding dimension, as defined in the forward function.
pub fn encoder_backward_vec( comptime N: u32, dwte: []LlmFloat, dwpe: []LlmFloat, dout: []LlmFloat, inp: []u32, B: u32,
                            T: u32, C: u32) void {
    for (0..B) |b| {
        for (0..T) |t| {
            for(0..C/N) |i| {
                var idx = b * T * C + t * C + i * N;
                var dout_bt = dout[idx..][0..N].*;
                var ix: usize = inp[b * T + t];
                var dwte_ix: @Vector(N , f32) = dwte[ix * C + i * N..][0..N].*;
                var dwpe_t: @Vector(N , f32) = dwpe[t * C + i * N..][0..N].*;
                const res_wte: [N]f32 = dout_bt + dwte_ix;
                const res_wpe: [N]f32 = dout_bt + dwpe_t;
                @memcpy(dwte[ix * C + i * N..][0..N], &res_wte);
                @memcpy(dwpe[t * C + i * N..][0..N], &res_wpe);
                }
            }
        }
    }
/// Applies layer normalization to the input tensor. Layer normalization is a standard technique in neural networks
/// to stabilize and accelerate training. It normalizes inputs across the features instead of the batch dimension.
/// @param output: The buffer where the normalized output will be stored.
/// @param mean: Buffer to store the computed mean of the input tensor for each sequence.
/// @param rstd: Buffer to store the reciprocal of the standard deviation (inverse standard deviation).
/// @param input: The input tensor to be normalized.
/// @param weight: The gamma parameter for scaling in the normalization.
/// @param bias: The beta parameter for shifting in the normalization.
/// @param B: Batch size.
/// @param T: Sequence length.
/// @param C: Number of features (embedding dimension in the context of transformers).
pub fn layernorm_forward(output: []LlmFloat, mean: []LlmFloat, rstd: []LlmFloat, input: []LlmFloat, weight: []LlmFloat,
                    bias: []LlmFloat,B: u32,T: u32,C: u32) void {
    var eps: LlmFloat = 1e-5;
    for (0..B) |b| {
        for (0..T) |t| {
            var x = input[b * T * C + t * C ..];
            var m: LlmFloat = 0.0;
            for (0..C) |i| {
                m += x[i];
                }
            m /= @floatFromInt( C);

            var v: LlmFloat = 0.0;
            for (0..C) |i| {
                var diff = x[i] - m;
                v += diff * diff;
                }
            v /= @floatFromInt( C);

            var s = 1.0 / math.sqrt(v + eps);

            var out = output[b * T * C + t * C ..];
            for (0..C) |c| {
                out[c] = (x[c] - m) * s * weight[c] + bias[c];
                }
            mean[b * T + t] = m;
            rstd[b * T + t] = s;
            }
        }
    }
/// Vectorized:
/// Applies layer normalization to the input tensor. Layer normalization is a standard technique in neural networks
/// to stabilize and accelerate training. It normalizes inputs across the features instead of the batch dimension.
/// @param output: The buffer where the normalized output will be stored.
/// @param mean: Buffer to store the computed mean of the input tensor for each sequence.
/// @param rstd: Buffer to store the reciprocal of the standard deviation (inverse standard deviation).
/// @param input: The input tensor to be normalized.
/// @param weight: The gamma parameter for scaling in the normalization.
/// @param bias: The beta parameter for shifting in the normalization.
/// @param B: Batch size.
/// @param T: Sequence length.
/// @param C: Number of features (embedding dimension in the context of transformers).
pub fn layernorm_forward_vec(comptime N: usize, output: []LlmFloat, mean: []LlmFloat, rstd: []LlmFloat, input: []LlmFloat,
                weight: []LlmFloat, bias: []LlmFloat, B: u32, T: u32, C: u32) void {
    var eps: LlmFloat = 1e-5;
    comptime var op :std.builtin.ReduceOp = .Add;
    for (0..B) |b| {
        for (0..T) |t| {
            var start:usize = b * T * C + t * C;
            var m: LlmFloat = 0.0; // mean
            var v: LlmFloat = 0.0; // variance
            for (0..C/N)|i|{
                var x:@Vector(N , f32) = input[start + i*N ..][0..N].*;
                m += @reduce(op, x);
                }

            m /= @floatFromInt( C);
            var v_m : @Vector(N, f32) = @splat(m);
            for (0..C/N)|i|{
                var x:@Vector(N , f32) = input[start + i*N ..][0..N].*;
                x -= v_m;
                v += @reduce(op, x * x);
                }

            v /= @floatFromInt( C);

            var s = 1.0 / math.sqrt(v + eps);

            var s_vector : @Vector(N, f32) = @splat(s);
            for (0..C/N)|i|{
                var diff:@Vector(N , f32) = input[start + i*N ..][0..N].*;
                diff -= v_m;
                var w_vector : @Vector(N, f32) = weight[i*N..][0..N].*;
                var bias_vector : @Vector(N, f32) = bias[i*N..][0..N].*;
                w_vector *= s_vector;
                var res: [N]f32 = @mulAdd(@Vector(N, f32),diff, w_vector, bias_vector);
                @memcpy(output[start + i*N ..start + i*N + N], &res);
                }
            mean[b * T + t] = m;
            rstd[b * T + t] = s;
            }
        }
    }

/// Computes the gradients for the layer normalization layer during the backward pass of training.
/// Layer normalization is used to stabilize the learning process by normalizing the inputs across the features.
/// @param dinp: Gradient buffer for the input activations where the computed gradients will be accumulated.
/// @param dweight: Gradient buffer for the scale parameters (gamma in some contexts), where gradients will be accumulated.
/// @param dbias: Gradient buffer for the shift parameters (beta in some contexts), where gradients will be accumulated.
/// @param doutput: The gradient received from the downstream layer, which needs to be back-propagated.
/// @param inp: The input tensor to the layer normalization function; required for computing gradients.
/// @param weight: The scale tensor applied during the forward pass.
/// @param mean: Computed mean of the inputs during the forward pass, required for backpropagation.
/// @param rstd: Computed reciprocal of the standard deviation (1/std) of the inputs during the forward pass.
/// @param B: Batch size, indicating the number of separate input sets.
/// @param T: Sequence or temporal length in the input.
/// @param C: Number of channels or features in the input, which corresponds to the dimensionality of the mean and std dev calculations.
pub fn layernorm_backward(dinp: []LlmFloat, dweight: []LlmFloat, dbias: []LlmFloat,doutput: []LlmFloat,inp:
            []LlmFloat,weight: []LlmFloat,mean: []LlmFloat,rstd: []LlmFloat,B: u32,T: u32,C: u32) void {
    for (0..B) |b| {
        for (0..T) |t| {
            var dout = doutput[b * T * C + t * C ..];
            var inp_bt = inp[b * T * C + t * C..];
            var dinp_bt = dinp[b * T * C + t * C..];
            var mean_bt = mean[b * T + t];
            var rstd_bt = rstd[b * T + t];

            var dnorm_mean:LlmFloat = 0.0;
            var dnorm_norm_mean:LlmFloat = 0.0;
            for (0..C) |c| {
                var norm_bti = (inp_bt[c] - mean_bt) * rstd_bt;
                var dnorm_i = weight[c] * dout[c];
                dnorm_mean += dnorm_i;
                dnorm_norm_mean += dnorm_i * norm_bti;
                }
            dnorm_mean /= @floatFromInt(C);
            dnorm_norm_mean /= @floatFromInt(C);

            for (0..C) |c| {
                var norm_bti = (inp_bt[c] - mean_bt) * rstd_bt;
                var dnorm_i = weight[c] * dout[c];
                dbias[c] += dout[c];
                dweight[c] += norm_bti * dout[c];
                var dval = dnorm_i - dnorm_mean - norm_bti * dnorm_norm_mean;
                dval *= rstd_bt;
                dinp_bt[c] += dval;
                }
            }
        }
    }

/// Computes the forward pass of a matrix multiplication for a mini-batch. This operation is central in
/// neural networks, especially in fully connected and convolutional layers (transformed into matrix multiplications).
/// @param output: The output buffer where the results will be stored.
/// @param input: The input tensor that contains data from the previous layer.
/// @param weight: The weight matrix to be multiplied with the input.
/// @param bias: The bias vector to be added after matrix multiplication, can be optional.
/// @param B: Batch size, number of separate data entries processed.
/// @param T: Sequence length, typically the number of time steps in sequence data.
/// @param C: Number of input channels or features per time step.
/// @param OC: Number of output channels, corresponding to the number of neurons in a fully connected layer.
pub fn matmul_forward( output: []LlmFloat, input: []LlmFloat, weight: []LlmFloat, bias: []LlmFloat, B: u32, T: u32,
        C: u32, OC: u32,
) void{

    for(0..B) |b| {
        for(0..T) |t| {
            var out_bt:[]LlmFloat = output[b * T * OC + t * OC..];
            var inp_bt:[]LlmFloat = input[b * T * C + t * C..];
            for(0..OC) |o| {
                var val:LlmFloat = 0.0;
                if(bias.len != 0) {
                    val = bias[o];
                    }
                var wrow:[]LlmFloat = weight[o * C..];
                for(0..C) |i| {
                    val += inp_bt[i] * wrow[i];
                    }
                out_bt[o] = val;
                }
            }
        }
    }
/// Vectorized
/// Computes the forward pass of a matrix multiplication for a mini-batch. This operation is central in
/// neural networks, especially in fully connected and convolutional layers (transformed into matrix multiplications).
/// @param output: The output buffer where the results will be stored.
/// @param input: The input tensor that contains data from the previous layer.
/// @param weight: The weight matrix to be multiplied with the input.
/// @param bias: The bias vector to be added after matrix multiplication, can be optional.
/// @param B: Batch size, number of separate data entries processed.
/// @param T: Sequence length, typically the number of time steps in sequence data.
/// @param C: Number of input channels or features per time step.
/// @param OC: Number of output channels, corresponding to the number of neurons in a fully connected layer.
pub fn matmul_forward_vec( comptime N : usize,output: []LlmFloat,input: []LlmFloat,weight: []LlmFloat,bias: []LlmFloat,
                B: usize,T: usize,C: usize,OC: usize
) void {
    for (0..B) |b| {
        for (0..T) |t| {
            var out_bt = output[b * T * OC + t * OC ..];
            for (0..OC) |o| {
                for (0..C/N) |i| {
                    const inp_bt_v : @Vector(N , LlmFloat) = input[b * T * C + t * C + i*N ..][0..N].*;
                    const wrow_v : @Vector(N , LlmFloat) = weight[o * C + i*N ..][0..N].*;
                    const res = @mulAdd(@Vector(N, LlmFloat), wrow_v, inp_bt_v, @splat(0.0));
                    var final_res = @reduce(.Add, res) ;
                    out_bt[o] += final_res;
                    }
                if(bias.len != 0) {
                    out_bt[o] += bias[o];
                    }
                }
            }
        }
    }
/// Computes the gradients for the matrix multiplication operation in the backward pass of training.
/// It back-propagates gradients through the network, adjusting weights and biases based on the error
/// relative to the expected output.
/// @param dinp: Gradient buffer for the inputs, where gradients will be accumulated.
/// @param dweight: Gradient buffer for the weights, where gradients will be accumulated.
/// @param dbias: Gradient buffer for the biases, where gradients will be accumulated (if bias is used).
/// @param dout: The gradient received from the downstream layer (back-propagated error).
/// @param inp: The input tensor to the forward function, required for gradient computation.
/// @param weight: The weight matrix used in the forward pass, required for gradient computation.
/// @param B: Batch size.
/// @param T: Sequence length.
/// @param C: Number of input channels or features.
/// @param OC: Number of output channels.
pub fn matmul_backward( dinp: []LlmFloat, dweight: []LlmFloat, dbias: []LlmFloat, dout: []LlmFloat, inp: []LlmFloat,
                weight: []LlmFloat, B: u32, T: u32, C: u32, OC: u32) void{
    for(0..B) |b| {
        for(0..T) |t| {
            var dout_bt = dout[b * T * OC + t * OC..];
            var dinp_bt = dinp[b * T * C + t * C..];
            for(0..OC) |o| {
                var wrow = weight[o * C..];
                var d:LlmFloat = dout_bt[o];
                for(0..C) |i| {
                    dinp_bt[i] += wrow[i] * d;
                    }
                }
            }
        }
    for(0..OC) |o| {
        for(0..B) |b| {
            for(0..T) |t| {
                var dout_bt = dout[b * T * OC + t * OC..];
                var inp_bt = inp[b * T * C + t * C..];
                var dwrow = dweight[o * C..];
                var d:LlmFloat = dout_bt[o];
                if(dbias.len != 0) {
                    dbias[o] += d;
                    }
                for(0..C) |i| {
                    dwrow[i] += inp_bt[i] * d;
                    }
                }
            }
        }
    }
/// Vectorized
/// Computes the gradients for the matrix multiplication operation in the backward pass of training.
/// It back-propagates gradients through the network, adjusting weights and biases based on the error
/// relative to the expected output.
/// @param dinp: Gradient buffer for the inputs, where gradients will be accumulated.
/// @param dweight: Gradient buffer for the weights, where gradients will be accumulated.
/// @param dbias: Gradient buffer for the biases, where gradients will be accumulated (if bias is used).
/// @param dout: The gradient received from the downstream layer (back-propagated error).
/// @param inp: The input tensor to the forward function, required for gradient computation.
/// @param weight: The weight matrix used in the forward pass, required for gradient computation.
/// @param B: Batch size.
/// @param T: Sequence length.
/// @param C: Number of input channels or features.
/// @param OC: Number of output channels.
pub fn matmul_backward_vec( comptime N: usize, dinp: []LlmFloat, dweight: []LlmFloat, dbias: []LlmFloat, dout: []LlmFloat,
                inp: []LlmFloat, weight: []LlmFloat, B: u32, T: u32, C: u32, OC: usize) void{
    for(0..B) |b| {
        for(0..T) |t| {
            var dout_bt = dout[b * T * OC + t * OC..];
            var dinp_bt_v : @Vector(N , LlmFloat)  = undefined;
            for(0..OC) |o| {

                for(0..C/N) |i| {
                    dinp_bt_v = dinp[b * T * C + t * C + i * N..][0..N].*;
                    var v_wrow : @Vector(N, LlmFloat) = weight[o * C + i * N..][0..N].*;
                    var d_t : @Vector(N, LlmFloat) = @splat(dout_bt[o]);
                    var zero : @Vector(N, LlmFloat) = @splat(0);
                    dinp_bt_v +=  @mulAdd(@Vector(N, f32),v_wrow, d_t,zero);
                    var tmp:[N]LlmFloat = dinp_bt_v;
                    @memcpy(dinp[b * T * C + t * C + i * N..b * T * C + t * C + (i+1) * N], &tmp);
                    }
                }
            }
        }
    for(0..OC) |o| {
        for(0..B) |b| {
            for(0..T) |t| {
                var dout_bt = dout[b * T * OC + t * OC..];
                for (0..C/N) |i| {
                    var dv_wrow : @Vector(N, LlmFloat) = dweight[o * C + i * N..][0..N].*;

                    var inp_bt_v : @Vector(N , LlmFloat) = inp[b * T * C + t * C + i * N ..][0..N].*;
                    var d_t : @Vector(N, LlmFloat) = @splat(dout_bt[o]);

                    dv_wrow += inp_bt_v * d_t;
                    var tmp:[N]LlmFloat = dv_wrow;
                    @memcpy(dweight[o * C + i * N ..o * C + (i+1)*N], &tmp);
                    }

                if(dbias.len != 0) {
                    dbias[o] += dout_bt[o];
                    }
                }
            }
        }
    }
/// Computes the attention mechanism for a transformer model.
/// The function handles multi-head self-attention which is a core component of the transformer architecture.
/// It calculates the weighted sum of values based on the softmax of the dot products of queries and keys.
///
/// @param output: The resulting output after applying attention and weighted sum.
/// @param preatt: The pre-softmax attention scores which are computed as dot products of queries and keys.
/// @param att: The post-softmax attention scores.
/// @param inp: Concatenated query, key, and value vectors for all heads.
/// @param B: Batch size, denoting the number of sequences.
/// @param T: Sequence length.
/// @param C: Dimension of each input token, which is split into parts for query, key, and value.
/// @param NH: Number of attention heads.
pub fn attention_forward( output : []LlmFloat, preatt: []LlmFloat, att: []LlmFloat, inp: []LlmFloat, B: u32, T: u32,
        C: u32, NH: u32
) void{
    var C3:u32 = C*3;
    var hs:u32 = C / NH; // head size
    var hs_float:LlmFloat = @floatFromInt(hs);
    var scale:LlmFloat = 1.0 / math.sqrt(hs_float);

    for(0..B) |b| {
        for(0..T) |t| {
            for(0..NH) |h| {
                var query_t = inp[b * T * C3 + t * C3 + h * hs..];
                var preatt_bth = preatt[b * NH * T * T + h * T * T + t * T..];
                var att_bth = att[b * NH * T * T + h * T * T + t * T..];
                var maxval = -math.floatMin(LlmFloat);
                for(0..t+1) |t2| {
                    var key_t2 = inp[b * T * C3 + t2 * C3 + h * hs + C..];
                    var val:LlmFloat = 0.0;
                    for(0..hs) |i| {
                        var q:LlmFloat = query_t[i];
                        var k:LlmFloat = key_t2[i];
                        val += k * q;
                        }
                    val *= scale;
                    if(val > maxval) {
                        maxval = val;
                        }
                    preatt_bth[t2] = val;
                    }
                var expsum:LlmFloat = 0.0;
                for(0..t+1) |t2| {
                    var expv:LlmFloat = math.exp(preatt_bth[t2] - maxval);
                    expsum += expv;
                    att_bth[t2] = expv;
                    }
                var expsum_inv:LlmFloat = 0.0;
                if(expsum != 0.0) {
                    expsum_inv = 1.0 / expsum;
                    }
                for(0..T) |t2| {
                    if(t2 <= t) {
                        att_bth[t2] *= expsum_inv;
                        } else {
                        att_bth[t2] = 0.0;
                        }
                    }
                var out_bth = output[b * T * C + t * C + h * hs..];
                for(0..hs) |i| {
                    out_bth[i] = 0.0;
                    }
                for(0..t+1) |t2| {
                    var value_t2 = inp[b * T * C3 + t2 * C3 + h * hs + C * 2..];
                    var att_btht2:LlmFloat = att_bth[t2];
                    for(0..hs) |i| {
                        out_bth[i] += att_btht2 * value_t2[i];
                        }
                    }
                }
            }
        }
    }
/// Vectorized
/// Computes the attention mechanism for a transformer model.
/// The function handles multi-head self-attention which is a core component of the transformer architecture.
/// It calculates the weighted sum of values based on the softmax of the dot products of queries and keys.
///
/// @param output: The resulting output after applying attention and weighted sum.
/// @param preatt: The pre-softmax attention scores which are computed as dot products of queries and keys.
/// @param att: The post-softmax attention scores.
/// @param inp: Concatenated query, key, and value vectors for all heads.
/// @param B: Batch size, denoting the number of sequences.
/// @param T: Sequence length.
/// @param C: Dimension of each input token, which is split into parts for query, key, and value.
/// @param NH: Number of attention heads.
pub fn attention_forward_vec( comptime N: usize, output : []LlmFloat, preatt: []LlmFloat, att: []LlmFloat, inp: []LlmFloat,
            B: usize, T: usize, C: usize, NH: usize
) void{
    var C3:usize = C*3;
    var hs:usize = C / NH; // head size
    var hs_float:LlmFloat = @floatFromInt(hs);
    var scale:LlmFloat = 1.0 / math.sqrt(hs_float);
    const add_op : std.builtin.ReduceOp = std.builtin.ReduceOp.Add;
    for(0..B) |b| {
        for(0..T) |t| {
            for(0..NH) |h| {
                var query_t = inp[b * T * C3 + t * C3 + h * hs..];
                var preatt_bth = preatt[b * NH * T * T + h * T * T + t * T..];
                var att_bth = att[b * NH * T * T + h * T * T + t * T..];
                var maxval = -math.floatMin(LlmFloat);
                for(0..t+1) |t2| {
                    var key_t2 = inp[b * T * C3 + t2 * C3 + h * hs + C..];
                    var val:LlmFloat = 0.0;
                    for(0..hs/N) |i| {
                        var q:@Vector(N,LlmFloat) = query_t[i*N..][0..N].*;
                        var k:@Vector(N,LlmFloat)  = key_t2[i*N..][0..N].*;
                        val +=@reduce(add_op,q*k);
                        }
                    val *= scale;
                    if(val > maxval) {
                        maxval = val;
                        }
                    preatt_bth[t2] = val;
                    }
                var expsum:LlmFloat = 0.0;
                for(0..t+1) |t2| {
                    var expv:LlmFloat = math.exp(preatt_bth[t2] - maxval);
                    expsum += expv;
                    att_bth[t2] = expv;
                    }
                var expsum_inv:LlmFloat = 0.0;
                if(expsum != 0.0) {
                    expsum_inv = 1.0 / expsum;
                    }
                for(0..T) |t2| {
                    if(t2 <= t) {
                        att_bth[t2] *= expsum_inv;
                        } else {
                        att_bth[t2] = 0.0;
                        }
                    }
                var out_bth = output[b * T * C + t * C + h * hs..];
                for(0..hs) |i| {
                    out_bth[i] = 0.0;
                    }
                for(0..t+1) |t2| {
                    var att_btht2:@Vector(N,LlmFloat) = @splat(att_bth[t2]);
                    for(0..hs/N) |i| {
                        var value_t2:@Vector(N,LlmFloat) = inp[b * T * C3 + t2 * C3 + h * hs + C * 2 + i*N..][0..N].*;
                        var out_bth_v:@Vector(N,LlmFloat) = out_bth[i*N..][0..N].*;
                        var res:[N]LlmFloat = @mulAdd(@Vector(N,LlmFloat), att_btht2, value_t2, out_bth_v);
                        @memcpy(out_bth[i*N..][0..N], &res);
                        }
                    }
                }
            }
        }
    }
/// Computes gradients for the attention mechanism during the backward pass.
/// This function backpropagates errors from the output of the attention layer to the inputs,
/// which include queries, keys, and values, and computes gradients for these components.
///
/// @param dinp: Gradient buffer for the input activations.
/// @param dpreatt: Gradient buffer for pre-softmax attention scores.
/// @param datt: Gradient buffer for post-softmax attention scores.
/// @param dout: Gradient buffer for the output activations.
/// @param inp: Input activations to the attention layer.
/// @param att: Post-softmax attention scores computed during the forward pass.
/// @param B: Batch size.
/// @param T: Sequence length.
/// @param C: Channel size of the input.
/// @param NH: Number of attention heads.
pub fn attention_backward( dinp: []LlmFloat, dpreatt: []LlmFloat, datt: []LlmFloat, dout: []LlmFloat, inp: []LlmFloat,
            att: []LlmFloat, B: u32, T: u32, C: u32, NH: u32) void{
    var C3:u32 = C * 3;
    var hs:u32 = @intCast(C / NH);
    var hs_float:LlmFloat = @floatFromInt(hs);
    var scale:LlmFloat = 1.0 / math.sqrt(hs_float);
    for(0..B)|b|{
        for(0..T)|t|{
            for(0..NH)|h|{
                var att_bth    = att[b*NH*T*T + h*T*T + t*T..];
                var datt_bth   = datt[b*NH*T*T + h*T*T + t*T..];
                var dpreatt_bth = dpreatt[b*NH*T*T + h*T*T + t*T..];
                var dquery_t   = dinp[b*T*C3 + t*C3 + h*hs..];
                var query_t    = inp[b*T*C3 + t*C3 + h*hs..];
                // backward pass 4, through the value accumulation
                var dout_bth = dout[b * T * C + t * C + h * hs..];
                for(0..t+1)|t2|{
                    var value_t2 = inp[b*T*C3 + t2*C3 + h*hs + C*2..];
                    var dvalue_t2 = dinp[b*T*C3 + t2*C3 + h*hs + C*2..];
                    for(0..hs)|i|{
                        datt_bth[t2] += value_t2[i] * dout_bth[i];
                        dvalue_t2[i] += att_bth[t2] * dout_bth[i];
                        }
                    }
                for (0..t+1) |t2| {
                    for (0..t+1) |t3|{
                        var indicator:LlmFloat = 0.0;
                        if(t3 == t2){
                            indicator = 1.0;
                            }
                        var local_derivative:LlmFloat = att_bth[t2] * (indicator - att_bth[t3]);
                        dpreatt_bth[t3] += local_derivative * datt_bth[t2];
                        }
                    }
                for(0..t+1)|t2|{
                    var key_t2 = inp[b * T * C3 + t2 * C3 + h * hs + C..];
                    var dkey_t2 = dinp[b * T * C3 + t2 * C3 + h * hs + C..];
                    for(0..hs)|i|{
                        dquery_t[i] += key_t2[i] * dpreatt_bth[t2]*scale;
                        dkey_t2[i] += query_t[i] * dpreatt_bth[t2]*scale;
                        }
                    }
                }
            }
        }
    }
/// Vectorized
/// Computes gradients for the attention mechanism during the backward pass.
/// This function backpropagates errors from the output of the attention layer to the inputs,
/// which include queries, keys, and values, and computes gradients for these components.
///
/// @param dinp: Gradient buffer for the input activations.
/// @param dpreatt: Gradient buffer for pre-softmax attention scores.
/// @param datt: Gradient buffer for post-softmax attention scores.
/// @param dout: Gradient buffer for the output activations.
/// @param inp: Input activations to the attention layer.
/// @param att: Post-softmax attention scores computed during the forward pass.
/// @param B: Batch size.
/// @param T: Sequence length.
/// @param C: Channel size of the input.
/// @param NH: Number of attention heads.
pub fn attention_backward_vec( comptime NChannelsPerHead: usize, dinp: []LlmFloat, dpreatt: []LlmFloat, datt: []LlmFloat,
                        dout: []LlmFloat, inp: []LlmFloat, att: []LlmFloat, B: u32, T: u32, C: u32, NH: u32) void{
    var C3:u32 = C * 3;
    var hs:u32 = @intCast(C / NH);
    var hs_float:LlmFloat = @floatFromInt(hs);
    var scale:LlmFloat = 1.0 / math.sqrt(hs_float);
    for(0..B)|b|{
        for(0..T)|t|{
            for(0..NH)|h|{
                var att_bth    = att[b*NH*T*T + h*T*T + t*T..];
                var datt_bth   = datt[b*NH*T*T + h*T*T + t*T..];
                var dpreatt_bth = dpreatt[b*NH*T*T + h*T*T + t*T..];
                var dquery_t   = dinp[b*T*C3 + t*C3 + h*hs..];
                var query_t    = inp[b*T*C3 + t*C3 + h*hs..];
                // backward pass 4, through the value accumulation
                var dout_bth = dout[b * T * C + t * C + h * hs..];
                for(0..t+1)|t2|{
                    var value_t2 = inp[b*T*C3 + t2*C3 + h*hs + C*2..];
                    var dvalue_t2 = dinp[b*T*C3 + t2*C3 + h*hs + C*2..];
                    comptime var op_add :std.builtin.ReduceOp = .Add;

                    for (0..hs/NChannelsPerHead)|i|{
                        var v_dout_bth : VecType = dout_bth[i*NChannelsPerHead..][0..NChannelsPerHead].*;
                        var v_value_t2 : VecType = value_t2[i*NChannelsPerHead..][0..NChannelsPerHead].*;
                        var v_dvalue_t2 : VecType = dvalue_t2[i*NChannelsPerHead..][0..NChannelsPerHead].*;
                        var res_value_outatt = @reduce(op_add,v_value_t2 * v_dout_bth);
                        datt_bth[t2] += res_value_outatt;

                        const v_att_bth : VecType = @splat(att_bth[t2]);
                        var res:[NChannelsPerHead]f32 = v_dvalue_t2 + v_dout_bth * v_att_bth;
                        var start = b*T*C3 + t2*C3 + h*hs + C*2 + i*NChannelsPerHead;
                        var end = b*T*C3 + t2*C3 + h*hs + C*2 + (i+1)*NChannelsPerHead;
                        @memcpy(dinp[start..end], &res);
                        }
                    }
                for (0..t+1) |t2| {
                    for (0..t+1) |t3|{
                        var indicator:LlmFloat = 0.0;
                        if(t3 == t2){
                            indicator = 1.0;
                            }
                        var local_derivative:LlmFloat = att_bth[t2] * (indicator - att_bth[t3]);
                        dpreatt_bth[t3] += local_derivative * datt_bth[t2];
                        }
                    }
                for(0..t+1)|t2|{
                    var key_t2 = inp[b * T * C3 + t2 * C3 + h * hs + C..];
                    var dkey_t2 = dinp[b * T * C3 + t2 * C3 + h * hs + C..];
                    for (0..hs/NChannelsPerHead)|i|{
                        var v_query_t : VecType = query_t[i*NChannelsPerHead..][0..NChannelsPerHead].*;
                        var v_key_t2 : VecType = key_t2[i*NChannelsPerHead..][0..NChannelsPerHead].*;
                        var v_dkey_t2 : VecType = dkey_t2[i*NChannelsPerHead..][0..NChannelsPerHead].*;
                        var v_dquery_t : VecType = dquery_t[i*NChannelsPerHead..][0..NChannelsPerHead].*;
                        var v_scale_dpreatt : VecType = @splat(dpreatt_bth[t2]*scale);
                        var res_dquery:[NChannelsPerHead]f32 = v_dquery_t + v_key_t2 * v_scale_dpreatt;
                        @memcpy(dquery_t[i*NChannelsPerHead..][0..NChannelsPerHead], &res_dquery);
                        var res_dkey_t2:[NChannelsPerHead]f32 = v_dkey_t2 + v_query_t * v_scale_dpreatt;
                        @memcpy(dkey_t2[i*NChannelsPerHead..][0..NChannelsPerHead], &res_dkey_t2);
                        }
                    }
                }
            }
        }
    }

/// Applies the Gaussian Error Linear Unit (GELU) activation function on the input data.
/// GELU is used in transformer models and is defined as x * Φ(x), where Φ(x) is the cumulative distribution function
/// of the standard normal distribution.
///
/// @param output: The array where the output will be stored.
/// @param input: The input array on which the GELU function is applied.
/// @param N: The number of elements in the input and output arrays.
pub fn gelu_forward( output: []LlmFloat, input: []LlmFloat, N: u32) void {
    comptime var s = math.sqrt(2.0 / math.pi);
    for (0..N) |i| {
        var x = input[i];
        var cdf = 0.5 * (1.0 + math.tanh(s * (x + 0.044715 * math.pow(LlmFloat,x, 3))));
        output[i] = x * cdf;
        }
    }

/// Vectorized:
/// Applies the Gaussian Error Linear Unit (GELU) activation function on the input data.
/// GELU is used in transformer models and is defined as x * Φ(x), where Φ(x) is the cumulative distribution function
/// of the standard normal distribution.
///
/// @param output: The array where the output will be stored.
/// @param input: The input array on which the GELU function is applied.
/// @param N: The number of elements in the input and output arrays.
//ToDo Fix issue with tanh producing NaN in the following implementation. Zig does not have an operator for tanh operating in vector mode.
pub fn gelu_forward_vec(comptime N:usize, output: []LlmFloat, input: []LlmFloat, BT4C: usize) void {
    var s:@Vector(N , LlmFloat) = @splat(math.sqrt(2.0 / math.pi));
    var half:@Vector(N , LlmFloat) = @splat(0.5);
    var one: @Vector(N , LlmFloat) = @splat(1.0);
    var g_coeff: @Vector(N , LlmFloat) = @splat(0.044715);
    for(0..BT4C/N) |i| {
        var x: @Vector(N , f32) = input[i*N..][0..N].*;
        var x_cube: @Vector(N , f32) = x * x * x; // x^3
        var x_par: @Vector(N , f32) =  s * (x + g_coeff * x_cube); // s*(x+0.044715*x^3)
        var tanh_x: @Vector(N , f32) =(@exp(x_par) - @exp(-x_par))/(@exp(x_par) + @exp(-x_par)) ;
        var cdf = half * (one + tanh_x);
        var res: [N]LlmFloat = x * cdf;
        @memcpy(output[i*N..(i+1)*N], &res);
        }
    }

/// Computes the gradient of the GELU activation function with respect to the input tensor, using the gradients
/// from the next layer. This function is used in the backpropagation process to propagate gradients back through
/// the network for the GELU activation nodes.
///
/// @param dinput: The gradient with respect to the input of the GELU function, which this function computes.
/// @param input: The input tensor to the GELU function from the forward pass.
/// @param doutput: The gradient with respect to the output of the GELU function, received from the next layer.
/// @param N: The number of elements in the input and output gradient arrays.
pub fn gelu_backward( dinput: []LlmFloat, input: []LlmFloat, doutput: []LlmFloat, N: u32)
void {
    comptime var s = math.sqrt(2.0 / math.pi);
    for (0..N) |i| {
        var x = input[i];
        var square = x * x * 0.044715;
        var cube = square * x;
        var tanh_arg = s * (x + cube);
        var tanh_out = math.tanh(tanh_arg);
        var coshf_out = math.cosh(tanh_arg);
        var sech2 = 1.0 / (coshf_out * coshf_out);
        var local_grad = 0.5 * (1.0 + tanh_out) + x * 0.5 * sech2 * s * (1.0 + 3.0 * square);
        dinput[i] += local_grad * doutput[i];
        }
    }

/// Adds the residual connection to the input tensor. This function performs an element-wise addition of the input
/// tensor and a residual tensor, often used in neural networks to help in training deep architectures by allowing
/// gradients to flow through the network mitigating the vanishing of the gradients.
///
/// @param output: Output tensor where the result of the addition is stored.
/// @param input: Input tensor to which the residual connection is added.
/// @param residual: Residual tensor that is added to the input.
/// @param N: Number of elements in each tensor.
fn residual_forward( output:[]LlmFloat, input:[]LlmFloat, residual:[]LlmFloat, N:u32) void{
    for(0..N) |i| {
        output[i] = input[i] + residual[i];
        }
    }
/// Vectorized
/// Adds the residual connection to the input tensor. This function performs an element-wise addition of the input
/// tensor and a residual tensor, often used in neural networks to help in training deep architectures by allowing
/// gradients to flow through the network mitigating the vanishing of the gradients.
///
/// @param output: Output tensor where the result of the addition is stored.
/// @param input: Input tensor to which the residual connection is added.
/// @param residual: Residual tensor that is added to the input.
/// @param N: Number of elements in each tensor.
fn residual_forward_vec( comptime N:usize, output:[]LlmFloat, input:[]LlmFloat, residual:[]LlmFloat, BTC:u32) void{
    for(0..BTC/VectorSize) |i| {
        var inp:@Vector(N , f32) = input[i*N..][0..N].*;
        var res:@Vector(N , f32) = residual[i*N..][0..N].*;
        var tmp: [VectorSize]f32  = inp + res;
        @memcpy(output[i*VectorSize..(i+1)*VectorSize], &tmp);
        }
    }
/// Computes the backward pass for a residual layer in a neural network. This function is used during
/// backpropagation to distribute the gradients through the skip connections (residual connections).
///
/// Residual connections help in training deep neural networks by allowing gradients to flow through
/// multiple layers unimpeded by the vanishing gradient problem.
///
/// @param dinput: Gradient with respect to the input of the residual block, which will be updated by this function.
/// @param dresidual: Gradient with respect to the output of the residual block, which will be updated by this function.
/// @param doutput: Gradient of the loss with respect to the output of the residual block.
/// @param N: The size of the array, typically matching the number of neurons or elements in a layer.
fn residual_backward( dinput:[]LlmFloat, dresidual:[]LlmFloat, doutput:[]LlmFloat, N:u32) void{
    for(0..N) |i| {
        dinput[i] += doutput[i];
        dresidual[i] += doutput[i];
        }
    }
/// Applies the softmax function to logits to transform them into something the community calls them probabilities,
/// but to me is just a bunch of numbers that sum to 1.
/// The softmax function is computed over the last dimension of the logits tensor
///
/// @param probs: Output tensor where the softmax probabilities are stored.
/// @param logits: Input tensor containing logits over which softmax is to be applied.
/// @param B: Batch size.
/// @param T: Sequence length.
/// @param V: Number of classes or vocabulary size in the context of language models.
fn softmax_forward( probs:[]LlmFloat, logits:[]LlmFloat, B:u32, T:u32, V:u32) void
    {
    for(0..B) |b| {
        for(0..T) |t| {
            var logit_bt = logits[b * T * V + t * V..];
            var prob_bt = probs[b * T * V + t * V..];
            var maxval = -math.floatMax(LlmFloat);
            for(0..V) |v| {
                if(logit_bt[v] > maxval) {
                    maxval = logit_bt[v];
                    }
                }
            var expsum:LlmFloat = 0.0;
            for(0..V) |v| {
                var expv:LlmFloat = math.exp(logit_bt[v] - maxval);
                expsum += expv;
                prob_bt[v] = expv;
                }
            var expsum_inv:LlmFloat = 0.0;
            if(expsum != 0.0) {
                expsum_inv = 1.0 / expsum;
                }
            for(0..V) |v| {
                prob_bt[v] *= expsum_inv;
                }
            }
        }
    }
/// Computes the cross-entropy loss for each example in a batch.
///
/// @param losses: An array where computed loss values will be stored.
/// @param probs: The something called probabilities associated.
/// @param targets: The true class indices for each example.
/// @param B: The number of examples in a batch.
/// @param T: The number of time steps or sequence length in each example (used in sequence models).
/// @param V: The number of classes or the vocabulary size.
fn crossentropy_forward( losses: []LlmFloat, probs: []LlmFloat, targets: []u32, B: u32, T: u32, V: u32) void
    {
    for (0..B) |b| {
        for (0..T) |t| {
            var prob_bt = probs[b * T * V + t * V ..];
            var target_bt = targets[b * T + t];
            losses[b * T + t] = -@log(@max(prob_bt[target_bt], 1e-10)); // Avoid log(0)
            }
        }
    }
/// Backpropagates errors through a softmax layer combined with a cross-entropy loss function.
/// This function calculates the gradients of the loss with respect to the logits (inputs to the softmax function),
/// which are used to update the model during training.
///
/// @param dlogits: Gradient of the loss with respect to the logits, which will be updated by this function.
/// @param dlosses: Gradient of the loss with respect to the output of the network (loss values).
/// @param probs: Probabilities obtained from the forward pass of softmax.
/// @param targets: The true class indices for each example.
/// @param B: Number of examples in a batch.
/// @param T: Number of time steps or sequence length in each example (used in sequence models).
/// @param V: Number of classes or vocabulary size.
fn crossentropy_softmax_backward( dlogits: []LlmFloat, dlosses: []LlmFloat, probs: []LlmFloat, targets: []u32, B: u32,
                T: u32, V: u32) void
    {
    for (0..B) |b| {
        for (0..T) |t| {
            var dlogits_bt = dlogits[b * T * V + t * V ..];
            var prob_bt = probs[b * T * V + t * V ..];
            var ix = targets[b * T + t];
            var dloss = dlosses[b * T + t];
            for (0..V) |v| {
                var p = prob_bt[v];
                var indicator: LlmFloat = 0.0;
                if (v == ix) {
                    indicator = 1.0;
                    }
                dlogits_bt[v] += (p - indicator) * dloss;
                }
            }
        }
    }
/// Print the header of the GPT-2 model.
/// This function prints the header of the GPT-2 model, which includes information such as the model header,
/// version, maximum sequence length, vocabulary size, number of layers, number of attention heads, and the number of channels.
/// @param model: The GPT-2 model for which the header is to be printed.
pub fn printHead(model: GPT2) void {
    std.debug.print("[GPT-2]\nheader: {}\nversion: {}\nmax_seq_len: {}\nvocab_size: {}\nnum_layers: {}\nnum_heads: {}\nchannels: {}\n", .{
        model.config.model_header,
        model.config.model_version,
        model.config.max_seq_len,
        model.config.vocab_size,
        model.config.num_layers,
        model.config.num_heads,
        model.config.channels,
        });
    }

/// Initializes a GPT2 model from a checkpoint file. This function sets up the model's configuration,
/// allocates memory for its parameters based on the configuration, and loads these parameters from the checkpoint.
///
/// @param model: Pointer to the GPT2 struct that will be initialized.
/// @param checkpoint_path: Path to the binary file containing the model's checkpoint data.
/// @return void: On successful initialization, returns nothing but can throw errors on failures like
/// invalid model header or version, or file reading issues.
fn  gpt2_build_from_checkpoint(model: *GPT2, checkpoint_path: []const u8) !void{

    var model_header:UIntList = try read_n_parameters_from_file(u32, checkpoint_path, 256, 0);
    var config = GPT2Config{
        .model_header = model_header.items[0],
        .model_version = model_header.items[1],
        .max_seq_len = model_header.items[2],
        .vocab_size = model_header.items[3],
        .num_layers = model_header.items[4],
        .num_heads = model_header.items[5],
        .channels = model_header.items[6],
        };
    if (config.model_header != 20240326) {
        return Errors.InvalidModelHeader;
        }
    if (config.model_version != 1) {
        return Errors.InvalidModelVersion;
        }

    model.config = config;
    printHead(model.*);
    // Define param sizes
    model.params_sizes[0] = config.vocab_size * config.channels;
    model.params_sizes[1] = config.max_seq_len * config.channels;
    model.params_sizes[2] = config.num_layers * config.channels;
    model.params_sizes[3] = config.num_layers * config.channels;
    model.params_sizes[4] = config.num_layers * 3 * config.channels * config.channels;
    model.params_sizes[5] = config.num_layers * 3 * config.channels;
    model.params_sizes[6] = config.num_layers * config.channels * config.channels;
    model.params_sizes[7] = config.num_layers * config.channels;
    model.params_sizes[8] = config.num_layers * config.channels;
    model.params_sizes[9] = config.num_layers * config.channels;
    model.params_sizes[10] = config.num_layers * 4 * config.channels * config.channels;
    model.params_sizes[11] = config.num_layers * 4 * config.channels;
    model.params_sizes[12] = config.num_layers * config.channels * 4 * config.channels;
    model.params_sizes[13] = config.num_layers * config.channels;
    model.params_sizes[14] = config.channels;
    model.params_sizes[15] = config.channels;
    // Calculate total number of parameters
    model.num_parameters = 0;
    for (model.params_sizes) |size| {
        model.num_parameters += size;
        }
    std.debug.print("num_parameters: {}\n", .{model.num_parameters});

    var params_memory: FloatList = try read_n_parameters_from_file(LlmFloat, checkpoint_path, model.num_parameters, 256);
    model.params_memory = params_memory.items;
    var iter: u32 = 0;
    model.params.wte = model.params_memory[iter..iter+model.params_sizes[0]];
    iter += model.params_sizes[0];
    model.params.wpe = model.params_memory[iter..iter+model.params_sizes[1]];
    iter += model.params_sizes[1];
    model.params.ln1w = model.params_memory[iter..iter+model.params_sizes[2]];
    iter += model.params_sizes[2];
    model.params.ln1b = model.params_memory[iter..iter+model.params_sizes[3]];
    iter += model.params_sizes[3];
    model.params.qkvw = model.params_memory[iter..iter+model.params_sizes[4]];
    iter += model.params_sizes[4];
    model.params.qkvb = model.params_memory[iter..iter+model.params_sizes[5]];
    iter += model.params_sizes[5];
    model.params.attprojw = model.params_memory[iter..iter+model.params_sizes[6]];
    iter += model.params_sizes[6];
    model.params.attprojb = model.params_memory[iter..iter+model.params_sizes[7]];
    iter += model.params_sizes[7];
    model.params.ln2w = model.params_memory[iter..iter+model.params_sizes[8]];
    iter += model.params_sizes[8];
    model.params.ln2b = model.params_memory[iter..iter+model.params_sizes[9]];
    iter += model.params_sizes[9];
    model.params.fcw = model.params_memory[iter..iter+model.params_sizes[10]];
    iter += model.params_sizes[10];
    model.params.fcb = model.params_memory[iter..iter+model.params_sizes[11]];
    iter += model.params_sizes[11];
    model.params.fcprojw = model.params_memory[iter..iter+model.params_sizes[12]];
    iter += model.params_sizes[12];
    model.params.fcprojb = model.params_memory[iter..iter+model.params_sizes[13]];
    iter += model.params_sizes[13];
    model.params.lnfw = model.params_memory[iter..iter+model.params_sizes[14]];
    iter += model.params_sizes[14];
    model.params.lnfb = model.params_memory[iter..iter+model.params_sizes[15]];
    iter += model.params_sizes[15];

    // Initialize other fields

    model.acts_memory = undefined;
    model.grads_memory = undefined;
    model.m_memory = undefined;
    model.v_memory = undefined;
    model.grads_acts_memory = undefined;
    model.inputs = undefined;
    model.targets = undefined;
    model.batch_size = 0;
    model.seq_len = 0;
    model.mean_loss = -1.0; // Designates no loss
    }

/// Executes the forward pass of the GPT-2 model using inputs and optionally computes losses using targets.
/// The function dynamically allocates and initializes model states if not already done, applies the model layers,
/// and handles different data tensor sizes efficiently using vectorized operations when possible.
///
/// @param allocator: Memory allocator for dynamic memory management.
/// @param model: Pointer to the GPT2 model structure.
/// @param inputs: Array of input tokens.
/// @param targets: Array of target tokens for training (optional for inference).
/// @param B: Batch size, the number of sequences per batch.
/// @param T: Sequence length, the number of tokens per sequence.
/// @return void: This function returns nothing but throws errors in case of memory allocation failures or mismatched configurations.

fn gpt2_forward(allocator: std.mem.Allocator, model:*GPT2, inputs:[]u32, targets:[]u32, B:u32, T:u32) !void{

    var V = model.config.vocab_size;
    var C = model.config.channels;
    var NH = model.config.num_heads;
    var L = model.config.num_layers;

    if (!model.init_params) {
        model.init_params = true;
        model.batch_size = B;
        model.seq_len = T;
        model.act_sizes[0] = B * T * C;
        model.act_sizes[1] = L * B * T * C;
        model.act_sizes[2] = L * B * T;
        model.act_sizes[3] = L * B * T;
        model.act_sizes[4] = L * B * T * 3 * C;
        model.act_sizes[5] = L * B * T * C;
        model.act_sizes[6] = L * B * NH * T * T;
        model.act_sizes[7] = L * B * NH * T * T;
        model.act_sizes[8] = L * B * T * C;
        model.act_sizes[9] = L * B * T * C;
        model.act_sizes[10] = L * B * T * C;
        model.act_sizes[11] = L * B * T;
        model.act_sizes[12] = L * B * T;
        model.act_sizes[13] = L * B * T * 4 * C;
        model.act_sizes[14] = L * B * T * 4 * C;
        model.act_sizes[15] = L * B * T * C;
        model.act_sizes[16] = L * B * T * C;
        model.act_sizes[17] = B * T * C;
        model.act_sizes[18] = B * T;
        model.act_sizes[19] = B * T;
        model.act_sizes[20] = B * T * V;
        model.act_sizes[21] = B * T * V;
        model.act_sizes[22] = B * T;
        // Lets compute the number of activations. Mostly for debugging reasons.
        var num_activations: u32 = 0;
        for (model.act_sizes) |size| {
            num_activations += size;
            }
        model.num_activations = num_activations;
        model.acts_memory = try allocator.alloc(LlmFloat, num_activations);
        var iter: u32 = 0;
        model.acts.encoded = model.acts_memory[iter..iter+model.act_sizes[0]];
        iter += model.act_sizes[0];
        model.acts.ln1 = model.acts_memory[iter..iter+model.act_sizes[1]];
        iter += model.act_sizes[1];
        model.acts.ln1_mean = model.acts_memory[iter..iter+model.act_sizes[2]];
        iter += model.act_sizes[2];
        model.acts.ln1_rstd = model.acts_memory[iter..iter+model.act_sizes[3]];
        iter += model.act_sizes[3];
        model.acts.qkv = model.acts_memory[iter..iter+model.act_sizes[4]];
        iter += model.act_sizes[4];
        model.acts.atty = model.acts_memory[iter..iter+model.act_sizes[5]];
        iter += model.act_sizes[5];
        model.acts.preatt = model.acts_memory[iter..iter+model.act_sizes[6]];
        iter += model.act_sizes[6];
        model.acts.att = model.acts_memory[iter..iter+model.act_sizes[7]];
        iter += model.act_sizes[7];
        model.acts.attproj = model.acts_memory[iter..iter+model.act_sizes[8]];
        iter += model.act_sizes[8];
        model.acts.residual2 = model.acts_memory[iter..iter+model.act_sizes[9]];
        iter += model.act_sizes[9];
        model.acts.ln2 = model.acts_memory[iter..iter+model.act_sizes[10]];
        iter += model.act_sizes[10];
        model.acts.ln2_mean = model.acts_memory[iter..iter+model.act_sizes[11]];
        iter += model.act_sizes[11];
        model.acts.ln2_rstd = model.acts_memory[iter..iter+model.act_sizes[12]];
        iter += model.act_sizes[12];
        model.acts.fch = model.acts_memory[iter..iter+model.act_sizes[13]];
        iter += model.act_sizes[13];
        model.acts.fch_gelu = model.acts_memory[iter..iter+model.act_sizes[14]];
        iter += model.act_sizes[14];
        model.acts.fcproj = model.acts_memory[iter..iter+model.act_sizes[15]];
        iter += model.act_sizes[15];
        model.acts.residual3 = model.acts_memory[iter..iter+model.act_sizes[16]];
        iter += model.act_sizes[16];
        model.acts.lnf = model.acts_memory[iter..iter+model.act_sizes[17]];
        iter += model.act_sizes[17];
        model.acts.lnf_mean = model.acts_memory[iter..iter+model.act_sizes[18]];
        iter += model.act_sizes[18];
        model.acts.lnf_rstd = model.acts_memory[iter..iter+model.act_sizes[19]];
        iter += model.act_sizes[19];
        model.acts.logits = model.acts_memory[iter..iter+model.act_sizes[20]];
        iter += model.act_sizes[20];
        model.acts.probs = model.acts_memory[iter..iter+model.act_sizes[21]];
        iter += model.act_sizes[21];
        model.acts.losses = model.acts_memory[iter..iter+model.act_sizes[22]];
        iter += model.act_sizes[22];

        model.inputs = try allocator.alloc(u32, B * T);
        model.targets = try allocator.alloc(u32, B * T);

        std.debug.print("num_activations: {}\n", .{model.num_activations});
        }
    else{
        if((B > model.batch_size) or (T != model.seq_len)){
            std.debug.print("Batch size or sequence length mismatch\n", .{});
            return;
            }
        }
    // Reset the activations memory
    @memset(model.acts_memory, 0);
    // Copy inputs and targets
    if(model.inputs.len > inputs.len){
        @memcpy(model.inputs, inputs);
        }
    else{
        model.inputs = try allocator.realloc(model.inputs, inputs.len);
        @memcpy(model.inputs, inputs);
        }
    if(targets.len != 0){
        @memcpy(model.targets, targets);
        }
    if( (C%VectorSize == 0) and (C>VectorSize) ){
        encoder_forward_vec(VectorSize, model.acts.encoded,inputs,model.params.wte,model.params.wpe,B,T,C);
        }
    else{
        encoder_forward(model.acts.encoded,inputs,model.params.wte,model.params.wpe,B,T,C);
        }

    var residual : []LlmFloat = undefined;
    for(0..L) |l| {
        if(l==0){
            residual = model.acts.encoded;
            }
        else{
            residual = model.acts.residual3[(l-1) * B * T * C..];
            }
        var l_ln1w = model.params.ln1w[l * C..];
        var l_ln1b = model.params.ln1b[l * C..];
        var l_qkvw = model.params.qkvw[l * 3 * C * C..];
        var l_qkvb = model.params.qkvb[l * 3 * C..];
        var l_attprojw = model.params.attprojw[l * C * C..];
        var l_attprojb = model.params.attprojb[l * C..];
        var l_ln2w = model.params.ln2w[l * C..];
        var l_ln2b = model.params.ln2b[l * C..];
        var l_fcw = model.params.fcw[l * 4 * C * C..];
        var l_fcb = model.params.fcb[l * 4 * C..];
        var l_fcprojw = model.params.fcprojw[l * C * 4 * C..];
        var l_fcprojb = model.params.fcprojb[l * C..];
        var l_ln1 = model.acts.ln1[l * B * T * C..];
        var l_ln1_mean = model.acts.ln1_mean[l * B * T..];
        var l_ln1_rstd = model.acts.ln1_rstd[l * B * T..];
        var l_qkv = model.acts.qkv[l * B * T * 3 * C..];
        var l_atty = model.acts.atty[l * B * T * C..];
        var l_preatt = model.acts.preatt[l * B * NH * T * T..];
        var l_att = model.acts.att[l * B * NH * T * T..];
        var l_attproj = model.acts.attproj[l * B * T * C..];
        var l_residual2 = model.acts.residual2[l * B * T * C..];
        var l_ln2 = model.acts.ln2[l * B * T * C..];
        var l_ln2_mean = model.acts.ln2_mean[l * B * T..];
        var l_ln2_rstd = model.acts.ln2_rstd[l * B * T..];
        var l_fch = model.acts.fch[l * B * T * 4 * C..];
        var l_fch_gelu = model.acts.fch_gelu[l * B * T * 4 * C..];
        var l_fcproj = model.acts.fcproj[l * B * T * C..];
        var l_residual3 = model.acts.residual3[l * B * T * C..];
        // ToDo implement a generic wat to handle also when C is not a multiple of VectorChannels
        if( (C%VectorSize == 0) and (C>VectorSize) ){
            layernorm_forward_vec(VectorSize,l_ln1, l_ln1_mean, l_ln1_rstd, residual, l_ln1w, l_ln1b, B, T, C);
            matmul_forward_vec(VectorSize,l_qkv, l_ln1, l_qkvw, l_qkvb, B, T, C, 3*C);
            attention_forward_vec(VectorSize,l_atty, l_preatt, l_att, l_qkv, B, T, C, NH);
            matmul_forward_vec(VectorSize,l_attproj, l_atty, l_attprojw, l_attprojb, B, T, C, C);
            residual_forward_vec(VectorSize,l_residual2, residual, l_attproj, B*T*C);
            layernorm_forward_vec(VectorSize,l_ln2,l_ln2_mean, l_ln2_rstd, l_residual2, l_ln2w, l_ln2b, B, T, C);
            matmul_forward_vec(VectorSize,l_fch, l_ln2, l_fcw, l_fcb, B, T, C, 4*C);
            gelu_forward(l_fch_gelu, l_fch, B*T*4*C);
            matmul_forward_vec(VectorSize,l_fcproj, l_fch_gelu, l_fcprojw, l_fcprojb, B, T, 4*C, C);
            residual_forward_vec(VectorSize,l_residual3, l_residual2, l_fcproj, B*T*C);
            }
        else
            {
            layernorm_forward(l_ln1, l_ln1_mean, l_ln1_rstd, residual, l_ln1w, l_ln1b, B, T, C);
            matmul_forward(l_qkv, l_ln1, l_qkvw, l_qkvb, B, T, C, 3*C);
            attention_forward(l_atty, l_preatt, l_att, l_qkv, B, T, C, NH);
            matmul_forward(l_attproj, l_atty, l_attprojw, l_attprojb, B, T, C, C);
            residual_forward(l_residual2, residual, l_attproj, B*T*C);
            layernorm_forward(l_ln2, l_ln2_mean, l_ln2_rstd, l_residual2, l_ln2w, l_ln2b, B, T, C);
            matmul_forward(l_fch, l_ln2, l_fcw, l_fcb, B, T, C, 4*C);
            gelu_forward(l_fch_gelu, l_fch, B*T*4*C);
            matmul_forward(l_fcproj, l_fch_gelu, l_fcprojw, l_fcprojb, B, T, 4*C, C);
            residual_forward(l_residual3, l_residual2, l_fcproj, B*T*C);
            }
        }
    residual = model.acts.residual3[(L-1) * B * T * C..]; // last residual is in residual3
    var biases: []LlmFloat = undefined;
    biases.len = 0;
    if( (C%VectorSize == 0) and (C>VectorSize) ){
        layernorm_forward_vec(VectorSize,model.acts.lnf, model.acts.lnf_mean, model.acts.lnf_rstd,
            residual, model.params.lnfw, model.params.lnfb, B, T, C);
        matmul_forward_vec(VectorSize,model.acts.logits, model.acts.lnf,
            model.params.wte, biases, B, T, C, V);
        }
    else{
        layernorm_forward(model.acts.lnf, model.acts.lnf_mean, model.acts.lnf_rstd,
            residual, model.params.lnfw, model.params.lnfb, B, T, C);
        matmul_forward(model.acts.logits, model.acts.lnf,
            model.params.wte, biases, B, T, C, V);
        }
    softmax_forward(model.acts.probs, model.acts.logits, B, T, V);
    if(targets.len != 0){
        crossentropy_forward(model.acts.losses,
            model.acts.probs,
            targets, B, T, V);
        var mean_loss:LlmFloat = 0.0;
        for(0..B) |b| {
            for(0..T) |t| {
                mean_loss += model.acts.losses[b * T + t];
                }
            }
        var batch_t : LlmFloat = @floatFromInt(B * T);
        mean_loss = mean_loss / batch_t;
        model.mean_loss = mean_loss;
        }
    else{
        // inference mode, no need to compute loss
        model.mean_loss = -1.0;
        }
    }

/// Resets the gradients after the backward pass.
/// This function sets the gradients to zero after the backward pass to prepare the model for the next iteration.
/// @param model: Pointer to the GPT2 model structure.
fn gpt2_zero_grad(model :*GPT2) void{
    if(model.init_grads){
        @memset(model.grads_memory, 0);
        }
    if(model.init_grads_acts){
        @memset(model.grads_acts_memory, 0);
        }
    }
/// Performs the backward propagation for the GPT-2 model starting from the output layer back to the inputs.
/// This function computes gradients for all parameters and activations in the model based on the loss calculated during the forward pass.
///
/// @param allocator: Memory allocator for potential reallocations.
/// @param model: Pointer to the GPT2 model structure.
/// @return void: This function returns nothing but can throw errors if memory allocation fails.

fn gpt2_backward(allocator: std.mem.Allocator, model :*GPT2) !void{

    if(model.mean_loss == -1.0) {
        std.debug.print("Error: must forward with targets before backward\n", .{});
        std.process.exit(1);
        }
    if (!model.init_grads) {
        var num_parameters: u32 = 0;
        for (model.params_sizes) |size| {
            num_parameters += size;
            }
        var num_activations: u32 = 0;
        for (model.act_sizes) |size| {
            num_activations += size;
            }
        var iter:usize = 0;

        model.grads_memory = try allocator.alloc(LlmFloat, num_parameters);

        model.grads.wte = model.grads_memory[iter..iter+model.params_sizes[0]];
        iter += model.params_sizes[0];

        model.grads.wpe = model.grads_memory[iter..iter+model.params_sizes[1]];
        iter += model.params_sizes[1];

        model.grads.ln1w = model.grads_memory[iter..iter+model.params_sizes[2]];
        iter += model.params_sizes[2];

        model.grads.ln1b = model.grads_memory[iter..iter+model.params_sizes[3]];
        iter += model.params_sizes[3];

        model.grads.qkvw = model.grads_memory[iter..iter+model.params_sizes[4]];
        iter += model.params_sizes[4];

        model.grads.qkvb = model.grads_memory[iter..iter+model.params_sizes[5]];
        iter += model.params_sizes[5];

        model.grads.attprojw = model.grads_memory[iter..iter+model.params_sizes[6]];
        iter += model.params_sizes[6];

        model.grads.attprojb = model.grads_memory[iter..iter+model.params_sizes[7]];
        iter += model.params_sizes[7];

        model.grads.ln2w = model.grads_memory[iter..iter+model.params_sizes[8]];
        iter += model.params_sizes[8];

        model.grads.ln2b = model.grads_memory[iter..iter+model.params_sizes[9]];
        iter += model.params_sizes[9];

        model.grads.fcw = model.grads_memory[iter..iter+model.params_sizes[10]];
        iter += model.params_sizes[10];

        model.grads.fcb = model.grads_memory[iter..iter+model.params_sizes[11]];
        iter += model.params_sizes[11];

        model.grads.fcprojw = model.grads_memory[iter..iter+model.params_sizes[12]];
        iter += model.params_sizes[12];

        model.grads.fcprojb = model.grads_memory[iter..iter+model.params_sizes[13]];
        iter += model.params_sizes[13];

        model.grads.lnfw = model.grads_memory[iter..iter+model.params_sizes[14]];
        iter += model.params_sizes[14];

        model.grads.lnfb = model.grads_memory[iter..iter+model.params_sizes[15]];
        iter += model.params_sizes[15];

        iter = 0;
        model.grads_acts_memory = try allocator.alloc(LlmFloat, num_activations);
        model.grads_acts.encoded = model.grads_acts_memory[iter..iter+model.act_sizes[0]];
        iter += model.act_sizes[0];
        model.grads_acts.ln1 = model.grads_acts_memory[iter..iter+model.act_sizes[1]];
        iter += model.act_sizes[1];
        model.grads_acts.ln1_mean = model.grads_acts_memory[iter..iter+model.act_sizes[2]];
        iter += model.act_sizes[2];
        model.grads_acts.ln1_rstd = model.grads_acts_memory[iter..iter+model.act_sizes[3]];
        iter += model.act_sizes[3];
        model.grads_acts.qkv = model.grads_acts_memory[iter..iter+model.act_sizes[4]];
        iter += model.act_sizes[4];
        model.grads_acts.atty = model.grads_acts_memory[iter..iter+model.act_sizes[5]];
        iter += model.act_sizes[5];
        model.grads_acts.preatt = model.grads_acts_memory[iter..iter+model.act_sizes[6]];
        iter += model.act_sizes[6];
        model.grads_acts.att = model.grads_acts_memory[iter..iter+model.act_sizes[7]];
        iter += model.act_sizes[7];
        model.grads_acts.attproj = model.grads_acts_memory[iter..iter+model.act_sizes[8]];
        iter += model.act_sizes[8];
        model.grads_acts.residual2 = model.grads_acts_memory[iter..iter+model.act_sizes[9]];
        iter += model.act_sizes[9];
        model.grads_acts.ln2 = model.grads_acts_memory[iter..iter+model.act_sizes[10]];
        iter += model.act_sizes[10];
        model.grads_acts.ln2_mean = model.grads_acts_memory[iter..iter+model.act_sizes[11]];
        iter += model.act_sizes[11];
        model.grads_acts.ln2_rstd = model.grads_acts_memory[iter..iter+model.act_sizes[12]];
        iter += model.act_sizes[12];
        model.grads_acts.fch = model.grads_acts_memory[iter..iter+model.act_sizes[13]];
        iter += model.act_sizes[13];
        model.grads_acts.fch_gelu = model.grads_acts_memory[iter..iter+model.act_sizes[14]];
        iter += model.act_sizes[14];
        model.grads_acts.fcproj = model.grads_acts_memory[iter..iter+model.act_sizes[15]];
        iter += model.act_sizes[15];
        model.grads_acts.residual3 = model.grads_acts_memory[iter..iter+model.act_sizes[16]];
        iter += model.act_sizes[16];
        model.grads_acts.lnf = model.grads_acts_memory[iter..iter+model.act_sizes[17]];
        iter += model.act_sizes[17];
        model.grads_acts.lnf_mean = model.grads_acts_memory[iter..iter+model.act_sizes[18]];
        iter += model.act_sizes[18];
        model.grads_acts.lnf_rstd = model.grads_acts_memory[iter..iter+model.act_sizes[19]];
        iter += model.act_sizes[19];
        model.grads_acts.logits = model.grads_acts_memory[iter..iter+model.act_sizes[20]];
        iter += model.act_sizes[20];
        model.grads_acts.probs = model.grads_acts_memory[iter..iter+model.act_sizes[21]];
        iter += model.act_sizes[21];
        model.grads_acts.losses = model.grads_acts_memory[iter..iter+model.act_sizes[22]];
        iter += model.act_sizes[22];
        model.init_grads = true;
        model.init_grads_acts = true;
        gpt2_zero_grad(model);
        }
    else{
        gpt2_zero_grad(model);
        }
    // Short cuts
    var B:u32 = model.batch_size;
    var T:u32 = model.seq_len;
    var V:u32 = model.config.vocab_size;
    var C:u32 = model.config.channels;
    var NH:u32 = model.config.num_heads;
    var L:u32 = model.config.num_layers;
    var dbiases: []LlmFloat = undefined;
    dbiases.len = 0;

    var dloss_mean: LlmFloat = 1.0 / @as(LlmFloat, @floatFromInt((B * T)));
    for(0..B) |b| {
        for(0..T) |t| {
            model.grads_acts.losses[b * T + t] = dloss_mean;
            }
        }
    if( (C%VectorSize == 0) and (C>VectorSize) ){
        crossentropy_softmax_backward(
            model.grads_acts.logits,
            model.grads_acts.losses,
            model.acts.probs,
            model.targets,
            B, T, V);

        matmul_backward_vec(VectorSize,model.grads_acts.lnf,
            model.grads.wte, dbiases,
            model.grads_acts.logits,
            model.acts.lnf, model.params.wte, B, T, C, V);
        }
    else{
        crossentropy_softmax_backward(
            model.grads_acts.logits,
            model.grads_acts.losses,
            model.acts.probs,
            model.targets,
            B, T, V);
        matmul_backward(model.grads_acts.lnf,
            model.grads.wte, dbiases,
            model.grads_acts.logits,
            model.acts.lnf, model.params.wte, B, T, C, V);
        }

    var residual = model.acts.residual3[(L-1) * B * T * C..]; // last layer's residual
    var dresidual = model.grads_acts.residual3[(L-1) * B * T * C..]; // write to last layer's residual
    layernorm_backward(dresidual,model.grads.lnfw,model.grads.lnfb,model.grads_acts.lnf,
        residual,model.params.lnfw,model.acts.lnf_mean,model.acts.lnf_rstd,B,T,C);
    var l = L - 1;

    while(l >= 0) {
        if (l==0) {
            residual = model.acts.encoded;
            dresidual = model.grads_acts.encoded;
            }
        else {
            residual = model.acts.residual3[(l-1) * B * T * C..];
            dresidual = model.grads_acts.residual3[(l-1) * B * T * C..];
            }
        var l_ln1w = model.params.ln1w[l * C..];
        var l_qkvw = model.params.qkvw[l * 3 * C * C..];
        var l_attprojw = model.params.attprojw[l * C * C..];
        var l_ln2w = model.params.ln2w[l * C..];
        var l_fcw = model.params.fcw[l * 4 * C * C..];
        var l_fcprojw = model.params.fcprojw[l * C * 4 * C..];
        // get the slices for the gradients of the weights for this layer
        var dl_ln1w = model.grads.ln1w[l * C..];
        var dl_ln1b = model.grads.ln1b[l * C..];
        var dl_qkvw = model.grads.qkvw[l * 3 * C * C..];
        var dl_qkvb = model.grads.qkvb[l * 3 * C..];
        var dl_attprojw = model.grads.attprojw[l * C * C..];
        var dl_attprojb = model.grads.attprojb[l * C..];
        var dl_ln2w = model.grads.ln2w[l * C..];
        var dl_ln2b = model.grads.ln2b[l * C..];
        var dl_fcw = model.grads.fcw[l * 4 * C * C..];
        var dl_fcb = model.grads.fcb[l * 4 * C..];
        var dl_fcprojw = model.grads.fcprojw[l * C * 4 * C..];
        var dl_fcprojb = model.grads.fcprojb[l * C..];
        // get the slices for the activations for this layer
        var l_ln1 = model.acts.ln1[l * B * T * C..];
        var l_ln1_mean = model.acts.ln1_mean[l * B * T..];
        var l_ln1_rstd = model.acts.ln1_rstd[l * B * T..];
        var l_qkv = model.acts.qkv[l * B * T * 3 * C..];
        var l_atty = model.acts.atty[l * B * T * C..];
        var l_att = model.acts.att[l * B * NH * T * T..];
        var l_residual2 = model.acts.residual2[l * B * T * C..];
        var l_ln2 = model.acts.ln2[l * B * T * C..];
        var l_ln2_mean = model.acts.ln2_mean[l * B * T..];
        var l_ln2_rstd = model.acts.ln2_rstd[l * B * T..];
        var l_fch = model.acts.fch[l * B * T * 4 * C..];
        var l_fch_gelu = model.acts.fch_gelu[l * B * T * 4 * C..];
        // get the slices for the gradients of the activations for this layer
        var dl_ln1 = model.grads_acts.ln1[l * B * T * C..];
        var dl_qkv = model.grads_acts.qkv[l * B * T * 3 * C..];
        var dl_atty = model.grads_acts.atty[l * B * T * C..];
        var dl_preatt = model.grads_acts.preatt[l * B * NH * T * T..];
        var dl_att = model.grads_acts.att[l * B * NH * T * T..];
        var dl_attproj = model.grads_acts.attproj[l * B * T * C..];
        var dl_residual2 = model.grads_acts.residual2[l * B * T * C..];
        var dl_ln2 = model.grads_acts.ln2[l * B * T * C..];
        var dl_fch = model.grads_acts.fch[l * B * T * 4 * C..];
        var dl_fch_gelu = model.grads_acts.fch_gelu[l * B * T * 4 * C..];
        var dl_fcproj = model.grads_acts.fcproj[l * B * T * C..];
        var dl_residual3 = model.grads_acts.residual3[l * B * T * C..];
        // backward pass
        if( (C%VectorSize == 0) and (C>VectorSize) ){
            residual_backward(dl_residual2,dl_fcproj,dl_residual3,B*T*C);
            matmul_backward_vec(VectorSize,dl_fch_gelu,dl_fcprojw,dl_fcprojb,dl_fcproj,l_fch_gelu,
                l_fcprojw,B,T,4*C,C);
            gelu_backward(dl_fch,l_fch,dl_fch_gelu,B*T*4*C);
            matmul_backward_vec(VectorSize,dl_ln2,dl_fcw,dl_fcb,dl_fch,l_ln2,l_fcw,B,T,
                C,4*C);
            layernorm_backward(dl_residual2,dl_ln2w,dl_ln2b,dl_ln2,l_residual2,
                l_ln2w,l_ln2_mean,l_ln2_rstd,B, T, C);
            residual_backward(dresidual,dl_attproj,dl_residual2,B*T*C);
            matmul_backward_vec(VectorSize,dl_atty,dl_attprojw,dl_attprojb,dl_attproj,l_atty,
                l_attprojw,B, T, C, C);
            attention_backward_vec(VectorSize,dl_qkv,dl_preatt,dl_att, dl_atty,l_qkv, l_att,
                B, T, C, NH);

            matmul_backward_vec(VectorSize,dl_ln1,dl_qkvw, dl_qkvb, dl_qkv,l_ln1, l_qkvw, B,
                T, C, 3*C);
            layernorm_backward(dresidual,dl_ln1w, dl_ln1b,dl_ln1, residual,
                l_ln1w,l_ln1_mean, l_ln1_rstd, B, T, C);
            }
        else{
            residual_backward(dl_residual2,dl_fcproj,dl_residual3,B*T*C);
            matmul_backward(dl_fch_gelu,dl_fcprojw,dl_fcprojb,dl_fcproj,l_fch_gelu,
                l_fcprojw,B,T,4*C,C);
            gelu_backward(dl_fch,l_fch,dl_fch_gelu,B*T*4*C);
            matmul_backward(dl_ln2,dl_fcw,dl_fcb,dl_fch,l_ln2,l_fcw,B,T,
                C,4*C);
            layernorm_backward(dl_residual2,dl_ln2w,dl_ln2b,dl_ln2,l_residual2,
                l_ln2w,l_ln2_mean,l_ln2_rstd,B, T, C);
            residual_backward(dresidual,dl_attproj,dl_residual2,B*T*C);
            matmul_backward(dl_atty,dl_attprojw,dl_attprojb,dl_attproj,l_atty,
                l_attprojw,B, T, C, C);
            attention_backward(dl_qkv,dl_preatt,dl_att, dl_atty,l_qkv, l_att,
                B, T, C, NH);
            matmul_backward(dl_ln1,dl_qkvw, dl_qkvb, dl_qkv,l_ln1, l_qkvw, B,
                T, C, 3*C);
            layernorm_backward(dresidual,dl_ln1w, dl_ln1b,dl_ln1, residual,
                l_ln1w,l_ln1_mean, l_ln1_rstd, B, T, C);
            }
        if (l > 0) {
            l -= 1;
            }
        else {
            break;
            }
        }
    if( (C%VectorSize == 0) and (C>VectorSize) ){
        encoder_backward_vec(VectorSize,
            model.grads.wte,
            model.grads.wpe,
            model.grads_acts.encoded,
            model.inputs, B, T, C);
        }
    else{
        encoder_backward(
            model.grads.wte,
            model.grads.wpe,
            model.grads_acts.encoded,
            model.inputs, B, T, C);
        }
    }
/// Updates the parameters of the GPT-2 model using the Adam optimizer.
/// This function modifies the model weights based on the gradients computed during the backward pass.
///
/// @param allocator: Memory allocator used for allocating optimizer-specific memories.
/// @param model: Pointer to the GPT2 model structure.
/// @param learning_rate: Learning rate for the optimizer.
/// @param beta1: Exponential decay rate for the first moment estimates in Adam.
/// @param beta2: Exponential decay rate for the second moment estimates in Adam.
/// @param epsilon: Small constant for numerical stability in the Adam optimizer.
/// @param weighted_decay: Weight decay parameter for regularization.
/// @param t: Time step or iteration number, used for bias correction in Adam.
/// @return void: This function returns nothing but can throw errors if memory allocation fails.
fn gpt2_update(allocator: std.mem.Allocator, model :*GPT2, learning_rate:LlmFloat, beta1:LlmFloat, beta2:LlmFloat,
            epsilon:LlmFloat, weigthed_decay:LlmFloat, t:u32) !void{

    if(!model.init_adam){
        model.m_memory = try allocator.alloc(LlmFloat, model.num_parameters);
        model.v_memory = try allocator.alloc(LlmFloat, model.num_parameters);
        @memset(model.m_memory, 0);
        @memset(model.v_memory, 0);
        model.init_adam = true;
        }
    var t_float : LlmFloat = @as(LlmFloat, @floatFromInt(t));

    for(0..model.num_parameters) |i| {
        var param = model.params_memory[i];
        var grad = model.grads_memory[i];
        // update the first moment (momentum)
        var m:LlmFloat = beta1 * model.m_memory[i] + (1.0 - beta1) * grad;
        // update the second moment (RMSprop)
        var v:LlmFloat = beta2 * model.v_memory[i] + (1.0 - beta2) * grad * grad;
        // bias-correct both moments
        var m_hat:LlmFloat = m / (1.0 - cfuncs.powf(beta1, t_float));
        var v_hat:LlmFloat = v / (1.0 - cfuncs.powf(beta2, t_float));

        // update
        model.m_memory[i] = m;
        model.v_memory[i] = v;
        var sqrt_v_hat:LlmFloat = cfuncs.sqrtf(v_hat);
        var tmp = learning_rate * (m_hat / (sqrt_v_hat + epsilon) + weigthed_decay * param);
        model.params_memory[i] -= tmp;
        }
    }
// DataLoader
fn dataloader_init(allocator: std.mem.Allocator,
loader:*DataLoader, filename:
[]const u8, B:u32, T:u32) !void
    {
    loader.B = B;
    loader.T = T;
    loader.tokens_file = std.fs.cwd().openFile(filename, .{ .mode = .read_only }) catch {
        std.debug.print("Error opening tokens file\n", .{});
        return error.UnableToOpenFile;
        };

    const file_size = try loader.tokens_file.getEndPos();
    if (file_size < (B * T + 1) * @sizeOf(i32)) {
        std.debug.print("Error: file size is too small for the batch size and sequence length\n", .{});
        return error.FileTooSmall;
        }

    loader.file_size = file_size;
    loader.current_position = 0;
    loader.batch = try allocator.alloc(u32, B * T + 1);
    loader.inputs = loader.batch;
    loader.targets = loader.batch[1..];
    loader.num_batches = @intCast(file_size / (B * T * @sizeOf(u32)));
    }

fn dataloader_reset( loader:*DataLoader) void{
    loader.current_position = 0;
    }
fn dataloader_next_batch(loader:*DataLoader) !void{
    var B : u32 = loader.B;
    var T : u32 = loader.T;
    if(loader.current_position + (B*T+1) * @sizeOf(i32) >= loader.file_size){
        loader.current_position = 0;
        }
    _ = try loader.tokens_file.seekTo(loader.current_position);
    for (0..B * T + 1) |i| {
        loader.batch[i] = try loader.tokens_file.reader().readVarInt(u32,builtin.cpu.arch.endian(),4);
        }
    loader.current_position += loader.B * loader.T * @sizeOf(i32);
    }
fn dataloader_free( loader:*DataLoader) void{
    defer loader.tokens_file.close();
    //free(loader.batch);
    }

fn sample_mult(probabilities: []LlmFloat, n:u32, coin:LlmFloat) u32{
    var cdf:LlmFloat = 0.0;
    for(0..n) |i| {
        cdf += probabilities[i];
        if(coin < cdf){
            return @as(u32, @intCast(i));
            }
        }
    return n - 1;
    }

pub fn gpt2_free(allocator:std.mem.Allocator, model:*GPT2) void{
    if(model.params_memory.len > 0){
        allocator.free(model.params_memory);
        }
    if(model.acts_memory.len > 0){
        allocator.free(model.acts_memory);
        }
    if(model.grads_memory.len > 0){
        allocator.free(model.grads_memory);
        }
    if(model.grads_acts_memory.len > 0){
        allocator.free(model.grads_acts_memory);
        }
    if(model.m_memory.len > 0){
        allocator.free(model.m_memory);
        }
    if(model.v_memory.len > 0){
        allocator.free(model.v_memory);
        }
    if(model.inputs.len > 0){
        allocator.free(model.inputs);
        }
    if(model.targets.len > 0){
        allocator.free(model.targets);
        }
    }

pub fn main() !void {
    const train_steps = 40000;
    const val_period = 10;
    const GPT2_EOT = 50256;
    const rng_state: u32 = 1337;
    const gen_max_len: u32 = 512;
    const B: u32 = 4;
    const T: u32 = 64;

    var rnd = RndGen.init(rng_state);

    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const stdout = std.io.getStdOut().writer();
    const allocator = arena.allocator();
    try stdout.print("{s} in , {s}!\n", .{program_name, "zig"});
    var model:GPT2 = undefined;
    var gen_tokens : []u32 = try allocator.alloc(u32, gen_max_len);
    model.init_params = false;
    model.init_grads = false;
    model.init_grads_acts = false;
    model.init_adam = false;

    var tokenizer: Tokenizer = undefined;
    try tokenizer_init(allocator,&tokenizer, "data/gpt2_tokenizer.bin");

    gpt2_build_from_checkpoint(&model,
        "data/gpt2_124M.bin",
    ) catch |err| {
        std.debug.print("Error: {}\n", .{err});
        return err;
        };

    printParams(model);
    var train_tokens = "data/tiny_shakespeare_train.bin";
    var val_tokens = "data/tiny_shakespeare_val.bin";
    var train_loader:DataLoader = undefined;
    try dataloader_init(allocator,&train_loader, train_tokens, B, T);
    try stdout.print("Using filename {s} for tokens, {s}!\n", .{train_tokens, "zig"});
    try stdout.print("Train dataset num_batches: {d}\n", .{train_loader.num_batches});
    var val_loader:DataLoader= undefined;
    try dataloader_init(allocator, &val_loader, val_tokens, B, T);
    try stdout.print("Val dataset num_batches: {d}\n", .{val_loader.num_batches});
    const val_num_batches = 10;
    try stdout.print("Val dataset num_batches: {d}\n", .{val_num_batches});

    try stdout.print("Rng_State: {d}\nGen Max Len:{}\n", .{rng_state, gen_max_len});
    for(0..train_steps+1)|step|{
        const print_start = "##########################################################################################";
        try stdout.print("{s}\n", .{print_start});
        const s = "*********************************";
        try stdout.print("{s} Step {} {s}\n", .{s, step, s});
        const startNorm = try Instant.now();
        if(step+1 % val_period == 0){
            var val_loss:LlmFloat = 0.0;
            dataloader_reset(&val_loader);
            for(0..val_num_batches) |_| {
                try dataloader_next_batch(&val_loader);
                try gpt2_forward(allocator, &model, val_loader.inputs, val_loader.targets, B, T);
                val_loss += model.mean_loss;
                }
            val_loss = val_loss / val_num_batches;
            try stdout.print("Step {d}, Val loss: {}\n", .{step, val_loss});
            }
        if((step > 0) and (step % 10 == 0)){
            gen_tokens[0] = GPT2_EOT;
            for(1..gen_max_len)|t|{
                var no_targets:[]u32 = undefined;
                no_targets.len = 0;
                try gpt2_forward(allocator, &model, gen_tokens, no_targets, 1, @as(u32, @intCast(t)));
                var probs = model.acts.probs[(t-1)*model.config.vocab_size..];
                var coin = rnd.random().float(LlmFloat);
                var next_token = sample_mult(probs, model.config.vocab_size, coin);
                gen_tokens[t] = next_token;
                if (tokenizer.init_ok){
                    var token_str = tokenizer_decode(tokenizer, next_token);
                    try stdout.print("{s}", .{token_str});
                    }
                }
            try stdout.print("Generated Tokens ", .{});
            for(0..gen_max_len)|t|{
                if(gen_tokens[t] == GPT2_EOT){
                    break;
                    }
                var token = gen_tokens[t];
                try stdout.print("{d} ", .{token});
                }
            }
        try dataloader_next_batch(&train_loader);
        try gpt2_forward(allocator, &model, train_loader.inputs, train_loader.targets, B, T);
        try gpt2_backward(allocator, &model);
        try gpt2_update(allocator, &model, 0.0001, 0.9, 0.999, 1e-8, 0.0, @as(u32, @intCast(step+1)));

        const endNorm = try Instant.now();
        const elapsedNorm: f64 = @floatFromInt(endNorm.since(startNorm));
        try stdout.print("Step {} train loss {} took {} ms\n", .{step, model.mean_loss, elapsedNorm/time.ns_per_ms});
        }
    dataloader_free(&train_loader);
    dataloader_free(&val_loader);
    gpt2_free(allocator, &model);
    }

