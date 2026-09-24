// The frame counter: the last thing the shim records on the game's command
// list each evaluate.
//
// D3D12 command lists cannot record fence Signal/Wait (queue-only), and the
// shim never sees the game's queue -- so there is no direct way to order HIP
// work against the game's submission. The counter is the workaround: the next
// evaluate reads it back through the shim's own queue, and seeing the value
// N-1 proves every shim op recorded on list N-1 (encode, staging copy,
// resolve, dumps -- the write is last) actually executed. Only then is it
// safe to run HIP on the staged input and to rewrite the model texture.
//
// A stale read (game recording ahead without submitting) is detected, not
// guessed at: the value is simply lower than expected and the HIP work is
// skipped for that evaluate. No polling -- blocking here could deadlock a
// game that submits from the same thread it records on.

RWByteAddressBuffer g_counter : register(u0);

cbuffer Params : register(b0) {
    uint g_frame;
    uint g_pad0;
    uint g_pad1;
    uint g_pad2;
};

[numthreads(1, 1, 1)]
void main(uint3 dtid : SV_DispatchThreadID) {
    g_counter.Store(0, g_frame);
}
