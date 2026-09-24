// Scaled copy: the identity upscaler.
//
// This exists so the shim is installable before the neural core does: with a
// plain resample the frame is correct, just soft. The RenoDX encode/resolve
// passes will reuse this root signature and descriptor scaffolding.
//
// b0 constants:
//   srcRect  = x, y, w, h   source subrect, in texels
//   dstRect  = x, y, w, h   destination subrect, in texels
//   sizes    = srcW, srcH, dstW, dstH   full texture dimensions
//   flags    = x: 1 -> 1:1 copy (bypass filtering), else linear

cbuffer CB : register(b0)
{
    uint4 SrcRect;
    uint4 DstRect;
    uint4 Sizes;
    uint4 Flags;
};

Texture2D<float4> g_src : register(t0);
SamplerState g_samp : register(s0);
RWTexture2D<float4> g_dst : register(u0);

[numthreads(8, 8, 1)]
void main(uint3 tid : SV_DispatchThreadID)
{
    uint2 d = tid.xy;
    if (d.x >= DstRect.z || d.y >= DstRect.w)
        return;

    uint2 dstPos = DstRect.xy + d;

    // Fraction across the destination subrect, at pixel centres.
    float2 f = (float2(d) + 0.5f) / (float2)DstRect.zw;

    // Matching position in source texels, then normalised for Sample().
    float2 srcPx = (float2)SrcRect.xy + f * (float2)SrcRect.zw;
    float2 uv = srcPx / (float2)Sizes.xy;

    float4 c;
    if (Flags.x == 1u)
        c = g_src.Load(int3((int)srcPx.x, (int)srcPx.y, 0));
    else
        c = g_src.SampleLevel(g_samp, uv, 0.0f);

    g_dst[dstPos] = c;
}
