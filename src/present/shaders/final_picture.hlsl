// Final-picture front end (S343): the two passes around the engine.
//   PsDown  the presented frame (display-encoded) reduced to the model's
//           working size, RGBA16F in [0, 1].
//   PsUp    the frame plus the engine's change map (transfer=nvidia: the
//           change of the encoded frame it was shown, in RGB), enlarged
//           bilinearly, written back into the back buffer.
// Compiled three times (ENTRY = VsMain / PsDown / PsUp).

cbuffer Constants : register(b0)
{
    float2 gSrcSize;   // the frame copy, in texels
    float2 gDstSize;   // the pass's target, in pixels
};

Texture2D<float4> gFrame : register(t0);
Texture2D<float4> gDelta : register(t1);
SamplerState gLinear : register(s0);

struct Vertex
{
    float4 pos : SV_Position;
    float2 uv : TEXCOORD0;
};

// One triangle over the target.
Vertex VsMain(uint id : SV_VertexID)
{
    Vertex o;
    o.uv = float2((id << 1) & 2, id & 2);
    o.pos = float4(o.uv * float2(2.0, -2.0) + float2(-1.0, 1.0), 0.0, 1.0);
    return o;
}

float4 PsDown(Vertex i) : SV_Target
{
    const float2 foot = gSrcSize / gDstSize;  // source texels a target pixel
    if (foot.x <= 1.001 && foot.y <= 1.001)
        return float4(saturate(gFrame.Load(int3(i.pos.xy, 0)).rgb), 1.0);
    // A box over the footprint from four bilinear taps (exact at 2x).
    const float2 o = 0.25 * foot / gSrcSize;
    float3 c = gFrame.SampleLevel(gLinear, i.uv + float2(-o.x, -o.y), 0).rgb;
    c += gFrame.SampleLevel(gLinear, i.uv + float2(o.x, -o.y), 0).rgb;
    c += gFrame.SampleLevel(gLinear, i.uv + float2(-o.x, o.y), 0).rgb;
    c += gFrame.SampleLevel(gLinear, i.uv + float2(o.x, o.y), 0).rgb;
    return float4(saturate(c * 0.25), 1.0);
}

float4 PsUp(Vertex i) : SV_Target
{
    const float4 src = gFrame.Load(int3(i.pos.xy, 0));
    float3 d = gDelta.SampleLevel(gLinear, i.uv, 0).rgb;
    if (any(isnan(d)) || any(isinf(d))) d = float3(0.0, 0.0, 0.0);
    return float4(saturate(src.rgb + d), src.a);
}
