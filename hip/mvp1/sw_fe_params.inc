// The pre-block parameter block the real entry kernel reads.
//
// HANDOFF §13.4, with every offset used here PINNED BY THE LOAD ITSELF in
// cubin_00's PTX rather than inferred from the summary:
//
//   +136  f32   axis A: b      )  the sampling triples, read as
//   +140  f32   axis B: b      )  ld.param.v2.b32 pairs -- note the
//   +144  f32   axis A: a      )  8-byte stride, so a triple is NOT three
//   +148  f32   axis B: a      )  consecutive floats. Reading it as one
//   +152  f32   axis A: m      )  would silently grab the neighbouring lane.
//   +156  f32   axis B: m      )
//   +180  f32   -> f16, the `hi` slot the row assembly packs (the port calls
//               this value hi176; §13.3's label is one field off)
//   +192  u64   lo: flag, hi: input scale -- the kernel multiplies by 2*hi
//   +200  b32   the PCG seed, hashed with 0x9E3779B9
//   +208  v2.b32 (W, H)
//   +240  v2.b32 token-grid dims, divided by 4 in-kernel (patchify factor 4)
//
// Only the fields this port consumes are named. The rest stays opaque so the
// layout stays honest: a field we have not pinned does not get a name that
// implies we have.
//
// NOT struct fields, though the port once took them as parameters: the row
// assembly's 1.0 and 0.5 are immediates in the kernel (mov.b32 %r65/%r66), and
// the 2*hi scale is computed, not stored. Our s_sel/s_a/s_b stay port-level.
struct SwFeParams {
    unsigned long long tex[5];        //    0  colour (mandatory) + 4 optional
    unsigned char  opaque0[72];       //   40  dims / flags / rects
    unsigned char  affine_lane4[24];  //  112  the +4 lanes of the four triples
    float  ay_b, ax_b;                //  136
    float  ay_a, ax_a;                //  144
    float  ay_m, ax_m;                //  152
    unsigned char  opaque1[16];       //  160
    float  hi180;                     //  176
    unsigned char  opaque2[4];        //  180
    unsigned long long exposure;      //  184
    unsigned long long inscale;       //  192
    int    seed;                      //  200
    unsigned char  opaque3[4];        //  204
    int    W, H;                      //  208
    unsigned long long outptr;        //  216
    unsigned long long auxptr;        //  224
    unsigned char  opaque4[8];        //  232
    int    gridW, gridH;              //  240
    unsigned char  opaque5[16];       //  248
};

static_assert(sizeof(SwFeParams) == 264, "SwFeParams must be 264 bytes");

// Identity affine (a=1, b=0, m=1) -- the direct sample this port used before
// the map was ported.
#define SW_FE_IDENTITY_A 1.0f
#define SW_FE_IDENTITY_B 0.0f
#define SW_FE_IDENTITY_M 1.0f
