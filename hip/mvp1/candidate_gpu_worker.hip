// Persistent dispatcher for the existing exact candidate diagnostic tests.
// Test bodies, kernels, allocations and comparisons are reused verbatim.
#include "head70_test_common.h"
#include <cstdint>
#include <cstring>
#include <utility>
#ifdef _WIN32
#include <io.h>
#include <fcntl.h>
#endif

#define main job_spatial64_test
#include "spatial64_test.hip"
#undef main
#define main job_c64_ffn_candidate_test
#include "c64_ffn_candidate_test.hip"
#undef main
#define main job_c64_attention_candidate_test
#include "c64_attention_candidate_test.hip"
#undef main
#define main job_spatial64_output_test
#include "spatial64_output_test.hip"
#undef main
#define main job_encoder64_downsample_test
#include "encoder64_downsample_test.hip"
#undef main
#define main job_spatial128_test
#include "spatial128_test.hip"
#undef main
#define main job_c128_ffn_candidate_test
#include "c128_ffn_candidate_test.hip"
#undef main
#define main job_c128_attention_candidate_test
#include "c128_attention_candidate_test.hip"
#undef main
#define main job_spatial128_output_test
#include "spatial128_output_test.hip"
#undef main
#define main job_encoder128_downsample_test
#include "encoder128_downsample_test.hip"
#undef main
#define main job_spatial256_window_test
#include "spatial256_window_test.hip"
#undef main
#define main job_c256_ffn_candidate_test
#include "c256_ffn_candidate_test.hip"
#undef main
#define main job_c256_attention_candidate_test
#include "c256_attention_candidate_test.hip"
#undef main
#define main job_spatial256_output_test
#include "spatial256_output_test.hip"
#undef main
#define main job_encoder256_downsample_test
#include "encoder256_downsample_test.hip"
#undef main

#define main job_vit_bridge_ptx_test
#include "vit_bridge_ptx_test.hip"
#undef main
#define main job_vit_expand_chain_test
#include "vit_expand_chain_test.hip"
#undef main
#define main job_decoder39_entry_test
#include "decoder39_entry_test.hip"
#undef main
#define main job_spatial32_peer_test
#include "spatial32_peer_test.hip"
#undef main
#define main job_c32_peer_body_test
#include "c32_peer_body_test.hip"
#undef main
#define main job_spatial32_peer_output_test
#include "spatial32_peer_output_test.hip"
#undef main
#define main job_upsample66_prefix_test
#include "upsample66_prefix_test.hip"
#undef main
#define main job_head70_test
#include "head70_test.hip"
#undef main

#define main job_split512_block_test
#include "split512_block_test.hip"
#undef main
#define main job_split512_bridge_test
#include "split512_bridge_test.hip"
#undef main

struct Job { const char* name; int (*run)(int, char**); };
static const Job jobs[] = {
    {"split512_block_test", job_split512_block_test},
    {"split512_bridge_test", job_split512_bridge_test},
    {"vit_bridge_ptx_test", job_vit_bridge_ptx_test},
    {"vit_expand_chain_test", job_vit_expand_chain_test},
    {"decoder39_entry_test", job_decoder39_entry_test},
    {"spatial32_peer_test", job_spatial32_peer_test},
    {"c32_peer_body_test", job_c32_peer_body_test},
    {"spatial32_peer_output_test", job_spatial32_peer_output_test},
    {"upsample66_prefix_test", job_upsample66_prefix_test},
    {"head70_test", job_head70_test},
    {"spatial64_test", job_spatial64_test},
    {"c64_ffn_candidate_test", job_c64_ffn_candidate_test},
    {"c64_attention_candidate_test", job_c64_attention_candidate_test},
    {"spatial64_output_test", job_spatial64_output_test},
    {"encoder64_downsample_test", job_encoder64_downsample_test},
    {"spatial128_test", job_spatial128_test},
    {"c128_ffn_candidate_test", job_c128_ffn_candidate_test},
    {"c128_attention_candidate_test", job_c128_attention_candidate_test},
    {"spatial128_output_test", job_spatial128_output_test},
    {"encoder128_downsample_test", job_encoder128_downsample_test},
    {"spatial256_window_test", job_spatial256_window_test},
    {"c256_ffn_candidate_test", job_c256_ffn_candidate_test},
    {"c256_attention_candidate_test", job_c256_attention_candidate_test},
    {"spatial256_output_test", job_spatial256_output_test},
    {"encoder256_downsample_test", job_encoder256_downsample_test},
};

static int dispatch(int argc, char** argv) {
    if (argc < 1) return 2;
    for (const auto& job : jobs)
        if (!strcmp(argv[0], job.name)) return job.run(argc, argv);
    fprintf(stderr, "unknown GPU test: %s\n", argv[0]);
    return 2;
}

static bool read_exact(void* data, size_t bytes) {
    return fread(data, 1, bytes, stdin) == bytes;
}

int main(int argc, char** argv) {
    if (argc == 2 && !strcmp(argv[1], "--list")) {
        for (const auto& job : jobs) puts(job.name);
        return 0;
    }
    if (argc != 2 || strcmp(argv[1], "--worker"))
        return dispatch(argc - 1, argv + 1);
#ifdef _WIN32
    if (_setmode(_fileno(stdin), _O_BINARY) == -1) return 2;
#endif
    puts("D5GPUWORKER 1");
    fflush(stdout);
    for (;;) {
        // Little-endian uint32 job id, argument count, then length-prefixed
        // argv bytes. Windows argv bytes use the same ANSI codepage as CRT
        // paths in the standalone tests. EOF between jobs closes cleanly.
        uint32_t id = 0, count = 0;
        size_t got = fread(&id, 1, sizeof(id), stdin);
        if (got == 0 && feof(stdin)) return 0;
        if (got != sizeof(id) || !read_exact(&count, sizeof(count)) ||
                !id || !count || count > 8) {
            fputs("invalid GPU worker request header\n", stderr);
            return 2;
        }
        std::vector<std::string> args;
        for (uint32_t i = 0; i < count; ++i) {
            uint32_t length = 0;
            if (!read_exact(&length, sizeof(length)) || length > 32768) {
                fputs("invalid GPU worker argument size\n", stderr);
                return 2;
            }
            std::string value(length, '\0');
            if (!read_exact(value.data(), length) || value.find('\0') != std::string::npos) {
                fputs("invalid GPU worker argument\n", stderr);
                return 2;
            }
            args.push_back(std::move(value));
        }
        std::vector<char*> pointers;
        for (auto& value : args) pointers.push_back(value.data());
        pointers.push_back(nullptr);
        int result = dispatch(int(count), pointers.data());
        printf("D5GPU_DONE %u %d\n", id, result);
        fflush(stdout);
        fflush(stderr);
        // A failed test may have returned before freeing allocations. Retire
        // the entire context, and never continue after a failed comparison.
        if (result) return result;
    }
}
