// Forced include for a throwaway engine build: log every file opened for
// reading to $FOPEN_LOG (one path per line).
#pragma once
#include <cstdio>
#include <cstdlib>
static inline FILE* logged_fopen(const char* path, const char* mode) {
    static FILE* log = [] {
        const char* l = getenv("FOPEN_LOG");
        return l ? ::fopen(l, "a") : (FILE*)nullptr;
    }();
    if (log && mode && mode[0] == 'r') { fprintf(log, "%s\n", path); fflush(log); }
    return ::fopen(path, mode);
}
#define fopen logged_fopen
