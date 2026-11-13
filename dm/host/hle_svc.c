
// Horizon kernel system calls (triggered by the SDK runtime once the SDK is loaded).
// Only what the P1 boot chain needs is implemented: memory layout + heap + query +
// debug output + timing.
//
// Register convention: inputs w1.. / outputs w0=Result, x1=value
// (GetInfo / SetHeapSize), consistent with the public Horizon ABI.

#include "dm.h"

#include <sys/mman.h>
#include <sys/types.h>
#include <errno.h>
#include <unistd.h>
#include <sched.h>
#include <time.h>
#include <stdatomic.h>
#include <mach/mach.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

// Reserve one whole block of guest address space, partitioned the Horizon way.
// It only takes address space, not physical memory.
#define RESERVE DM_GAME_MEM_RESERVE

static uint8_t *g_as;
static size_t g_reserve;

static uint8_t *g_heap;
static _Atomic size_t g_heap_cur;
static size_t g_heap_cap;

int dm_guest_heap_readable(uint64_t a, size_t n) {
    uint64_t b = (uint64_t)(uintptr_t)g_heap;
    size_t cur = g_heap_cur;
    return g_heap && a >= b && n <= cur && a - b <= cur - n;
}

static uint8_t *g_alias;
static size_t g_alias_cap;
static _Atomic uint64_t *g_alias_bits;
static uint64_t *g_alias_exposed;

static uint8_t *g_stack;
static size_t g_stack_cap;

int dm_guest_as_contains(uintptr_t a) {
    uintptr_t b = (uintptr_t)g_as;
    return g_as && a >= b && a - b < g_reserve;
}

static void *alias_windows_thread(void *arg);
static void *regions_thread(void *arg);

static void mem_init_once(void) {
    size_t pg = (size_t)getpagesize();

    if (RESERVE < 0x10000000 ||
        DM_GAME_MEM_ALIAS > RESERVE - 0x10000000 ||
        DM_GAME_MEM_STACK > RESERVE - 0x10000000 - DM_GAME_MEM_ALIAS ||
        ((RESERVE | DM_GAME_MEM_ALIAS | DM_GAME_MEM_STACK) & (pg - 1))) {
        die("guest address-space profile has invalid native boundaries");
    }

    // DM_ALIAS_SIZE=<bytes>: override alias region size for diagnostics.
    size_t alias = DM_GAME_MEM_ALIAS;
    const char *as = getenv("DM_ALIAS_SIZE");

    if (as && *as) {
        alias = (size_t)strtoull(as, NULL, 0);

        if (alias < 0x10000000 || (alias & (pg - 1))) {
            die("DM_ALIAS_SIZE %s: at least 256 MB, in whole host pages", as);
        }
    }

    // Keep the profile's heap and stack sizes unchanged.
    // Only the alias region can be overridden.
    g_reserve = RESERVE - DM_GAME_MEM_ALIAS + alias;

    LOG(
        "[mem] reserve request: reserve=%#llx (%llu bytes), "
        "profile_reserve=%#llx, profile_alias=%#llx, "
        "runtime_alias=%#llx, stack=%#llx, page=%#zx",
        (unsigned long long)g_reserve,
        (unsigned long long)g_reserve,
        (unsigned long long)RESERVE,
        (unsigned long long)DM_GAME_MEM_ALIAS,
        (unsigned long long)alias,
        (unsigned long long)DM_GAME_MEM_STACK,
        pg
    );

    errno = 0;

    g_as = mmap(
        NULL,
        g_reserve,
        PROT_NONE,
        MAP_PRIVATE | MAP_ANON,
        -1,
        0
    );

    if (g_as == MAP_FAILED) {
        int mmap_errno = errno;

        LOG(
            "[mem] mmap guest AS FAILED: reserve=%#llx (%llu bytes), "
            "errno=%d, profile_reserve=%#llx, profile_alias=%#llx, "
            "runtime_alias=%#llx, stack=%#llx, page=%#zx",
            (unsigned long long)g_reserve,
            (unsigned long long)g_reserve,
            mmap_errno,
            (unsigned long long)RESERVE,
            (unsigned long long)DM_GAME_MEM_ALIAS,
            (unsigned long long)alias,
            (unsigned long long)DM_GAME_MEM_STACK,
            pg
        );

        LOG("[mem] mmap errno=%d", mmap_errno);
        die("reserve guest AS failed");
    }

    // Fixed partition:
    // alias region -> stack region -> heap region.
    g_alias = g_as + 0x10000000;
    g_alias_cap = alias;

    g_stack = g_alias + g_alias_cap;
    g_stack_cap = DM_GAME_MEM_STACK;

    g_heap = g_stack + g_stack_cap;
    g_heap_cap = g_reserve - (size_t)(g_heap - g_as);

    LOG(
        "[mem] partition: base=%p reserve=%#llx "
        "alias=%p+%#zx stack=%p+%#zx heap=%p+%#zx",
        g_as,
        (unsigned long long)g_reserve,
        g_alias,
        g_alias_cap,
        g_stack,
        g_stack_cap,
        g_heap,
        g_heap_cap
    );

    if (!(g_alias_bits = calloc(
        (g_alias_cap >> 18) + 1,
        sizeof *g_alias_bits
    ))) {
        die("alias region page bitmap calloc");
    }

    if (!(g_alias_exposed = calloc(
        g_alias_cap / pg / 64 + 1,
        sizeof *g_alias_exposed
    ))) {
        die("alias exposure bitmap calloc");
    }

    LOG(
        "guest AS %p +%#llx heap %p alias %p stack %p",
        g_as,
        (unsigned long long)g_reserve,
        g_heap,
        g_alias,
        g_stack
    );

    const char *aw = getenv("DM_ALIAS_WINDOWS");

    if (aw && atoi(aw) > 0) {
        pthread_t th;

        if (!pthread_create(
            &th,
            NULL,
            alias_windows_thread,
            (void *)(uintptr_t)atoi(aw)
        )) {
            pthread_detach(th);
        }
    }

    const char *rf = getenv("DM_REGIONS");

    if (rf && *rf) {
        pthread_t th;

        if (!pthread_create(
            &th,
            NULL,
            regions_thread,
            (void *)rf
        )) {
            pthread_detach(th);
        }
    }
}

static void mem_init(void) {
    static pthread_once_t once = PTHREAD_ONCE_INIT;
    pthread_once(&once, mem_init_once);
}
