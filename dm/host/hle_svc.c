// Horizon kernel system calls (triggered by the SDK runtime once the SDK is loaded).
// Only what the P1 boot chain needs is implemented: memory layout + heap + query + debug output + timing.
// Register convention: inputs w1.. / outputs w0=Result, x1=value (GetInfo / SetHeapSize), consistent with the public Horizon ABI.

#include "dm.h"

#include <sys/mman.h>
#include <errno.h>
#include <unistd.h>
#include <sched.h>
#include <time.h>
#include <stdatomic.h>

#include <mach/mach.h>
#include <mach/mach_vm.h>

#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>

// Reserve one whole block of guest address space, partitioned the Horizon way (address values are the real addresses returned by the host's mmap).
// It only takes address space, not memory.
// Size 5 GB: the game needs the first 1 GB (alias region, stacks) + one whole heap block (application memory pool of 3285 MiB, see APP_POOL below),
// about 4.2 GB in total; on iOS without the "extended virtual addressing" entitlement a single reservation can be at most 6 GB
// (read on a real device: 6 GB succeeds, 7 GB fails), so it cannot be made any wider --
// with an 8 GB reservation, builds from developer accounts that cannot sign that entitlement (free accounts) would exit at this step.
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

    if (pg == 0 || (pg & (pg - 1)) != 0)
        die("invalid host page size: %#zx", pg);

    if (RESERVE < 0x10000000 ||
        DM_GAME_MEM_ALIAS > RESERVE - 0x10000000 ||
        DM_GAME_MEM_STACK > RESERVE - 0x10000000 - DM_GAME_MEM_ALIAS ||
        ((RESERVE | DM_GAME_MEM_ALIAS | DM_GAME_MEM_STACK) & (pg - 1))) {
        die("guest address-space profile has invalid native boundaries");
    }

    // DM_ALIAS_SIZE=<bytes>: override the alias region size.
    // Validate the input before using it in size calculations.
    size_t alias = (size_t)DM_GAME_MEM_ALIAS;
    const char *as = getenv("DM_ALIAS_SIZE");

    if (as && *as) {
        char *end = NULL;
        unsigned long long parsed;

        if (*as == '-') {
            die("DM_ALIAS_SIZE must not be negative: %s", as);
        }

        errno = 0;
        parsed = strtoull(as, &end, 0);

        if (errno == ERANGE || end == as || !end || *end != '\0' ||
            parsed > (unsigned long long)SIZE_MAX) {
            die("invalid DM_ALIAS_SIZE value: %s", as);
        }

        alias = (size_t)parsed;

        if (alias < 0x10000000 || (alias & (pg - 1))) {
            die("DM_ALIAS_SIZE %s: at least 256 MB, in whole host pages", as);
        }
    }

    // Calculate the reservation size without allowing size_t overflow.
    size_t reserve_base = (size_t)RESERVE - (size_t)DM_GAME_MEM_ALIAS;

    if (alias > SIZE_MAX - reserve_base) {
        die("DM_ALIAS_SIZE overflows guest address-space reservation: alias=%#zx base=%#zx",
            alias, reserve_base);
    }

    g_reserve = reserve_base + alias;

    LOG("[mem] reserve request: reserve=%#llx (%llu bytes), profile_reserve=%#llx, profile_alias=%#llx, runtime_alias=%#llx, stack=%#llx, page=%#zx",
        (unsigned long long)g_reserve,
        (unsigned long long)g_reserve,
        (unsigned long long)RESERVE,
        (unsigned long long)DM_GAME_MEM_ALIAS,
        (unsigned long long)alias,
        (unsigned long long)DM_GAME_MEM_STACK,
        pg);

    // First try the normal mmap reservation.
    g_as = mmap(NULL, g_reserve, PROT_NONE, MAP_PRIVATE | MAP_ANON, -1, 0);

    if (g_as == MAP_FAILED) {
        int mmap_errno = errno;
        kern_return_t mach_status = KERN_FAILURE;
        mach_vm_address_t mach_addr = 0;
        int mach_allocated = 0;

        LOG("[mem] mmap guest AS FAILED: reserve=%#llx (%llu bytes), errno=%d, profile_reserve=%#llx, profile_alias=%#llx, runtime_alias=%#llx, stack=%#llx, page=%#zx",
            (unsigned long long)g_reserve,
            (unsigned long long)g_reserve,
            mmap_errno,
            (unsigned long long)RESERVE,
            (unsigned long long)DM_GAME_MEM_ALIAS,
            (unsigned long long)alias,
            (unsigned long long)DM_GAME_MEM_STACK,
            pg);

        // Fallback: ask Mach VM for an address-space reservation.
        // This reserves virtual address space only; it does not allocate
        // or commit the full reservation as physical memory.
        mach_status = mach_vm_allocate(
            mach_task_self(),
            &mach_addr,
            (mach_vm_size_t)g_reserve,
            VM_FLAGS_ANYWHERE
        );

        if (mach_status == KERN_SUCCESS) {
            mach_allocated = 1;

            mach_status = mach_vm_protect(
                mach_task_self(),
                mach_addr,
                (mach_vm_size_t)g_reserve,
                FALSE,
                VM_PROT_NONE
            );

            if (mach_status == KERN_SUCCESS) {
                g_as = (uint8_t *)(uintptr_t)mach_addr;

                LOG("[mem] Mach VM guest AS reservation succeeded: base=%p reserve=%#llx mmap_errno=%d",
                    g_as,
                    (unsigned long long)g_reserve,
                    mmap_errno);
            } else {
                kern_return_t dealloc_status = mach_vm_deallocate(
                    mach_task_self(),
                    mach_addr,
                    (mach_vm_size_t)g_reserve
                );

                LOG("[mem] Mach VM protect failed: mach_status=%d deallocate_status=%d reserve=%#llx",
                    mach_status,
                    dealloc_status,
                    (unsigned long long)g_reserve);
            }
        }

        if (g_as == MAP_FAILED) {
            LOG("[mem] guest AS reservation failed: mmap_errno=%d mach_status=%d mach_allocated=%d reserve=%#llx profile_reserve=%#llx profile_alias=%#llx runtime_alias=%#llx stack=%#llx page=%#zx",
                mmap_errno,
                mach_status,
                mach_allocated,
                (unsigned long long)g_reserve,
                (unsigned long long)RESERVE,
                (unsigned long long)DM_GAME_MEM_ALIAS,
                (unsigned long long)alias,
                (unsigned long long)DM_GAME_MEM_STACK,
                pg);

            die("reserve guest AS failed: mmap_errno=%d mach_status=%d",
                mmap_errno, mach_status);
        }
    }

    // Fixed partition (all inside the reserved block).
    g_alias = g_as + 0x10000000;
    g_alias_cap = alias;
    g_stack = g_alias + g_alias_cap;
    g_stack_cap = DM_GAME_MEM_STACK;
    g_heap = g_stack + g_stack_cap;
    g_heap_cap = g_reserve - (size_t)(g_heap - g_as);

    if (!(g_alias_bits = calloc((g_alias_cap >> 18) + 1, sizeof *g_alias_bits)))
        die("alias region page bitmap calloc");

    if (!(g_alias_exposed = calloc(g_alias_cap / pg / 64 + 1, sizeof *g_alias_exposed)))
        die("alias exposure bitmap calloc");

    LOG("guest AS %p +%#llx heap %p alias %p stack %p",
        g_as, (unsigned long long)g_reserve, g_heap, g_alias, g_stack);

    const char *aw = getenv("DM_ALIAS_WINDOWS");
    if (aw && atoi(aw) > 0) {
        pthread_t th;
        if (!pthread_create(&th, NULL, alias_windows_thread, (void *)(uintptr_t)atoi(aw)))
            pthread_detach(th);
    }

    const char *rf = getenv("DM_REGIONS");
    if (rf && *rf) {
        pthread_t th;
        if (!pthread_create(&th, NULL, regions_thread, (void *)rf))
            pthread_detach(th);
    }
}

static void mem_init(void) {
    static pthread_once_t once = PTHREAD_ONCE_INIT;
    pthread_once(&once, mem_init_once);
}

// Application memory pool = the 3285 MB the Switch (4 GB model) gives applications.
#define APP_POOL DM_GAME_APP_POOL
static size_t image_bytes(void) {
    size_t n = 0;
    for (int i = 0; i < g_nmod; i++)
        if (!g_mod[i].nro)
            n += g_mod[i].size;
    return (n + 0x1fffff) & ~(size_t)0x1fffff;
}
static size_t big_bss_bytes(void) {
    size_t n = 0;
    for (int i = 0; i < g_nmod; i++)
        if (!g_mod[i].nro && g_mod[i].patch_off < 0)
            n += g_mod[i].size - g_mod[i].file_image_len;
    return n;
}

void svc_guest_range(uint8_t **base, size_t *size) {
    *base = g_as;
    *size = g_reserve;
}

enum {
    I_CoreMask=0, I_PriorityMask=1, I_AliasAddr=2, I_AliasSize=3,
    I_HeapAddr=4, I_HeapSize=5, I_TotalMem=6, I_UsedMem=7,
    I_DebuggerAttached=8, I_ResourceLimit=9, I_IdleTick=10,
    I_RandomEntropy=11, I_AslrAddr=12, I_AslrSize=13,
    I_StackAddr=14, I_StackSize=15, I_SysResTotal=16,
    I_SysResUsed=17, I_ProgramId=18, I_InitProcIdRange=19,
    I_UserExcAddr=20, I_TotalNonSys=21, I_UsedNonSys=22,
    I_IsApplication=23, I_FreeThreadCount=24
};

static uint8_t g_seen[256];

#define MAX_MAP 4096
static struct {
    uint64_t base, size, src;
    uint32_t state;
} g_map[MAX_MAP];

static int g_nmap;
static pthread_mutex_t g_map_lock = PTHREAD_MUTEX_INITIALIZER;

static int map_find(uint64_t a, uint64_t n) {
    for (int i = 0; i < g_nmap; i++)
        if (a < g_map[i].base + g_map[i].size &&
            g_map[i].base < a + n)
            return i + 1;
    return 0;
}

static void map_add(uint64_t b, uint64_t n, uint64_t src, uint32_t st) {
    if (g_nmap >= MAX_MAP)
        die("mapping table full");
    g_map[g_nmap++] = (typeof(g_map[0])){b, n, src, st};
}

static void map_del(int i) {
    g_map[i] = g_map[--g_nmap];
}

static int64_t alias_pages(uint64_t off, uint64_t n, int map) {
    int64_t ch = 0;
    for (uint64_t p = off >> 12, e = (off + n + 0xFFF) >> 12; p < e;) {
        uint64_t sh = p & 63;
        uint64_t k = 64 - sh < e - p ? 64 - sh : e - p;
        uint64_t m = (k == 64 ? ~0ull : (1ull << k) - 1) << sh;
        uint64_t old = map
            ? atomic_fetch_or(&g_alias_bits[p >> 6], m)
            : atomic_fetch_and(&g_alias_bits[p >> 6], ~m);
        ch += __builtin_popcountll((map ? ~old : old) & m);
        p += k;
    }
    return map ? ch : -ch;
}

static int64_t alias_unmapped(uint64_t off, uint64_t n) {
    int64_t c = 0;
    for (uint64_t p = off >> 12, e = (off + n + 0xFFF) >> 12; p < e;) {
        uint64_t sh = p & 63;
        uint64_t k = 64 - sh < e - p ? 64 - sh : e - p;
        uint64_t m = (k == 64 ? ~0ull : (1ull << k) - 1) << sh;
        c += __builtin_popcountll(~atomic_load(&g_alias_bits[p >> 6]) & m);
        p += k;
    }
    return c;
}

static _Atomic int64_t g_alias_mapped;

static uint64_t alias_mask(unsigned n) {
    return n == 64 ? UINT64_MAX : (UINT64_C(1) << n) - 1;
}

static uint64_t alias_run_end(uint64_t p, uint64_t end, int mapped) {
    for (uint64_t first = p + 1; first < end;) {
        unsigned shift = first & 63;
        unsigned n = (unsigned)(end - first < 64 - shift ? end - first : 64 - shift);
        uint64_t bits = atomic_load(&g_alias_bits[first >> 6]);
        uint64_t different = (mapped ? ~bits : bits) & (alias_mask(n) << shift);
        if (different)
            return (first & ~UINT64_C(63)) + __builtin_ctzll(different);
        first += n;
    }
    return end;
}

static void alias_query(uint64_t addr, uint64_t *base, uint64_t *size, int *mapped) {
    uint64_t p = (addr - (uintptr_t)g_alias) >> 12;
    uint64_t lo = p;
    *mapped = (atomic_load(&g_alias_bits[p >> 6]) >> (p & 63)) & 1;

    while (lo) {
        uint64_t first = (lo - 1) & ~UINT64_C(63);
        uint64_t bits = atomic_load(&g_alias_bits[first >> 6]);
        uint64_t different = (*mapped ? ~bits : bits) & alias_mask((unsigned)(lo - first));
        if (different) {
            lo = first + 64 - __builtin_clzll(different);
            break;
        }
        lo = first;
    }

    uint64_t hi = alias_run_end(p, g_alias_cap >> 12, *mapped);
    *base = (uintptr_t)g_alias + (lo << 12);
    *size = (hi - lo) << 12;
}

static void alias_initialize(uint64_t off, size_t size) {
    size_t pg = (size_t)getpagesize();
    size_t ratio = pg >> 12;
    uint64_t end = (off + size) >> 12;

    for (uint64_t p = off >> 12; p < end;) {
        int mapped = (atomic_load(&g_alias_bits[p >> 6]) >> (p & 63)) & 1;
        uint64_t next = alias_run_end(p, end, mapped);

        if (!mapped) {
            uint64_t clear = p;
            for (uint64_t q = p; q < next;) {
                uint64_t host = q / ratio;
                uint64_t stop = (host + 1) * ratio;
                if (stop > next)
                    stop = next;

                if (!(g_alias_exposed[host >> 6] &
                      (UINT64_C(1) << (host & 63)))) {
                    if (clear < q)
                        memset(g_alias + (clear << 12), 0, (q - clear) << 12);
                    clear = stop;
                }
                q = stop;
            }
            if (clear < next)
                memset(g_alias + (clear << 12), 0, (next - clear) << 12);
        }
        p = next;
    }

    for (uint64_t h = off / pg, last = (off + size - 1) / pg; h <= last; ++h)
        g_alias_exposed[h >> 6] |= UINT64_C(1) << (h & 63);
}

static int memory_contains(uint64_t base, uint64_t capacity, uint64_t addr, uint64_t size) {
    return addr >= base && size <= capacity && addr - base <= capacity - size;
}

static uint32_t map_arguments(uint64_t dst, uint64_t src, uint64_t size) {
    if ((dst | src) & 0xfff)
        return 0xCC01;
    if (!size || (size & 0xfff))
        return 0xCA01;
    if (size > UINT64_MAX - src || size > UINT64_MAX - dst)
        return 0xD401;
    return memory_contains((uintptr_t)g_stack, g_stack_cap, dst, size) ? 0 : 0xDC01;
}

static int map_source_busy(uint64_t addr, uint64_t size) {
    for (int i = 0; i < g_nmap; ++i)
        if (addr < g_map[i].src + g_map[i].size &&
            g_map[i].src < addr + size)
            return 1;
    return 0;
}

static void module_readonly_run(uint64_t off, uint64_t start, uint64_t end,
                                uint64_t *lo, uint64_t *hi, uint64_t *perm) {
    if (start >= end)
        return;
    if (off < start) {
        if (start < *hi)
            *hi = start;
    } else if (off >= end) {
        if (end > *lo)
            *lo = end;
    } else {
        if (start > *lo)
            *lo = start;
        if (end < *hi)
            *hi = end;
        *perm &= ~UINT64_C(2);
    }
}

static void module_data_query(const module_t *m, uint64_t off,
                              uint64_t *lo, uint64_t *hi, uint64_t *perm) {
    module_readonly_run(off, 0, (m->text_end + 0x3fff) & ~UINT64_C(0x3fff),
                        lo, hi, perm);
    if (m->patch_off > 0)
        module_readonly_run(off, (uint64_t)m->patch_off & ~UINT64_C(0x3fff),
                            m->size, lo, hi, perm);
    module_readonly_run(off, m->key_page_off,
                        m->key_page_off + m->key_page_size, lo, hi, perm);
}

static int map_source_backed(uint64_t src, uint64_t size) {
    if (memory_contains((uintptr_t)g_heap, g_heap_cur, src, size))
        return 1;

    int dm_ue_heap_owns(uint64_t a, uint64_t n);
    if (dm_ue_heap_owns(src, size))
        return 1;

    if (memory_contains((uintptr_t)g_alias, g_alias_cap, src, size))
        return !alias_unmapped(src - (uintptr_t)g_alias, size);

    module_t *m = module_of((void *)(uintptr_t)src);
    if (!m || !m->loaded || !memory_contains((uintptr_t)m->base, m->size, src, size))
        return 0;

    uint64_t lo = m->data_off ? m->data_off : (m->ro_off ? m->ro_off : m->text_end);
    uint64_t hi = m->patch_off > 0 ? (uint64_t)m->patch_off : m->size;
    uint64_t off = src - (uintptr_t)m->base;
    uint64_t perm = 3;
    module_data_query(m, off, &lo, &hi, &perm);
    return perm == 3 && hi >= lo && memory_contains(lo, hi - lo, off, size);
}

int svc_nro_deactivate(module_t *m) {
    pthread_mutex_lock(&g_map_lock);
    uint64_t base = (uintptr_t)m->base;

    for (int i = 0; i < g_nmap; i++) {
        if (g_map[i].src < base + m->size &&
            base < g_map[i].src + g_map[i].size) {
            pthread_mutex_unlock(&g_map_lock);
            return 0;
        }
    }

    m->loaded = 0;
    pthread_mutex_unlock(&g_map_lock);
    return 1;
}

static void map_source_query(uint64_t addr, uint64_t *base, uint64_t *size,
                             uint32_t *attr, uint64_t *perm) {
    uint64_t lo = *base;
    uint64_t remaining = *size;

    for (int i = 0; i < g_nmap; ++i) {
        uint64_t src = g_map[i].src;
        uint64_t end = src + g_map[i].size;
        uint64_t lower = lo;
        uint64_t upper = remaining;

        if (addr >= src && addr < end) {
            if (src > lower)
                lower = src;
            if (end - lo < upper)
                upper = end - lo;
            *attr = 1;
            *perm = 0;
        } else if (end <= addr && end > lower) {
            lower = end;
        } else if (src > addr && src - lo < upper) {
            upper = src - lo;
        }

        remaining = upper - (lower - lo);
        lo = lower;
    }

    *base = lo;
    *size = remaining;
}

static void *alias_windows_thread(void *arg) {
    uint64_t win = (uint64_t)(uintptr_t)arg << 20;
    uint64_t a0 = (uint64_t)(uintptr_t)g_alias;
    uint64_t a1 = a0 + g_alias_cap;

    for (;;) {
        sleep(30);
        char line[8192] = {0};
        size_t o = 0;
        uint64_t tot = 0;

        pthread_mutex_lock(&g_map_lock);
        for (uint64_t w = a0 & ~(win - 1); w < a1 && o < sizeof line - 64; w += win) {
            uint64_t n = 0;
            uint64_t p0 = (w > a0 ? w - a0 : 0) >> 12;
            uint64_t p1 = ((w + win < a1 ? w + win : a1) - a0) >> 12;

            for (uint64_t p = p0; p < p1; p++)
                n += atomic_load(&g_alias_bits[p >> 6]) >> (p & 63) & 1;

            if (n) {
                o += snprintf(line + o, sizeof line - o, " +%#llx:%.1f",
                              (unsigned long long)(w - (a0 & ~(win - 1))),
                              n * 4096 / 1048576.0);
                tot += n;
            }
        }
        pthread_mutex_unlock(&g_map_lock);

        LOG("[mem] alias windows of %llu MB (window offset:MB mapped), %.1f MB in all:%s",
            (unsigned long long)(win >> 20), tot * 4096 / 1048576.0, line);
    }
    return NULL;
}

// ---- Address regions ----
typedef struct {
    uint64_t off, size, via;
    char who[40];
} region_t;

static region_t *g_rgn;
static int g_nrgn, g_rgn_cap;
static pthread_mutex_t g_rgn_lock = PTHREAD_MUTEX_INITIALIZER;
static _Thread_local uint64_t t_rgn_size, t_rgn_via;
static _Thread_local char t_rgn_who[40];

static void guest_caller(const uint64_t *r, char *buf, size_t n, uint64_t *via) {
    uint64_t lr = r[101], fp = r[29];
    module_t *m = module_of((void *)(uintptr_t)lr);
    *via = 0;

    for (int i = 0; i < 32 && m && !strcmp(m->name, "sdk"); i++) {
        *via = lr - (uint64_t)(uintptr_t)m->base;
        uint64_t rec[2];
        vm_size_t got = 0;

        if (!fp || (fp & 15) ||
            vm_read_overwrite(mach_task_self(), (vm_address_t)fp,
                              sizeof rec, (vm_address_t)rec, &got) ||
            got != sizeof rec)
            break;

        lr = rec[1];
        fp = rec[0];
        m = module_of((void *)(uintptr_t)lr);
    }

    if (m)
        snprintf(buf, n, "%s+%#llx", m->name,
                 (unsigned long long)(lr - (uint64_t)(uintptr_t)m->base));
    else
        snprintf(buf, n, "%#llx", (unsigned long long)lr);
}

static void region_ask(uint64_t *r, int param) {
    (void)param;
    t_rgn_size = r[2];
    guest_caller(r, t_rgn_who, sizeof t_rgn_who, &t_rgn_via);
}

static void region_got(uint64_t *r, int param) {
    mem_init();
    uint64_t a = param >= 0 && param < 31 ? r[param] : 0;
    uint64_t ab = (uint64_t)(uintptr_t)g_alias;

    if (a < ab || a - ab >= g_alias_cap)
        return;

    pthread_mutex_lock(&g_rgn_lock);

    if (g_nrgn == g_rgn_cap) {
        region_t *t = realloc(g_rgn, sizeof *t * (size_t)(g_rgn_cap ? 2 * g_rgn_cap : 256));
        if (t) {
            g_rgn = t;
            g_rgn_cap = g_rgn_cap ? 2 * g_rgn_cap : 256;
        }
    }

    int kept = g_nrgn < g_rgn_cap;
    if (kept) {
        g_rgn[g_nrgn].off = a - ab;
        g_rgn[g_nrgn].size = t_rgn_size;
        g_rgn[g_nrgn].via = t_rgn_via;
        memcpy(g_rgn[g_nrgn].who, t_rgn_who, sizeof t_rgn_who);
        g_nrgn++;
    }

    pthread_mutex_unlock(&g_rgn_lock);

    if (!kept) {
        static _Atomic int once;
        if (!atomic_fetch_add(&once, 1))
            LOG("[RED] address region table: out of memory at %d regions, later ones are not in it", g_nrgn);
    }

    if (t_rgn_size >= 0x10000000)
        LOG("[mem] address region %.2f GB at alias +%.2f GB, returning to %s",
            t_rgn_size / 1073741824.0, (a - ab) / 1073741824.0, t_rgn_who);
}

static void region_free(uint64_t *r, int param) {
    (void)param;
    uint64_t off = r[1] - (uint64_t)(uintptr_t)g_alias;

    pthread_mutex_lock(&g_rgn_lock);
    for (int i = 0; i < g_nrgn; i++) {
        if (g_rgn[i].off == off) {
            g_rgn[i] = g_rgn[--g_nrgn];
            break;
        }
    }
    pthread_mutex_unlock(&g_rgn_lock);
}

#define STDALLOC_LATER 0x10000000ull

static void stdalloc_init(uint64_t *r, int param) {
    (void)param;
    if (r[1])
        return;

    mem_init();
    uint64_t reach = 0x3e00000;

    pthread_mutex_lock(&g_rgn_lock);
    for (int i = 0; i < g_nrgn; i++)
        if (g_rgn[i].off + g_rgn[i].size > reach)
            reach = g_rgn[i].off + g_rgn[i].size;
    pthread_mutex_unlock(&g_rgn_lock);

    uint64_t cap = DM_GAME_STDALLOC_CAP
        ? DM_GAME_STDALLOC_CAP
        : g_alias_cap > reach + STDALLOC_LATER
            ? (g_alias_cap - reach - STDALLOC_LATER) & ~0x1ffffffull
            : 0;

    if (!cap || r[2] <= cap)
        return;

    char who[40];
    uint64_t via;
    guest_caller(r, who, sizeof who, &via);

    LOG("[mem] StandardAllocator::Initialize(nullptr, %.2f GB) returning to %s: %.2f GB of address space, %s",
        r[2] / 1073741824.0, who, cap / 1073741824.0,
        DM_GAME_STDALLOC_CAP
            ? "the profile's stdalloc_cap"
            : "what the alias region has room for (the profile key stdalloc_cap sets it)");

    r[2] = cap;
}

__attribute__((constructor))
static void regions_attach(void) {
    gate_obs_set(OBS_REGION_ASK, region_ask);
    gate_obs_set(OBS_REGION_GOT, region_got);
    gate_obs_set(OBS_REGION_FREE, region_free);
    gate_obs_set(OBS_STDALLOC_INIT, stdalloc_init);
}

static uint64_t bits_set(_Atomic uint64_t *b, uint64_t lo, uint64_t hi) {
    uint64_t n = 0;
    for (; lo < hi && (lo & 63); lo++)
        n += atomic_load(&b[lo >> 6]) >> (lo & 63) & 1;
    for (; lo + 64 <= hi; lo += 64)
        n += (uint64_t)__builtin_popcountll(atomic_load(&b[lo >> 6]));
    for (; lo < hi; lo++)
        n += atomic_load(&b[lo >> 6]) >> (lo & 63) & 1;
    return n;
}

static void bits_span(const uint64_t *b, uint64_t lo, uint64_t hi,
                      int64_t *first, int64_t *last) {
    *first = *last = -1;

    for (uint64_t i = lo; i < hi;) {
        uint64_t w = b[i >> 6] >> (i & 63);
        if (!w) {
            i = (i | 63) + 1;
            continue;
        }

        i += (uint64_t)__builtin_ctzll(w);
        if (i >= hi)
            break;

        if (*first < 0)
            *first = (int64_t)i;
        *last = (int64_t)i;
        i++;
    }
}

static void *regions_thread(void *arg) {
    const char *path = arg;
    char tmp[1024];
    snprintf(tmp, sizeof tmp, "%s.partial", path);
    uint64_t hpg = (uint64_t)getpagesize();

    for (;;) {
        FILE *f = fopen(tmp, "w");

        if (f) {
            fprintf(f, "# alias region %#llx; offset\tsize\treturns to\tmapped\tfirst opened\tend of the last opened (region-relative; - = none)\tvia (sdk offset)\n",
                    (unsigned long long)g_alias_cap);

            pthread_mutex_lock(&g_rgn_lock);
            int n = g_nrgn;
            region_t *t = malloc(sizeof *t * (size_t)(n ? n : 1));
            if (t && n)
                memcpy(t, g_rgn, sizeof *t * (size_t)n);
            pthread_mutex_unlock(&g_rgn_lock);

            pthread_mutex_lock(&g_map_lock);

            for (int i = 0; t && i < n; i++) {
                uint64_t lo = t[i].off;
                uint64_t hi = t[i].off + t[i].size > g_alias_cap
                    ? g_alias_cap : t[i].off + t[i].size;
                uint64_t mapped = bits_set(g_alias_bits, lo >> 12, hi >> 12);

                int64_t first, last;
                bits_span(g_alias_exposed, lo / hpg, (hi + hpg - 1) / hpg,
                          &first, &last);

                char fo[24] = "-";
                char lo2[24] = "-";

                if (first >= 0) {
                    snprintf(fo, sizeof fo, "%#llx",
                             (unsigned long long)((uint64_t)first * hpg > lo
                                 ? (uint64_t)first * hpg - lo : 0));
                    snprintf(lo2, sizeof lo2, "%#llx",
                             (unsigned long long)((uint64_t)(last + 1) * hpg - lo));
                }

                fprintf(f, "%#llx\t%#llx\t%s\t%#llx\t%s\t%s\t%#llx\n",
                        (unsigned long long)t[i].off,
                        (unsigned long long)t[i].size,
                        t[i].who,
                        (unsigned long long)mapped * 4096,
                        fo,
                        lo2,
                        (unsigned long long)t[i].via);
            }

            pthread_mutex_unlock(&g_map_lock);
            free(t);
            fclose(f);
            rename(tmp, path);
        }

        sleep(2);
    }

    return NULL;
}

int svc_thread(int num, uint64_t *regs);

uint64_t svc_dispatch(int num, uint64_t *regs, uint64_t orig_addr) {
    {
        void hlecount_svc(int);
        hlecount_svc(num);
    }

    mem_init();

    if (svc_thread(num, regs))
        return regs[0];

    switch (num) {
    case 0x01: {
        size_t size = regs[1];

        if (size > g_heap_cap) {
            LOG("SetHeapSize OOM: req %#zx > cap %#zx", size, g_heap_cap);
            regs[0] = 0xD001;
            return 0;
        }

        pthread_mutex_lock(&g_map_lock);
        size_t previous = g_heap_cur;
        size_t image = image_bytes();

        if (size < previous && map_source_busy((uintptr_t)g_heap + size, previous - size)) {
            pthread_mutex_unlock(&g_map_lock);
            regs[0] = 0xD401;
            return 0;
        }

        if (image > APP_POOL || size > APP_POOL - image) {
            static int once;
            if (!once++)
                LOG("[mem] SetHeapSize %#zx + modules %#zx > app pool %#llx (used memory reported %#llx + bss %#zx)",
                    size, image_bytes() - big_bss_bytes(),
                    (unsigned long long)APP_POOL,
                    (unsigned long long)DM_GAME_MEM_USED,
                    big_bss_bytes());

            pthread_mutex_unlock(&g_map_lock);
            regs[0] = 0x10801;
            return 0;
        }

        uint64_t charged = (uint64_t)DM_GAME_MEM_USED +
                           big_bss_bytes() +
                           atomic_load(&g_alias_mapped);

        if (DM_GAME_MEM_MAP_LIMIT && (charged > APP_POOL || size > APP_POOL - charged)) {
            pthread_mutex_unlock(&g_map_lock);
            regs[0] = 0x10801;
            return 0;
        }

        int rc = 0;
        if (size > g_heap_cur)
            rc = mprotect(g_heap + g_heap_cur, size - g_heap_cur, PROT_READ | PROT_WRITE);
        else if (size < g_heap_cur)
            rc = mprotect(g_heap + size, g_heap_cur - size, PROT_NONE);

        if (rc)
            die("SetHeapSize mprotect failed: req=%#zx errno=%d", size, errno);

        if (g_flag.svc)
            LOG("SetHeapSize %#zx -> base %p mprotect=%d errno=%d", size, g_heap, rc, errno);
        else if (size > g_heap_cur)
            LOG("[mem] heap %.2f GB (heap region %.2f GB)",
                size / 1073741824.0, g_heap_cap / 1073741824.0);

        g_heap_cur = size;
        pthread_mutex_unlock(&g_map_lock);

        regs[0] = 0;
        regs[1] = (uint64_t)g_heap;
        return 0;
    }

    case 0x02:
    case 0x03: {
        uint64_t addr = regs[0];
        uint64_t size = regs[1];

        pthread_mutex_lock(&g_map_lock);
        int busy = size > UINT64_MAX - addr ||
                   (size && map_source_busy(addr, size));
        pthread_mutex_unlock(&g_map_lock);

        regs[0] = busy ? 0xD401 : 0;
        return 0;
    }

    case 0x04: {
        uint64_t dst = regs[0];
        uint64_t src = regs[1];
        uint64_t size = regs[2];
        uint32_t result = map_arguments(dst, src, size);

        if (result) {
            regs[0] = result;
            return 0;
        }

        pthread_mutex_lock(&g_map_lock);

        if (map_find(dst, size) || map_source_busy(src, size) || !map_source_backed(src, size))
            result = 0xD401;
        else if (g_nmap == MAX_MAP)
            result = 0xCE01;

        if (result) {
            pthread_mutex_unlock(&g_map_lock);
            regs[0] = result;
            LOG("[RED] MapMemory refused (%#x): %#llx <- %#llx +%#llx",
                result, (unsigned long long)dst,
                (unsigned long long)src, (unsigned long long)size);
            return 0;
        }

        uint64_t mask = (size_t)getpagesize() - 1;
        uint64_t lo = dst & ~mask;
        uint64_t hi = (dst + size + mask) & ~mask;

        if (mprotect((void *)lo, hi - lo, PROT_READ | PROT_WRITE))
            die("MapMemory mprotect errno=%d", errno);

        dm_guest_write_prepare((void *)lo, hi - lo);
        memcpy((void *)dst, (void *)src, size);
        map_add(dst, size, src, 0x0B);

        pthread_mutex_unlock(&g_map_lock);

        if (g_flag.svc)
            LOG("MapMemory %#llx <- %#llx +%#llx",
                (unsigned long long)dst,
                (unsigned long long)src,
                (unsigned long long)size);

        regs[0] = 0;
        return 0;
    }

    case 0x05: {
        uint64_t dst = regs[0];
        uint64_t src = regs[1];
        uint64_t size = regs[2];
        uint32_t result = map_arguments(dst, src, size);

        if (result) {
            regs[0] = result;
            return 0;
        }

        pthread_mutex_lock(&g_map_lock);
        int k = map_find(dst, size);

        if (!k || g_map[k - 1].base != dst ||
            g_map[k - 1].size != size || g_map[k - 1].src != src) {
            pthread_mutex_unlock(&g_map_lock);
            LOG("UnmapMemory does not match a mapping %#llx+%#llx",
                (unsigned long long)dst, (unsigned long long)size);
            regs[0] = 0xD401;
            return 0;
        }

        dm_guest_write_prepare((void *)src, size);
        memcpy((void *)src, (void *)dst, size);
        map_del(k - 1);

        pthread_mutex_unlock(&g_map_lock);
        regs[0] = 0;
        return 0;
    }

    case 0x06: {
        uint64_t mi = regs[0];
        uint64_t addr = regs[2];
        uint8_t *a = (uint8_t *)addr;
        uint64_t base, size, state, perm;
        uint32_t attr = 0;
        int k;

        pthread_mutex_lock(&g_map_lock);

        uint64_t ab = (uintptr_t)g_alias;
        uint64_t ae = ab + g_alias_cap;
        uint64_t rb = 0, rn = 0;
        int mapped = 0;
        int in_alias = addr >= ab && addr < ae;

        if (in_alias)
            alias_query(addr, &rb, &rn, &mapped);

        if (a >= g_heap && a < g_heap + g_heap_cur) {
            base = (uint64_t)g_heap;
            size = g_heap_cur;
            state = 5;
            perm = 3;
        } else if ((k = map_find(addr, 1))) {
            base = g_map[k - 1].base;
            size = g_map[k - 1].size;
            state = g_map[k - 1].state;
            perm = 3;
        } else if (in_alias && mapped) {
            base = rb;
            size = rn;
            state = 5;
            perm = 3;
        } else if (a >= g_as && a < g_as + g_reserve) {
            uint64_t lo = (uint64_t)g_as;
            uint64_t hi = (uint64_t)g_as + g_reserve;

            if (a >= g_heap)
                lo = (uint64_t)g_heap + g_heap_cur;
            else if (hi > (uint64_t)g_heap)
                hi = (uint64_t)g_heap;

            if (in_alias) {
                if (rb > ab && rb > lo)
                    lo = rb;
                if (rb + rn < ae && rb + rn < hi)
                    hi = rb + rn;
            } else if (g_alias_cap && addr < ab) {
                alias_query(ab, &rb, &rn, &mapped);
                uint64_t edge = mapped ? ab : rb + rn;
                if ((mapped || edge < ae) && edge < hi)
                    hi = edge;
            } else if (g_alias_cap) {
                alias_query(ae - 1, &rb, &rn, &mapped);
                uint64_t edge = mapped ? ae : rb;
                if ((mapped || edge > ab) && edge > lo)
                    lo = edge;
            }

            for (int i = 0; i < g_nmap; i++) {
                uint64_t e = g_map[i].base + g_map[i].size;
                if (e <= addr && e > lo)
                    lo = e;
                if (g_map[i].base > addr && g_map[i].base < hi)
                    hi = g_map[i].base;
            }

            base = lo;
            size = hi - lo;
            state = 0;
            perm = 0;
        } else if (module_of(a)) {
            module_t *mm = module_of(a);
            uint64_t off = (uint64_t)(a - mm->base);
            uint64_t mb = (uint64_t)mm->base;
            uint64_t ro = mm->ro_off ? mm->ro_off : mm->text_end;
            uint64_t da = mm->data_off ? mm->data_off : ro;
            uint64_t pa = mm->patch_off > 0 ? (uint64_t)mm->patch_off : mm->size;

            if (a < mm->base) {
                base = mb + mm->patch_off;
                size = mm->patch_len;
                state = 4;
                perm = 5;
            } else if (off < ro) {
                base = mb;
                size = ro;
                state = 3;
                perm = 5;
            } else if (off < da) {
                base = mb + ro;
                size = da - ro;
                state = 3;
                perm = 1;
            } else if (off < pa) {
                base = mb + da;
                size = pa - da;
                state = 4;
                perm = 3;
            } else {
                base = mb + pa;
                size = mm->size - pa;
                state = 4;
                perm = 5;
            }

            if (state == 4 && perm == 3) {
                uint64_t lo = base - mb;
                uint64_t hi = lo + size;
                module_data_query(mm, off, &lo, &hi, &perm);
                base = mb + lo;
                size = hi - lo;
            }
        } else {
            uint64_t top = (uint64_t)pthread_get_stackaddr_np(pthread_self());
            uint64_t ssz = (uint64_t)pthread_get_stacksize_np(pthread_self());

            if (addr < top && addr >= top - ssz) {
                base = top - ssz;
                size = ssz;
                state = 0xB;
                perm = 3;
            } else {
                base = addr & ~0xFFFull;
                size = 0x1000;
                state = 0;
                perm = 0;
            }
        }

        map_source_query(addr, &base, &size, &attr, &perm);
        pthread_mutex_unlock(&g_map_lock);

        if (mi) {
            struct {
                uint64_t base, size;
                uint32_t state, attr, perm, ipc, device, reserved;
            } out = {base, size, (uint32_t)state, attr, (uint32_t)perm, 0, 0, 0};

            _Static_assert(sizeof out == 40, "MemoryInfo ABI size");
            memcpy((void *)(uintptr_t)mi, &out, sizeof out);
        }

        regs[0] = 0;
        regs[1] = 0;
        return 0;
    }

    case 0x2C:
    case 0x2D: {
        uintptr_t a = regs[0];
        size_t size = regs[1];

        if (!size || ((a | size) & 0xfff) ||
            !memory_contains((uintptr_t)g_alias, g_alias_cap, a, size)) {
            LOG("SVC %#x outside the alias region %p+%#zx", num, (void *)a, size);
            regs[0] = 0xD401;
            return 0;
        }

        pthread_mutex_lock(&g_map_lock);

        if (num == 0x2D && map_source_busy(a, size)) {
            pthread_mutex_unlock(&g_map_lock);
            regs[0] = 0xD401;
            return 0;
        }

        uint64_t off = a - (uintptr_t)g_alias;
        int64_t need = alias_unmapped(off, size) * 4096;

        if (DM_GAME_MEM_MAP_LIMIT && num == 0x2C) {
            int64_t room = (int64_t)APP_POOL -
                (int64_t)(g_heap_cur + DM_GAME_MEM_USED + big_bss_bytes()) -
                atomic_load(&g_alias_mapped);

            if (need && need > room) {
                pthread_mutex_unlock(&g_map_lock);
                static _Atomic int nref;

                if (atomic_fetch_add(&nref, 1) < 8)
                    LOG("[mem] MapPhysicalMemory +%#zx refused: %.2f GB mapped, the application pool is full (ResultLimitReached, the SDK frees cached pages and retries)",
                        size, atomic_load(&g_alias_mapped) / 1073741824.0);

                regs[0] = 0x10801;
                return 0;
            }
        }

        uintptr_t pg = (uintptr_t)getpagesize();
        uintptr_t lo = (uintptr_t)a;
        uintptr_t hi = (uintptr_t)a + size;
        uintptr_t mlo = lo & ~(pg - 1);
        uintptr_t mhi = (hi + pg - 1) & ~(pg - 1);
        int commit = num == 0x2C;
        int64_t changed = commit ? need : (int64_t)size - need;

        if (!commit) {
            if ((lo - mlo) / 4096 !=
                (uint64_t)alias_unmapped(mlo - (uintptr_t)g_alias, lo - mlo))
                mlo += pg;

            if ((mhi - hi) / 4096 !=
                (uint64_t)alias_unmapped(hi - (uintptr_t)g_alias, mhi - hi))
                mhi -= pg;
        }

        if (changed && mhi > mlo) {
            if (mprotect((void *)mlo, mhi - mlo,
                         commit ? PROT_READ | PROT_WRITE : PROT_NONE))
                die("SVC %#x mprotect %p+%#zx errno=%d",
                    num, (void *)a, size, errno);

            if (commit)
                dm_guest_write_prepare((void *)mlo, mhi - mlo);
            else
                (void)madvise((void *)mlo, mhi - mlo, MADV_FREE);
        }

        if (commit && changed)
            alias_initialize(off, size);

        if (g_flag.svc)
            LOG("%sPhysicalMemory %p+%#zx",
                commit ? "Map" : "Unmap", (void *)a, size);

        if (num == 0x2C) {
            static _Atomic uint64_t hw;
            uint64_t end = off + size;
            uint64_t old = atomic_load(&hw);

            while (end > old && !atomic_compare_exchange_weak(&hw, &old, end)) {}

            if (end > old && end >> 28 != old >> 28)
                LOG("[mem] alias region mapped up to +%.2f GB (alias region %.2f GB)",
                    end / 1073741824.0, g_alias_cap / 1073741824.0);
        }

        {
            int64_t d = alias_pages(off, size, commit) * 4096;
            int64_t before = atomic_fetch_add(&g_alias_mapped, d);
            int64_t after = before + d;

            pthread_mutex_unlock(&g_map_lock);

            if (before >> 28 != after >> 28)
                LOG("[mem] alias region mapped %.2f GB (application pool %.2f GB)",
                    after / 1073741824.0, APP_POOL / 1073741824.0);
        }

        regs[0] = 0;
        return 0;
    }

    case 0x26: {
        uint32_t res = (regs[1] && regs[2] >= 4) ? *(uint32_t *)regs[1] : 0;
        module_t *mm = module_of((void *)regs[30]);
        static int nb;

        LOG("Break#%d reason=%#llx Result=%#x svc at %s+%#llx",
            ++nb, regs[0], res, mm ? mm->name : "?", orig_addr);
        dm_backtrace(regs[29]);

        if (regs[0] & 0x80000000ull)
            return 0;

        _exit(42);
    }

    case 0x1E:
        guest_clock_read_wait();
        regs[0] = 0;
        regs[1] = guest_ns() * 12 / 625;
        return 0;

    case 0x27:
        if (g_flag.dbg)
            fprintf(stderr, "[guest] %.*s\n", (int)regs[1], (char *)regs[0]);
        regs[0] = 0;
        return 0;

    case 0x29: {
        uint64_t id = regs[1];
        uint64_t v = 0;

        switch (id) {
        case I_CoreMask:
            v = 7;
            break;
        case I_PriorityMask:
            v = 0;
            break;
        case I_AliasAddr:
            v = (uint64_t)g_alias;
            break;
        case I_AliasSize:
            v = g_alias_cap;
            break;
        case I_HeapAddr:
            v = (uint64_t)g_heap;
            break;
        case I_HeapSize:
            v = g_heap_cap;
            break;
        case I_AslrAddr:
            v = (uint64_t)g_as;
            break;
        case I_AslrSize:
            v = g_reserve;
            break;
        case I_StackAddr:
            v = (uint64_t)g_stack;
            break;
        case I_StackSize:
            v = g_stack_cap;
            break;
        case I_TotalMem:
        case I_TotalNonSys:
            v = APP_POOL;
            break;
        case I_UsedMem:
        case I_UsedNonSys:
            pthread_mutex_lock(&g_map_lock);
            v = g_heap_cur + DM_GAME_MEM_USED + big_bss_bytes();
            if (DM_GAME_MEM_MAP_LIMIT)
                v += (uint64_t)atomic_load(&g_alias_mapped);
            pthread_mutex_unlock(&g_map_lock);
            break;
        case I_SysResTotal:
            v = DM_GAME_SYS_RESOURCE;
            break;
        case I_SysResUsed:
            v = DM_GAME_SYS_RESOURCE ? 0x400000 : 0;
            break;
        case I_DebuggerAttached:
            v = 0;
            break;
        case I_RandomEntropy: {
            uint64_t dm_entropy64(void);
            v = dm_entropy64();
            break;
        }
        case I_ProgramId: {
            static uint64_t pid;
            if (!pid) {
                const char *e = getenv("DM_PROGRAM_ID");
                pid = e ? strtoull(e, NULL, 16) : DM_GAME_PROGRAM_ID;
            }
            v = pid;
            break;
        }
        case I_IsApplication:
            v = 1;
            break;
        case I_FreeThreadCount: {
            int thr_limit_used(void);
            int u = thr_limit_used();
            v = u < 96 ? 96 - u : 0;
            break;
        }
        case I_UserExcAddr: {
            static _Alignas(0x1000) uint8_t ctx[0x1000];
            v = (uint64_t)ctx;
            break;
        }
        default:
            v = 0;
            break;
        }

        regs[0] = 0;
        regs[1] = v;
        return 0;
    }

    default:
        if (!g_seen[num & 0xff]) {
            g_seen[num & 0xff] = 1;
            LOG("SVC %#x x0=%#llx x1=%#llx x2=%#llx (placeholder returns success)",
                num, regs[0], regs[1], regs[2]);
        }

        regs[0] = 0;
        return 0;
    }
}
