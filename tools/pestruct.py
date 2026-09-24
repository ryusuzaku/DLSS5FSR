"""
pestruct.py - dependency-free PE / CUDA fatbin / CUBIN(ELF) structure parser.

Written for the DLSSNR (nvngx_dlssnr.dll) forensic pipeline.
Stdlib only so it runs anywhere the repo is checked out.
"""

from __future__ import annotations

import struct
import json
from dataclasses import dataclass, field, asdict
from typing import Iterator

# ---------------------------------------------------------------------------
# constants
# ---------------------------------------------------------------------------

FATBIN_MAGIC = 0xBA55ED50          # NVIDIA fat binary container
FATBIN_MAGIC_BYTES = struct.pack("<I", FATBIN_MAGIC)

ELF_MAGIC = b"\x7fELF"

# Windows resource types we care about
RT_CURSOR = 1
RT_BITMAP = 2
RT_ICON = 3
RT_MENU = 4
RT_DIALOG = 5
RT_STRING = 6
RT_FONTDIR = 7
RT_FONT = 8
RT_ACCELERATOR = 9
RT_RCDATA = 10
RT_MESSAGETABLE = 11
RT_GROUP_CURSOR = 12
RT_GROUP_ICON = 14
RT_VERSION = 16
RT_MANIFEST = 24

RT_NAMES = {
    1: "RT_CURSOR", 2: "RT_BITMAP", 3: "RT_ICON", 4: "RT_MENU", 5: "RT_DIALOG",
    6: "RT_STRING", 7: "RT_FONTDIR", 8: "RT_FONT", 9: "RT_ACCELERATOR",
    10: "RT_RCDATA", 11: "RT_MESSAGETABLE", 12: "RT_GROUP_CURSOR",
    14: "RT_GROUP_ICON", 16: "RT_VERSION", 24: "RT_MANIFEST",
}

# Image directory entries
DIR_EXPORT = 0
DIR_IMPORT = 1
DIR_RESOURCE = 2
DIR_EXCEPTION = 3
DIR_SECURITY = 4
DIR_RELOC = 5
DIR_DEBUG = 6
DIR_TLS = 9
DIR_DELAY_IMPORT = 13

SECTION_FLAGS = [
    (0x00000020, "CNT_CODE"),
    (0x00000040, "CNT_INITIALIZED_DATA"),
    (0x00000080, "CNT_UNINITIALIZED_DATA"),
    (0x02000000, "MEM_DISCARDABLE"),
    (0x10000000, "MEM_SHARED"),
    (0x20000000, "MEM_EXECUTE"),
    (0x40000000, "MEM_READ"),
    (0x80000000, "MEM_WRITE"),
]

MACHINE = {
    0x014C: "i386", 0x8664: "AMD64", 0xAA64: "ARM64", 0x01C4: "ARMNT",
}


def sec_flags_str(ch: int) -> str:
    return "|".join(n for b, n in SECTION_FLAGS if ch & b)


# ---------------------------------------------------------------------------
# PE
# ---------------------------------------------------------------------------

@dataclass
class Section:
    name: str
    vsize: int
    vaddr: int
    rawsize: int
    rawoff: int
    flags: int
    flags_str: str

    def contains_rva(self, rva: int) -> bool:
        return self.vaddr <= rva < self.vaddr + max(self.vsize, self.rawsize)

    def rva_to_off(self, rva: int) -> int | None:
        if not self.contains_rva(rva):
            return None
        return self.rawoff + (rva - self.vaddr)


@dataclass
class ResourceEntry:
    type_id: int
    type_name: str
    name_id: int
    name: str
    lang: int
    rva: int
    size: int
    file_offset: int | None


@dataclass
class PE:
    path: str
    data: bytes = field(repr=False, default=b"")
    machine: int = 0
    machine_name: str = ""
    is_pe32plus: bool = False
    image_base: int = 0
    entry_rva: int = 0
    num_sections: int = 0
    sections: list[Section] = field(default_factory=list)
    directories: list[tuple[int, int]] = field(default_factory=list)
    resources: list[ResourceEntry] = field(default_factory=list)
    imports: list[str] = field(default_factory=list)
    exports: list[tuple[int, str]] = field(default_factory=list)

    # -- addressing ---------------------------------------------------------
    def rva_to_off(self, rva: int) -> int | None:
        for s in self.sections:
            o = s.rva_to_off(rva)
            if o is not None:
                return o
        return None

    def u32(self, off: int) -> int:
        return struct.unpack_from("<I", self.data, off)[0]

    def u16(self, off: int) -> int:
        return struct.unpack_from("<H", self.data, off)[0]

    def u64(self, off: int) -> int:
        return struct.unpack_from("<Q", self.data, off)[0]


def parse_pe(path: str) -> PE:
    with open(path, "rb") as fh:
        data = fh.read()

    pe = PE(path=path, data=data)

    if data[:2] != b"MZ":
        raise ValueError("not an MZ/PE image")
    e_lfanew = struct.unpack_from("<I", data, 0x3C)[0]
    if data[e_lfanew:e_lfanew + 4] != b"PE\x00\x00":
        raise ValueError(f"bad PE signature at 0x{e_lfanew:X}")

    coff = e_lfanew + 4
    pe.machine = struct.unpack_from("<H", data, coff)[0]
    pe.machine_name = MACHINE.get(pe.machine, hex(pe.machine))
    pe.num_sections = struct.unpack_from("<H", data, coff + 2)[0]
    opt_size = struct.unpack_from("<H", data, coff + 16)[0]
    opt = coff + 20
    magic = struct.unpack_from("<H", data, opt)[0]
    pe.is_pe32plus = magic == 0x20B
    if magic == 0x20B:
        pe.image_base = struct.unpack_from("<Q", data, opt + 24)[0]
        pe.entry_rva = struct.unpack_from("<I", data, opt + 16)[0]
        num_rva = struct.unpack_from("<I", data, opt + 108)[0]
        dir_off = opt + 112
    elif magic == 0x10B:
        pe.image_base = struct.unpack_from("<I", data, opt + 28)[0]
        pe.entry_rva = struct.unpack_from("<I", data, opt + 16)[0]
        num_rva = struct.unpack_from("<I", data, opt + 92)[0]
        dir_off = opt + 96
    else:
        raise ValueError(f"unknown optional header magic 0x{magic:X}")

    for i in range(num_rva):
        rva, size = struct.unpack_from("<II", data, dir_off + i * 8)
        pe.directories.append((rva, size))

    sec_off = opt + opt_size
    for i in range(pe.num_sections):
        b = sec_off + i * 40
        name = data[b:b + 8].rstrip(b"\x00").decode("ascii", "replace")
        vsize, vaddr, rawsize, rawoff = struct.unpack_from("<IIII", data, b + 8)
        # some linkers leave rawsize 0 for BSS; clamp to file
        rawoff = min(rawoff, len(data))
        rawsize = min(rawsize, max(0, len(data) - rawoff))
        ch = struct.unpack_from("<I", data, b + 36)[0]
        pe.sections.append(Section(name, vsize, vaddr, rawsize, rawoff, ch,
                                   sec_flags_str(ch)))

    if len(pe.directories) > DIR_RESOURCE and pe.directories[DIR_RESOURCE][0]:
        pe.resources = _parse_resources(pe, pe.directories[DIR_RESOURCE][0])
    if len(pe.directories) > DIR_IMPORT and pe.directories[DIR_IMPORT][0]:
        pe.imports = _parse_imports(pe, pe.directories[DIR_IMPORT][0])
    if len(pe.directories) > DIR_EXPORT and pe.directories[DIR_EXPORT][0]:
        pe.exports = _parse_exports(pe, pe.directories[DIR_EXPORT][0])
    return pe


def _parse_resources(pe: PE, rva: int) -> list[ResourceEntry]:
    """
    Walk the 3-level (type / name / lang) resource tree.

    Every OffsetToData inside the tree is relative to the resource
    directory base -- only the OffsetToData inside the leaf
    IMAGE_RESOURCE_DATA_ENTRY is an RVA. Getting this wrong is the classic
    reason a hand-rolled resource parser reports "no resources" on a PE
    that clearly has them.
    """
    out: list[ResourceEntry] = []
    base = pe.rva_to_off(rva)
    if base is None:
        return out

    def wstr(off: int) -> str:
        s = base + off
        ln = pe.u16(s)
        return pe.data[s + 2:s + 2 + ln * 2].decode("utf-16-le", "replace")

    def entries(off: int) -> list[tuple[int, int, bool]]:
        named = pe.u16(off + 12)
        ids = pe.u16(off + 14)
        res = []
        e = off + 16
        for _ in range(named + ids):
            name_field = pe.u32(e)
            data_field = pe.u32(e + 4)
            res.append((name_field, data_field & 0x7FFFFFFF,
                        bool(data_field & 0x80000000)))
            e += 8
        return res

    for t_id, t_off, t_sub in entries(base):
        if not t_sub or base + t_off + 16 > len(pe.data):
            continue
        type_id = t_id
        type_name = RT_NAMES.get(t_id, f"#{t_id}")
        if t_id & 0x80000000:
            type_name = wstr(t_id & 0x7FFFFFFF)
            type_id = -1

        for n_id, n_off, n_sub in entries(base + t_off):
            if not n_sub or base + n_off + 16 > len(pe.data):
                continue
            name_id = n_id
            name = f"#{n_id}"
            if n_id & 0x80000000:
                name = wstr(n_id & 0x7FFFFFFF)
                name_id = -1

            for l_id, l_off, l_sub in entries(base + n_off):
                d = base + l_off
                if d + 16 > len(pe.data):
                    continue
                data_rva = pe.u32(d)
                size = pe.u32(d + 4)
                out.append(ResourceEntry(type_id, type_name, name_id, name,
                                         l_id, data_rva, size,
                                         pe.rva_to_off(data_rva)))
    return out


def _parse_imports(pe: PE, rva: int) -> list[str]:
    out = []
    off = pe.rva_to_off(rva)
    if off is None:
        return out
    step = 20
    while True:
        ilt, ts, fc, name_rva, ft = struct.unpack_from("<IIIII", pe.data, off)
        if name_rva == 0:
            break
        n_off = pe.rva_to_off(name_rva)
        if n_off is None:
            break
        end = pe.data.find(b"\x00", n_off)
        out.append(pe.data[n_off:end].decode("ascii", "replace"))
        off += step
    return out


def _parse_exports(pe: PE, rva: int) -> list[tuple[int, str]]:
    off = pe.rva_to_off(rva)
    if off is None:
        return []
    num_fn = pe.u32(off + 20)
    num_names = pe.u32(off + 24)
    fn_rva = pe.u32(off + 28)
    name_rva = pe.u32(off + 32)
    ord_rva = pe.u32(off + 36)
    f_off = pe.rva_to_off(fn_rva)
    n_off = pe.rva_to_off(name_rva)
    o_off = pe.rva_to_off(ord_rva)
    if not (f_off and n_off and o_off):
        return []
    out = []
    for i in range(num_names):
        s_rva = pe.u32(n_off + i * 4)
        s_off = pe.rva_to_off(s_rva)
        if s_off is None:
            continue
        end = pe.data.find(b"\x00", s_off)
        nm = pe.data[s_off:end].decode("ascii", "replace")
        ordn = pe.u16(o_off + i * 2)
        out.append((ordn, nm))
    return sorted(out)


# ---------------------------------------------------------------------------
# CUDA fatbin
# ---------------------------------------------------------------------------

@dataclass
class FatbinEntry:
    index: int
    kind: int
    version: int
    header_size: int
    payload_size: int
    name: str
    offset: int          # absolute file offset of entry header
    payload_offset: int  # absolute file offset of payload
    is_elf: bool
    arch: str = ""
    is_ptx: bool = False
    sm: int = 0
    uncompressed_size: int = 0
    container_total: int = 0
    compression: str = ""
    entry_size: int = 0


def _cstr(data: bytes, off: int, limit: int = 256) -> str:
    end = data.find(b"\x00", off, min(off + limit, len(data)))
    if end < 0:
        end = min(off + limit, len(data))
    return data[off:end].decode("ascii", "replace")


def find_fatbins(data: bytes) -> list[int]:
    offs = []
    i = data.find(FATBIN_MAGIC_BYTES)
    while i != -1:
        offs.append(i)
        i = data.find(FATBIN_MAGIC_BYTES, i + 1)
    return offs


def parse_fatbin(data: bytes, base_off: int) -> list[FatbinEntry]:
    """
    Parse a fatbin container at absolute offset `base_off`.

    Layout as found in nvngx_dlssnr.dll (payloads are ZSTD-compressed):

      container @ base_off
        +0x00 u32 magic        0xBA55ED50
        +0x04 u16 version      1
        +0x06 u16 header_size  0x10
        +0x08 u64 total_bytes  sum of all entry_size fields
      repeated entries, entry[n+1] starts at entry[n] + entry_size
        +0x00 u16 kind         (1 = PTX, 2 = ELF)
        +0x02 u16 version      0x0101
        +0x04 u32 header_size  0x40
        +0x08 u64 entry_size   == align8(payload_size)
        +0x10 u32 payload_size
        +0x14 u32 unknown      (0 on sm_75 builds, 0x40 on sm_120 builds)
        +0x18 u16 unknown
        +0x1A u16 unknown
        +0x1C u32 sm_arch      e.g. 75 -> sm_75, 120 -> sm_120
        +0x38 u64 uncompressed_size   (fixed offset in every build seen)
        +0x40 char ptxas_options[]    only when header_size > 0x40
        +header_size payload   (ZSTD frame, or raw ELF)

    header_size varies (0x40 on sm_75/86/89 builds, 0x78 on sm_120 builds,
    which additionally store the ptxas option string). Always add
    header_size -- never assume 0x40.
    """
    entries: list[FatbinEntry] = []
    if struct.unpack_from("<I", data, base_off)[0] != FATBIN_MAGIC:
        return entries
    version = struct.unpack_from("<H", data, base_off + 4)[0]
    chs = struct.unpack_from("<H", data, base_off + 6)[0]
    total = struct.unpack_from("<Q", data, base_off + 8)[0]

    p = base_off + chs
    idx = 0
    # `total` counts bytes of the container body (headers + payloads).
    end = base_off + chs + total
    while p + 0x48 <= len(data) and p + 0x48 <= end:
        kind = struct.unpack_from("<H", data, p)[0]
        ever = struct.unpack_from("<H", data, p + 2)[0]
        e_hdr = struct.unpack_from("<I", data, p + 4)[0]
        e_size = struct.unpack_from("<Q", data, p + 8)[0]
        # payload_size is u32, NOT u64. Older builds leave +0x14 zeroed so a
        # u64 read happens to work; sm_120 builds put a value there and the
        # u64 read silently becomes garbage.
        p_size = struct.unpack_from("<I", data, p + 0x10)[0]
        sm = struct.unpack_from("<I", data, p + 0x1C)[0]
        if kind not in (1, 2) or not (0x10 <= e_hdr <= 0x400):
            break
        # payload_size == 0 means "stored uncompressed" -- the payload is
        # e_size bytes of raw ELF starting at the entry header end. Otherwise
        # payload_size is the ZSTD frame length (e_size == align8(payload_size)).
        if p_size == 0:
            if e_size == 0 or e_size > (len(data) - p):
                break
        elif e_size != (p_size + 7) & ~7:
            break
        if p + e_hdr + e_size > end:
            break
        usize = struct.unpack_from("<Q", data, p + 0x38)[0]
        payload_off = p + e_hdr
        ent = FatbinEntry(idx, kind, ever, e_hdr, e_size,
                          f"sm_{sm}" if sm else f"?{idx}", p,
                          payload_off,
                          data[payload_off:payload_off + 4] == ELF_MAGIC)
        # kind 1 = PTX text, kind 2 = ELF (SASS)
        ent.is_ptx = (kind == 1)
        ent.payload_size = p_size if p_size else e_size
        ent.entry_size = e_size
        ent.sm = sm
        ent.uncompressed_size = e_size if p_size == 0 else usize
        ent.container_total = total
        entries.append(ent)
        idx += 1
        # next entry sits after this one's header + aligned payload
        nxt = p + e_hdr + e_size
        if nxt <= p:
            break
        p = nxt
    return entries


def scan_elf_cubins(data: bytes) -> list[int]:
    offs = []
    i = data.find(ELF_MAGIC)
    while i != -1:
        # ELF64 LE, e_machine == 190 (EM_CUDA)
        if len(data) >= i + 20 and data[i + 4] == 2 and data[i + 5] == 1:
            if struct.unpack_from("<H", data, i + 18)[0] == 190:
                offs.append(i)
        i = data.find(ELF_MAGIC, i + 1)
    return offs


# ---------------------------------------------------------------------------
# CUBIN / ELF
# ---------------------------------------------------------------------------

@dataclass
class ElfSection:
    name: str
    type: int
    flags: int
    addr: int
    offset: int
    size: int
    link: int
    entsize: int


@dataclass
class Cubin:
    offset: int
    size: int
    machine: int
    arch: str
    sections: list[ElfSection] = field(default_factory=list)
    symbols: list[tuple[str, int, int, int]] = field(default_factory=list)
    kernels: list[str] = field(default_factory=list)
    nv_info: dict = field(default_factory=dict)


# Keyed by the DECIMAL SM code read from e_flags bits 8..15 -- these are plain
# decimal (75, 89, 120), not hex-looking, so writing 0x80 here would be wrong.
CUBIN_ARCH = {
    50: "sm_50", 52: "sm_52", 53: "sm_53", 60: "sm_60", 61: "sm_61",
    62: "sm_62", 70: "sm_70", 72: "sm_72", 75: "sm_75", 80: "sm_80",
    86: "sm_86", 87: "sm_87", 89: "sm_89", 90: "sm_90", 100: "sm_100",
    101: "sm_101", 120: "sm_120", 121: "sm_121",
}


def parse_cubin(data: bytes, off: int) -> Cubin | None:
    """Minimal ELF64 parser targeted at NVIDIA cubins."""
    if data[off:off + 4] != ELF_MAGIC:
        return None
    (e_type, e_machine) = struct.unpack_from("<HH", data, off + 16)
    e_shoff = struct.unpack_from("<Q", data, off + 40)[0]
    e_shentsize = struct.unpack_from("<H", data, off + 58)[0]
    e_shnum = struct.unpack_from("<H", data, off + 60)[0]
    e_shstrndx = struct.unpack_from("<H", data, off + 62)[0]
    if e_shoff == 0 or e_shnum == 0:
        return None

    raw = []
    for i in range(e_shnum):
        b = off + e_shoff + i * e_shentsize
        (sh_name, sh_type, sh_flags, sh_addr, sh_offset, sh_size,
         sh_link, sh_info, sh_align, sh_entsize) = struct.unpack_from(
            "<IIQQQQIIQQ", data, b)
        raw.append((sh_name, sh_type, sh_flags, sh_addr, sh_offset, sh_size,
                    sh_link, sh_info, sh_align, sh_entsize))

    shstr_off = off + raw[e_shstrndx][4] if e_shstrndx < len(raw) else off

    def nm(x: int) -> str:
        s = shstr_off + x
        e = data.find(b"\x00", s)
        return data[s:e].decode("ascii", "replace")

    secs = [ElfSection(nm(r[0]), r[1], r[2], r[3], off + r[4], r[5], r[6], r[9])
            for r in raw]

    # total span of the cubin
    end = max((s.offset + s.size for s in secs), default=off + 64)
    # sm arch comes from the .nv.info / ELF flags; fall back to e_machine probe
    arch = ""
    for s in secs:
        if s.name == ".nv.info" or s.name.startswith(".nv.constant"):
            pass
    # NVIDIA cubins encode the SM target in e_flags bits 8..15 -- this must
    # agree with the sm field in the enclosing fatbin entry header.
    e_flags = struct.unpack_from("<I", data, off + 48)[0]
    sm = (e_flags >> 8) & 0xFF
    if sm == 0:
        sm = e_flags & 0xFF
    if sm:
        arch = CUBIN_ARCH.get(sm, f"sm_{sm}")

    cub = Cubin(offset=off, size=end - off, machine=e_machine, arch=arch,
                sections=secs)

    # symbols + kernel names
    for s in secs:
        if s.name in (".symtab",) and s.entsize == 24:
            strtab = secs[s.link] if s.link < len(secs) else None
            if not strtab:
                continue
            n = s.size // 24
            for i in range(n):
                b = s.offset + i * 24
                st_name, st_info, st_other, st_shndx, st_value, st_size = \
                    struct.unpack_from("<IBBHQQ", data, b)
                so = strtab.offset + st_name
                se = data.find(b"\x00", so)
                sym = data[so:se].decode("ascii", "replace")
                cub.symbols.append((sym, st_value, st_size, st_info & 0xF))
                if (st_info & 0xF) in (1, 2) and st_shndx != 0 and sym:
                    if not sym.startswith("$"):
                        cub.kernels.append(sym)

    cub.kernels = sorted(set(cub.kernels))
    return cub


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def to_json(obj) -> str:
    return json.dumps(obj, indent=2, default=_json_default)


def _json_default(o):
    if hasattr(o, "__dataclass_fields__"):
        d = asdict(o)
        d.pop("data", None)
        return d
    return str(o)
