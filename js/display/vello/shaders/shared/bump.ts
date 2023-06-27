/* eslint-disable */

export default `const STAGE_BINNING:u32=0x1u;const STAGE_TILE_ALLOC:u32=0x2u;const STAGE_PATH_COARSE:u32=0x4u;const STAGE_COARSE:u32=0x8u;struct BumpAllocators{failed:atomic<u32>,binning:atomic<u32>,ptcl:atomic<u32>,tile:atomic<u32>,segments:atomic<u32>,blend:atomic<u32>,}`
