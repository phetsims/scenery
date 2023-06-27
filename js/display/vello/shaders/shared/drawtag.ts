/* eslint-disable */

export default `struct DrawMonoid{path_ix:u32,
clip_ix:u32,
scene_offset:u32,
info_offset:u32,}const DRAWTAG_NOP=0u;const DRAWTAG_FILL_COLOR=0x44u;const DRAWTAG_FILL_LIN_GRADIENT=0x114u;const DRAWTAG_FILL_RAD_GRADIENT=0x29cu;const DRAWTAG_FILL_IMAGE=0x248u;const DRAWTAG_BEGIN_CLIP=0x9u;const DRAWTAG_END_CLIP=0x21u;fn draw_monoid_identity()->DrawMonoid{return DrawMonoid();}fn combine_draw_monoid(a:DrawMonoid,b:DrawMonoid)->DrawMonoid{var c:DrawMonoid;c.path_ix=a.path_ix+b.path_ix;c.clip_ix=a.clip_ix+b.clip_ix;c.scene_offset=a.scene_offset+b.scene_offset;c.info_offset=a.info_offset+b.info_offset;return c;}fn map_draw_tag(tag_word:u32)->DrawMonoid{var c:DrawMonoid;c.path_ix=u32(tag_word !=DRAWTAG_NOP);c.clip_ix=tag_word&1u;c.scene_offset=(tag_word>>2u)&0x07u;c.info_offset=(tag_word>>6u)&0x0fu;return c;}`
