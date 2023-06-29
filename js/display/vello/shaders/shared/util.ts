/* eslint-disable */

export default `












fn read_draw_tag_from_scene(ix: u32) -> u32 {
    let tag_ix = config.drawtag_base + ix;
    var tag_word: u32;
    if tag_ix < config.drawtag_base + config.n_drawobj {
        tag_word = scene[tag_ix];
    } else {
        tag_word = DRAWTAG_NOP;
    }
    return tag_word;
}
`
