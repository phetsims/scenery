// Copyright 2017-2020, University of Colorado Boulder

/**
 * IO Type for Font
 *
 * @author Andrew Adare (PhET Interactive Simulations)
 */

import IOType from '../../../tandem/js/types/IOType.js';
import scenery from '../scenery.js';

const FontIO = new IOType( 'FontIO', {
  valueType: scenery.Font,
  documentation: 'Font handling for text drawing. Options:' +
                 '<ul>' +
                 '<li><strong>style:</strong> normal      &mdash; normal | italic | oblique </li>' +
                 '<li><strong>variant:</strong> normal    &mdash; normal | small-caps </li>' +
                 '<li><strong>weight:</strong> normal     &mdash; normal | bold | bolder | lighter | 100 | 200 | 300 | 400 | 500 | 600 | 700 | 800 | 900 </li>' +
                 '<li><strong>stretch:</strong> normal    &mdash; normal | ultra-condensed | extra-condensed | condensed | semi-condensed | semi-expanded | expanded | extra-expanded | ultra-expanded </li>' +
                 '<li><strong>size:</strong> 10px         &mdash; absolute-size | relative-size | length | percentage -- unitless number interpreted as px. absolute suffixes: cm, mm, in, pt, pc, px. relative suffixes: em, ex, ch, rem, vw, vh, vmin, vmax. </li>' +
                 '<li><strong>lineHeight:</strong> normal &mdash; normal | number | length | percentage -- NOTE: Canvas spec forces line-height to normal </li>' +
                 '<li><strong>family:</strong> sans-serif &mdash; comma-separated list of families, including generic families (serif, sans-serif, cursive, fantasy, monospace). ideally escape with double-quotes</li>' +
                 '</ul>',
  toStateObject( font ) {
    return {
      style: font.getStyle(),
      variant: font.getVariant(),
      weight: font.getWeight(),
      stretch: font.getStretch(),
      size: font.getSize(),
      lineHeight: font.getLineHeight(),
      family: font.getFamily()
    };
  },

  fromStateObject( stateObject ) {
    return new scenery.Font( stateObject );
  }
} );

scenery.register( 'FontIO', FontIO );
export default FontIO;