// Copyright 2017-2019, University of Colorado Boulder

/**
 * IO type for Font
 *
 * @author Andrew Adare (PhET Interactive Simulations)
 */

import validate from '../../../axon/js/validate.js';
import ObjectIO from '../../../tandem/js/types/ObjectIO.js';
import scenery from '../scenery.js';

class FontIO extends ObjectIO {
  /**
   * Encodes a Font instance to a state.
   * Serialize this font's configuration to an options object
   * @param {Font} font
   * @returns {Object}
   * @override
   */
  static toStateObject( font ) {
    validate( font, this.validator );
    return {
      style: font.getStyle(),
      variant: font.getVariant(),
      weight: font.getWeight(),
      stretch: font.getStretch(),
      size: font.getSize(),
      lineHeight: font.getLineHeight(),
      family: font.getFamily()
    };
  }

  /**
   * Decodes a state into a Font.
   * Use stateObject as the Font constructor's options argument
   * @param {Object} stateObject
   * @returns {Font}
   * @override
   */
  static fromStateObject( stateObject ) {
    return new scenery.Font( stateObject );
  }
}

FontIO.validator = { valueType: scenery.Font };

// Info from Font.js
FontIO.documentation = 'Font handling for text drawing. Options:' +
                       '<ul>' +
                       '<li><strong>style:</strong> normal      &mdash; normal | italic | oblique </li>' +
                       '<li><strong>variant:</strong> normal    &mdash; normal | small-caps </li>' +
                       '<li><strong>weight:</strong> normal     &mdash; normal | bold | bolder | lighter | 100 | 200 | 300 | 400 | 500 | 600 | 700 | 800 | 900 </li>' +
                       '<li><strong>stretch:</strong> normal    &mdash; normal | ultra-condensed | extra-condensed | condensed | semi-condensed | semi-expanded | expanded | extra-expanded | ultra-expanded </li>' +
                       '<li><strong>size:</strong> 10px         &mdash; absolute-size | relative-size | length | percentage -- unitless number interpreted as px. absolute suffixes: cm, mm, in, pt, pc, px. relative suffixes: em, ex, ch, rem, vw, vh, vmin, vmax. </li>' +
                       '<li><strong>lineHeight:</strong> normal &mdash; normal | number | length | percentage -- NOTE: Canvas spec forces line-height to normal </li>' +
                       '<li><strong>family:</strong> sans-serif &mdash; comma-separated list of families, including generic families (serif, sans-serif, cursive, fantasy, monospace). ideally escape with double-quotes</li>' +
                       '</ul>';

FontIO.typeName = 'FontIO';
ObjectIO.validateSubtype( FontIO );

scenery.register( 'FontIO', FontIO );
export default FontIO;