// Copyright 2016, University of Colorado Boulder

/**
 * IO type for Font
 *
 * @author Andrew Adare (PhET Interactive Simulations)
 */
define( function( require ) {
  'use strict';

  // modules
  var ObjectIO = require( 'TANDEM/types/ObjectIO' );
  var phetioInherit = require( 'TANDEM/phetioInherit' );
  var scenery = require( 'SCENERY/scenery' );
  var validate = require( 'AXON/validate' );

  /**
   * @constructor
   * IO type for phet/scenery's Font class
   * @param {Font} font - An instance of a scenery.Font type
   * @param {string} phetioID - Full name of this font instance
   */
  function FontIO( font, phetioID ) {
    ObjectIO.call( this, font, phetioID );
  }

  phetioInherit( ObjectIO, 'FontIO', FontIO, {}, {

    validator: { valueType: scenery.Font },

    // Info from Font.js
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


    /**
     * Encodes a Font instance to a state.
     * Serialize this font's configuration to an options object
     * @param {Font} font
     * @returns {Object}
     * @override
     */
    toStateObject: function( font ) {
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
    },

    /**
     * Decodes a state into a Font.
     * Use stateObject as the Font constructor's options argument
     * @param {Object} stateObject
     * @returns {Font}
     * @override
     */
    fromStateObject: function( stateObject ) {
      return new scenery.Font( stateObject );
    }
  } );

  scenery.register( 'FontIO', FontIO );

  return FontIO;
} );
