// Copyright 2018-2020, University of Colorado Boulder

/**
 * "definition" type for generalized paints (anything that can be passed in as a fill or stroke to a Path)
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import Property from '../../../axon/js/Property.js';
import scenery from '../scenery.js';
import Color from './Color.js';
import Paint from './Paint.js';

var PaintDef = {
  /**
   * Returns whether the parameter is considered to be a PaintDef.
   * @public
   *
   * @param {*} paint
   * @returns {boolean}
   */
  isPaintDef: function( paint ) {
    return paint === null ||
           typeof paint === 'string' ||
           paint instanceof Color ||
           paint instanceof Paint ||
           ( paint instanceof Property && (
             paint.value === null ||
             typeof paint.value === 'string' ||
             paint.value instanceof Color
           ) );
  },

  /**
   * Takes a snapshot of the given paint, returning the current color where possible.
   * @public
   *
   * @param {PaintDef} paint
   * @returns {Color}
   */
  toColor: function( paint ) {
    if ( typeof paint === 'string' ) {
      return new Color( paint );
    }
    if ( paint instanceof Color ) {
      return paint.copy();
    }
    if ( paint instanceof Property ) {
      return PaintDef.toColor( paint.value );
    }
    if ( paint instanceof scenery.Gradient ) {
      // Average the stops
      let color = Color.TRANSPARENT;
      const quantity = 0;
      paint.stops.forEach( function( stop ) {
        color = color.blend( PaintDef.toColor( stop.color ), 1 / ( quantity + 1 ) );
      } );
      return color;
    }

    // Fall-through value (null, Pattern, etc.)
    return Color.TRANSPARENT;
  }
};

scenery.register( 'PaintDef', PaintDef );

export default PaintDef;