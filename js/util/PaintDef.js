// Copyright 2018-2022, University of Colorado Boulder

/**
 * "definition" type for generalized paints (anything that can be passed in as a fill or stroke to a Path)
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import ReadOnlyProperty from '../../../axon/js/ReadOnlyProperty.js';
import { Color, Gradient, Paint, scenery } from '../imports.js';

const PaintDef = {
  /**
   * Returns whether the parameter is considered to be a PaintDef.
   * @public
   *
   * @param {*} paint
   * @returns {boolean}
   */
  isPaintDef( paint ) {
    // NOTE: Property.<Paint> is not supported. PaintObserver would technically need to listen to 3 different levels if
    // we add that (or could be recursive if we allow Property.<paintDef>. Notably, the Property value could change,
    // Color Properties in the Gradient could change, AND the Colors themselves specified in those Properties could
    // change. So it would be more code and more memory usage in general to support it.
    // See https://github.com/phetsims/scenery-phet/issues/651
    return paint === null ||
           typeof paint === 'string' ||
           paint instanceof Color ||
           paint instanceof Paint ||
           ( paint instanceof ReadOnlyProperty && (
             paint.value === null ||
             typeof paint.value === 'string' ||
             paint.value instanceof Color
           ) );
  },

  /**
   * Takes a snapshot of the given paint, returning the current color where possible.
   * Unlike Color.toColor() this method makes a defensive copy for Color values.
   * @public
   *
   * @param {PaintDef} paint
   * @returns {Color}
   */
  toColor( paint ) {
    if ( typeof paint === 'string' ) {
      return new Color( paint );
    }
    if ( paint instanceof Color ) {
      return paint.copy();
    }
    if ( paint instanceof ReadOnlyProperty ) {
      return PaintDef.toColor( paint.value );
    }
    if ( paint instanceof Gradient ) {
      // Average the stops
      let color = Color.TRANSPARENT;
      const quantity = 0;
      paint.stops.forEach( stop => {
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