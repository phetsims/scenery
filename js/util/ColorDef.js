// Copyright 2018-2020, University of Colorado Boulder

/**
 * "definition" type for generalized color paints (anything that can be given to a fill/stroke that represents just a
 * solid color). Does NOT include any type of gradient or pattern.
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import Property from '../../../axon/js/Property.js';
import IOType from '../../../tandem/js/types/IOType.js';
import NullableIO from '../../../tandem/js/types/NullableIO.js';
import OrIO from '../../../tandem/js/types/OrIO.js';
import ReferenceIO from '../../../tandem/js/types/ReferenceIO.js';
import StringIO from '../../../tandem/js/types/StringIO.js';
import scenery from '../scenery.js';
import Color from './Color.js';

const ColorDef = {

  /**
   * Link a listener to the given colorDef instance.  For a non-Property, this just calls back the listener with the value.
   * For a Property, it is linked and called back with the value.
   * @param {ColorDef} color
   * @param {function} listener
   * @public
   * @static
   */
  link( color, listener ) {
    assert && assert( ColorDef.isColorDef( color ), 'must be a colorDef' );
    if ( color instanceof Property ) {
      color.link( listener );
    }
    else {
      listener( color );
    }
  },

  /**
   * Unlink a listener from a colorDef instance.  For a non-Property, this is a no-op. For a Property, it is linked and
   * called back with the value.
   * @param {ColorDef} color
   * @param {function} listener
   * @public
   * @static
   */
  unlink( color, listener ) {
    assert && assert( ColorDef.isColorDef( color ), 'must be a colorDef' );
    if ( color instanceof Property ) {
      color.unlink( listener );
    }
  },

  // @returns {string|null}
  toCSS( color ) {
    assert && assert( ColorDef.isColorDef( color ), 'must be a colorDef' );
    return color === null ? null :
           typeof color === 'string' ? color :
           color instanceof Color ? color.toCSS() :
           ColorDef.toCSS( color.value );
  },

  /**
   * Returns whether the parameter is considered to be a ColorDef.
   * @public
   *
   * @param {*} color
   * @returns {boolean}
   */
  isColorDef( color ) {
    return color === null ||
           typeof color === 'string' ||
           color instanceof Color ||
           ( color instanceof Property && (
             color.value === null ||
             typeof color.value === 'string' ||
             color.value instanceof Color
           ) );
  }
};

// @public - phet-io IOType for serialization and documentation
ColorDef.ColorDefIO = new IOType( 'ColorDefIO', {
  isValidValue: ColorDef.isColorDef,
  supertype: NullableIO( OrIO( [ StringIO, Color.ColorIO, ReferenceIO( Property.PropertyIO( NullableIO( OrIO( [ StringIO, Color.ColorIO ] ) ) ) ) ] ) )
} );

scenery.register( 'ColorDef', ColorDef );

export default ColorDef;