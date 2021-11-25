// Copyright 2018-2021, University of Colorado Boulder

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
import { scenery, Color } from '../imports.js';

const ColorDef = {
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