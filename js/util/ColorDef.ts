// Copyright 2018-2025, University of Colorado Boulder

/**
 * "definition" type for generalized color paints (anything that can be given to a fill/stroke that represents just a
 * solid color). Does NOT include any type of gradient or pattern.
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import Property from '../../../axon/js/Property.js';
import { isTReadOnlyProperty } from '../../../axon/js/TReadOnlyProperty.js';
import IntentionalAny from '../../../phet-core/js/types/IntentionalAny.js';
import IOType, { AnyIOType } from '../../../tandem/js/types/IOType.js';
import NullableIO from '../../../tandem/js/types/NullableIO.js';
import OrIO from '../../../tandem/js/types/OrIO.js';
import ReferenceIO from '../../../tandem/js/types/ReferenceIO.js';
import StringIO from '../../../tandem/js/types/StringIO.js';
import Color from '../util/Color.js';
import scenery from '../scenery.js';
import TColor from '../util/TColor.js';

const ColorDef = {
  /**
   * Returns whether the parameter is considered to be a ColorDef.
   */
  isColorDef( color: unknown ): color is TColor {
    return color === null ||
           typeof color === 'string' ||
           color instanceof Color ||
           ( ( isTReadOnlyProperty( color ) ) && (
             color.value === null ||
             typeof color.value === 'string' ||
             color.value instanceof Color
           ) );
  },

  scenerySerialize( color: TColor ): string {
    if ( color === null ) {
      return 'null';
    }
    else if ( color instanceof Color ) {
      return `'${color.toCSS()}'`;
    }
    else if ( typeof color === 'string' ) {
      return `'${color}'`;
    }
    else {
      // Property fallback
      return ColorDef.scenerySerialize( color.value );
    }
  },

  // phet-io IOType for serialization and documentation
  ColorDefIO: null as unknown as AnyIOType // Defined below, typed here
};

ColorDef.ColorDefIO = new IOType<IntentionalAny, IntentionalAny>( 'ColorDefIO', {
  isValidValue: ColorDef.isColorDef,
  supertype: NullableIO( OrIO( [ StringIO, Color.ColorIO, ReferenceIO( Property.PropertyIO( NullableIO( OrIO( [ StringIO, Color.ColorIO ] ) ) ) ) ] ) )
} );

scenery.register( 'ColorDef', ColorDef );

export default ColorDef;