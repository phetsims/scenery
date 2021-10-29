// Copyright 2020-2021, University of Colorado Boulder

import Property from '../../../axon/js/Property.js';
import merge from '../../../phet-core/js/merge.js';
import scenery from '../scenery.js';
import Color from './Color.js';

/**
 * Convenience type for creating Property.<Color>
 *
 * @author Sam Reid (PhET Interactive Simulations)
 */
class ColorProperty extends Property {

  /**
   * @param {ColorDef} color
   * @param {Object} [options]
   */
  constructor( color, options ) {

    // client cannot specify superclass options that are controlled by this type
    if ( options ) {
      assert && assert( !options.hasOwnProperty( 'valueType' ), 'ColorProperty sets valueType' );
      assert && assert( !options.hasOwnProperty( 'phetioType' ), 'ColorProperty sets phetioType' );
    }

    options = merge( {
      valueType: Color,
      phetioType: Property.PropertyIO( Color.ColorIO )
    }, options );
    super( color, options );
  }
}

scenery.register( 'ColorProperty', ColorProperty );
export default ColorProperty;