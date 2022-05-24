// Copyright 2020-2022, University of Colorado Boulder

import Property, { PropertyOptions } from '../../../axon/js/Property.js';
import merge from '../../../phet-core/js/merge.js';
import { scenery, Color } from '../imports.js';

/**
 * Convenience type for creating Property.<Color>
 *
 * @author Sam Reid (PhET Interactive Simulations)
 */
export default class ColorProperty extends Property<Color> {
  constructor( color: Color, options?: PropertyOptions<Color> ) {

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
