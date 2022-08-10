// Copyright 2020-2022, University of Colorado Boulder

import Property, { PropertyOptions } from '../../../axon/js/Property.js';
import optionize, { EmptySelfOptions } from '../../../phet-core/js/optionize.js';
import { Color, scenery } from '../imports.js';

/**
 * Convenience type for creating Property.<Color>
 *
 * @author Sam Reid (PhET Interactive Simulations)
 */
export default class ColorProperty extends Property<Color> {
  public constructor( color: Color, providedOptions?: PropertyOptions<Color> ) {

    // client cannot specify superclass options that are controlled by this type
    if ( providedOptions ) {
      assert && assert( !providedOptions.hasOwnProperty( 'valueType' ), 'ColorProperty sets valueType' );
      assert && assert( !providedOptions.hasOwnProperty( 'phetioType' ), 'ColorProperty sets phetioType' );
    }

    const options = optionize<PropertyOptions<Color>, EmptySelfOptions, PropertyOptions<Color>>()( {
      valueType: Color,
      phetioValueType: Color.ColorIO
    }, providedOptions );
    super( color, options );
  }
}

scenery.register( 'ColorProperty', ColorProperty );
