// Copyright 2017-2020, University of Colorado Boulder

/**
 * IO Type for Color
 *
 * @author Sam Reid (PhET Interactive Simulations)
 * @author Andrew Adare (PhET Interactive Simulations)
 */

import IOType from '../../../tandem/js/types/IOType.js';
import scenery from '../scenery.js';
import Color from './Color.js';

const ColorIO = new IOType( 'ColorIO', {
  valueType: Color,
  documentation: 'A color, with rgba',
  toStateObject: color => color.toStateObject(),
  fromStateObject: stateObject => new Color( stateObject.r, stateObject.g, stateObject.b, stateObject.a )
} );

scenery.register( 'ColorIO', ColorIO );
export default ColorIO;