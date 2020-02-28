// Copyright 2017-2020, University of Colorado Boulder

/**
 * IO type for Color
 *
 * @author Sam Reid (PhET Interactive Simulations)
 * @author Andrew Adare (PhET Interactive Simulations)
 */

import validate from '../../../axon/js/validate.js';
import ObjectIO from '../../../tandem/js/types/ObjectIO.js';
import scenery from '../scenery.js';
import Color from './Color.js';

class ColorIO extends ObjectIO {

  /**
   * Encodes a Color into a state object.
   * @param {Color} color
   * @returns {Object}
   * @override
   */
  static toStateObject( color ) {
    validate( color, this.validator );
    return color.toStateObject();
  }

  /**
   * Decodes a state into a Color.
   * Use stateObject as the Font constructor's options argument
   * @param {Object} stateObject
   * @returns {Color}
   * @override
   */
  static fromStateObject( stateObject ) {
    return new Color( stateObject.r, stateObject.g, stateObject.b, stateObject.a );
  }
}

ColorIO.documentation = 'A color, with rgba';
ColorIO.validator = { valueType: Color };
ColorIO.typeName = 'ColorIO';
ObjectIO.validateSubtype( ColorIO );

scenery.register( 'ColorIO', ColorIO );
export default ColorIO;