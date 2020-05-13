// Copyright 2017-2020, University of Colorado Boulder

/**
 * IO type for Focus
 *
 * @author Sam Reid (PhET Interactive Simulations)
 * @author Jesse Greenberg (PhET Interactive Simulations)
 */

import validate from '../../../axon/js/validate.js';
import ObjectIO from '../../../tandem/js/types/ObjectIO.js';
import scenery from '../scenery.js';
import Focus from './Focus.js';

class FocusIO extends ObjectIO {

  /**
   * Convert the focus region to a plain JS object for serialization.
   * @param {Object} focus - the focus region which has {display,trail}
   * @returns {Object} - the serialized object
   * @public
   * @override
   */
  static toStateObject( focus ) {
    validate( focus, this.validator );
    const phetioIDs = [];
    focus.trail.nodes.forEach( function( node, i ) {

      // If the node was PhET-iO instrumented, include its phetioID instead of its index (because phetioID is more stable)
      if ( node.isPhetioInstrumented() ) {
        phetioIDs.push( node.tandem.phetioID );
      }
    } );

    return {
      focusedPhetioElement: phetioIDs
    };
  }
}

FocusIO.validator = { valueType: Focus };
FocusIO.documentation = 'A IO type for the instance in the simulation which currently has keyboard focus. FocusIO is ' +
                        'serialized into and Object with key `focusedPhetioElement` that is a list of PhET-iO elements, ' +
                        'from parent-most to child-most cooresponding to the PhET-iO element that was instrumented.';
FocusIO.typeName = 'FocusIO';
ObjectIO.validateSubtype( FocusIO );

scenery.register( 'FocusIO', FocusIO );
export default FocusIO;