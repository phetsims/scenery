// Copyright 2017-2020, University of Colorado Boulder

/**
 * IO Type for Focus
 *
 * @author Sam Reid (PhET Interactive Simulations)
 * @author Jesse Greenberg (PhET Interactive Simulations)
 */

import IOType from '../../../tandem/js/types/IOType.js';
import scenery from '../scenery.js';
import Focus from './Focus.js';

const FocusIO = new IOType( 'FocusIO', {
  valueType: Focus,
  documentation: 'A IO Type for the instance in the simulation which currently has keyboard focus. FocusIO is ' +
                 'serialized into and Object with key `focusedPhetioElement` that is a list of PhET-iO elements, ' +
                 'from parent-most to child-most cooresponding to the PhET-iO element that was instrumented.',
  toStateObject: focus => {
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
} );

scenery.register( 'FocusIO', FocusIO );
export default FocusIO;