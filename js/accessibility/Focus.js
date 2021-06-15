// Copyright 2017-2021, University of Colorado Boulder

/**
 * A scenery-internal type for tracking what currently has focus in Display. If a focused Node is shared between
 * two Displays, it is possible that only one Node may have focus between the two displays. This is especially
 * true for DOM focus since only one element can have DOM focus at a time.
 *
 * @author Jesse Greenberg
 */

import ArrayIO from '../../../tandem/js/types/ArrayIO.js';
import IOType from '../../../tandem/js/types/IOType.js';
import StringIO from '../../../tandem/js/types/StringIO.js';
import scenery from '../scenery.js';

class Focus {

  /**
   * @param {Display} display - Display containing the focused node
   * @param {Trail} trail - Trail to the focused node
   */
  constructor( display, trail ) {

    // @public (read-only)
    this.display = display;
    this.trail = trail;
  }
}

Focus.FocusIO = new IOType( 'FocusIO', {
  valueType: Focus,
  documentation: 'A IO Type for the instance in the simulation which currently has keyboard focus. FocusIO is ' +
                 'serialized into and Object with key `focusedPhetioElement` that is a list of PhET-iO elements, ' +
                 'from parent-most to child-most corresponding to the PhET-iO element that was instrumented.',
  toStateObject: focus => {
    const phetioIDs = [];
    focus.trail.nodes.forEach( ( node, i ) => {

      // If the node was PhET-iO instrumented, include its phetioID instead of its index (because phetioID is more stable)
      if ( node.isPhetioInstrumented() ) {
        phetioIDs.push( node.tandem.phetioID );
      }
    } );

    return {
      focusedPhetioElement: phetioIDs
    };
  },
  stateSchema: {
    focusedPhetioElement: ArrayIO( StringIO )
  }
} );

scenery.register( 'Focus', Focus );
export default Focus;