// Copyright 2017-2022, University of Colorado Boulder

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
import { Display, scenery, Trail } from '../imports.js';

class Focus {

  // The trail to the focused Node.
  public readonly trail: Trail;

  // The Display containing the Trail to the focused Node.
  public readonly display: Display;

  public static readonly FocusIO = new IOType( 'FocusIO', {
    valueType: Focus,
    documentation: 'A IO Type for the instance in the simulation which currently has keyboard focus. FocusIO is ' +
                   'serialized into and Object with key `focusedPhetioElement` that is a list of PhET-iO elements, ' +
                   'from parent-most to child-most corresponding to the PhET-iO element that was instrumented.',
    toStateObject: ( focus: Focus ) => {
      const phetioIDs: string[] = [];
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

  public constructor( display: Display, trail: Trail ) {
    this.display = display;
    this.trail = trail;
  }
}

scenery.register( 'Focus', Focus );
export default Focus;