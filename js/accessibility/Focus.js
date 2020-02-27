// Copyright 2017-2019, University of Colorado Boulder

/**
 * A scenery-internal type for tracking what currently has focus in Display.  This is the value for
 * the static Display.focusProperty.  If a focused node is shared between two Displays, only one
 * instance will have focus.
 *
 * @author Jesse Greenberg
 */

import inherit from '../../../phet-core/js/inherit.js';
import scenery from '../scenery.js';

/**
 * Constructor.
 * @param {Display} display - Display containing the focused node
 * @param {Trail} trail - Trail to the focused node
 */
function Focus( display, trail ) {

  // @public (read-only)
  this.display = display;
  this.trail = trail;
}

scenery.register( 'Focus', Focus );

inherit( Object, Focus );
export default Focus;