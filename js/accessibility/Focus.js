// Copyright 2017, University of Colorado Boulder

/**
 * A scenery-internal type for tracking what currently has focus in Display.  This is the value for
 * the static Display.focusProperty.  If a focused node is shared between two Displays, only one
 * instance will have focus.
 *
 * @author Jesse Greenberg
 */
define( function( require ) {
  'use strict';

  // modules
  var inherit = require( 'PHET_CORE/inherit' );
  var scenery = require( 'SCENERY/scenery' );

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

  return inherit( Object, Focus );
} );
