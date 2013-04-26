// Copyright 2002-2012, University of Colorado

/**
 * Accessibility peer, which is added to the dom for focus and keyboard navigation.
 *
 * @author Sam Reid
 */

define( function( require ) {
  "use strict";

  var inherit = require( 'PHET_CORE/inherit' );

  var scenery = require( 'SCENERY/scenery' );

  var Node = require( 'SCENERY/nodes/Node' ); // DOM inherits from Node

  //I cannot figure out why this import is required, but without it the sim crashes on startup.
  var Renderer = require( 'SCENERY/layers/Renderer' );

  scenery.AccessibilityPeer = function AccessibilityPeer( origin, domText, options ) {
    options = options || {};

    this.origin = origin;
    // will set the element after initializing
    scenery.DOM.call( this, $( domText ), options );
    if ( options.click ) {
      this._$element.click( options.click );
    }
  };

  inherit( scenery.AccessibilityPeer, scenery.DOM );

  return scenery.AccessibilityPeer;
} );