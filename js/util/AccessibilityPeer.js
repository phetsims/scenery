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

//    options.interactive = true;
    this.origin = origin;
    this.domText = domText;

    this.$element = $( domText );
    this.$element.attr( 'tabindex', 0 );
    this.$element.css( 'position', 'absolute' );
    this.markForDeletion = false;
    // will set the element after initializing
//    scenery.DOM.call( this, $( domText ), options );
    if ( options.click ) {
      this.$element.click( options.click );
    }
  };
  scenery.AccessibilityPeer.prototype = {
    syncBounds: function() {
      var globalBounds = this.origin.globalBounds;
      //TODO: add checks in here that will only set the values if changed
      this.$element.css( 'left', globalBounds.x );
      this.$element.css( 'top', globalBounds.y );
      this.$element.width( globalBounds.width );
      this.$element.height( globalBounds.height );
    }
  };

  return scenery.AccessibilityPeer;
} );
