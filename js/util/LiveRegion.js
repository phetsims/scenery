// Copyright 2002-2014, University of Colorado

/**
 * Live region is used with accessibility to read out changes in model state.
 * Should conform to the Axon property interface to make it easy to interchange.
 *
 * @author Sam Reid
 */

define( function( require ) {
  'use strict';

  var scenery = require( 'SCENERY/scenery' );

  var LiveRegion = scenery.LiveRegion = function LiveRegion( instance, property, options ) {
    var liveRegion = this;
    this.property = property;
    options = options || {};

    //Defaulting to 0 would mean using the document order, which can easily be incorrect for a PhET simulation.
    //For any of the nodes to use a nonzero tabindex, they must all use a nonzero tabindex, see #40
    options.tabIndex = options.tabIndex || 1;

    // TODO: if element is a DOM element, verify that no other accessibility liveRegion is using it! (add a flag, and remove on disposal)
//    this.element = '<div role="region" id="bird-info" aria-live="polite">';
    this.element = document.createElement( 'div' );
    this.element.setAttribute( 'aria-live', 'polite' );
    this.element.setAttribute( 'role', 'region' );
    this.textNode = document.createTextNode( '' );
    this.element.appendChild( this.textNode );

    //Just setting the text causes NVDA to read deltas, you have to replace the node to have it read the text
    this.listener = function( newText ) {
      liveRegion.element.removeChild( liveRegion.textNode );
      liveRegion.textNode = document.createTextNode( newText );
      liveRegion.element.appendChild( liveRegion.textNode );
    };
    property.link( this.listener );
  };

  LiveRegion.prototype = {
    constructor: scenery.LiveRegion,
    dispose: function() { this.property.unlink( this.listener ); }
  };

  return LiveRegion;
} );
