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
  
  //I cannot figure out why this import is required, but without it the sim crashes on startup.
  var Renderer = require( 'SCENERY/layers/Renderer' );
  
  var AccessibilityPeer = scenery.AccessibilityPeer = function AccessibilityPeer( instance, element, options ) {
    var peer = this;
    
    options = options || {};
    
    // TODO: if element is a DOM element, verify that no other accessibility peer is using it! (add a flag, and remove on disposal)
    this.element = ( typeof element === 'string' ) ? $( element )[0] : element;
    this.instance = instance;
    this.trail = instance.trail;
    
    this.element.setAttribute( 'tabindex', 0 );
    this.element.style.position = 'absolute';
    
    var scene = instance.getScene();
    this.clickListener = function PeerClickListener( event ) {
      if ( options.click ) { options.click( event ); }
    };
    this.focusListener = function PeerFocusListener( event ) {
      scene.focusPeer( peer );
    };
    this.blurListener = function PeerBlurListener( event ) {
      scene.blurPeer( peer );
    };
    this.element.addEventListener( 'click', this.clickListener );
    this.element.addEventListener( 'focus', this.focusListener );
    this.element.addEventListener( 'blur', this.blurListener );
  };
  
  AccessibilityPeer.prototype = {
    constructor: AccessibilityPeer,
    
    dispose: function() {
      this.element.removeEventListener( 'click', this.clickListener );
      this.element.removeEventListener( 'focus', this.focusListener );
      this.element.removeEventListener( 'blur', this.blurListener );
    },
    
    getGlobalBounds: function() {
      return this.trail.parentToGlobalBounds( this.trail.lastNode().getBounds() ).roundedOut();
    },
    
    syncBounds: function() {
      var globalBounds = this.getGlobalBounds();
      //TODO: add checks in here that will only set the values if changed
      this.element.style.left = globalBounds.x + 'px';
      this.element.style.top = globalBounds.y + 'px';
      this.element.style.width = globalBounds.width + 'px';
      this.element.style.height = globalBounds.height + 'px';
      // this.$element.css( 'left', globalBounds.x );
      // this.$element.css( 'top', globalBounds.y );
      // this.$element.width( globalBounds.width );
      // this.$element.height( globalBounds.height );
    }
  };
  
  return AccessibilityPeer;
} );
