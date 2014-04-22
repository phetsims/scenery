// Copyright 2002-2014, University of Colorado

/**
 * Accessibility peer, which is added to the dom for focus and keyboard navigation.
 *
 * @author Sam Reid
 */

define( function( require ) {
  'use strict';

  var scenery = require( 'SCENERY/scenery' );
  
  var AccessibilityPeer = scenery.AccessibilityPeer = function AccessibilityPeer( instance, element, options ) {
    var peer = this;
    
    options = options || {};

    //Defaulting to 0 would mean using the document order, which can easily be incorrect for a PhET simulation.
    //For any of the nodes to use a nonzero tabindex, they must all use a nonzero tabindex, see #40
    options.tabIndex = options.tabIndex || 1;
    
    // TODO: if element is a DOM element, verify that no other accessibility peer is using it! (add a flag, and remove on disposal)
    this.element = ( typeof element === 'string' ) ? $( element )[0] : element;

    if ( options.label ) {
      this.peerElement = document.createElement( 'div' );
      this.element.id = 'peer-' + instance.trail.getUniqueId();
      var label = document.createElement('label');
      label.appendChild(document.createTextNode(options.label));
      label.setAttribute('for',this.element.id);
      this.peerElement.appendChild(label);
      this.peerElement.appendChild(this.element);
    } else{
      this.peerElement = this.element;
    }

    this.instance = instance;
    this.trail = instance.trail;
    
    this.element.setAttribute( 'tabindex', options.tabIndex );
    this.element.style.position = 'absolute';
    
    // TODO: batch these also if the Scene is batching events
    var scene = instance.getScene();
    this.clickListener = function PeerClickListener( event ) {
      sceneryAccessibilityLog && sceneryAccessibilityLog( 'peer click on ' + instance.toString() + ': ' + instance.getNode().constructor.name );
      if ( options.click ) { options.click( event ); }
    };
    this.focusListener = function PeerFocusListener( event ) {
      sceneryAccessibilityLog && sceneryAccessibilityLog( 'peer focused: ' + instance.toString() + ': ' + instance.getNode().constructor.name );
      scene.focusPeer( peer );
    };
    this.blurListener = function PeerBlurListener( event ) {
      sceneryAccessibilityLog && sceneryAccessibilityLog( 'peer blurred: ' + instance.toString() + ': ' + instance.getNode().constructor.name );
      scene.blurPeer( peer );
    };
    this.element.addEventListener( 'click', this.clickListener );
    this.element.addEventListener( 'focus', this.focusListener );
    this.element.addEventListener( 'blur', this.blurListener );

    this.keepPeerBoundsInSync = true;
    if ( this.keepPeerBoundsInSync ) {
      this.boundsSyncListener = this.syncBounds.bind( this );

      instance.getNode().addEventListener( 'bounds', this.boundsSyncListener );
      this.syncBounds();

      //When the scene resizes, update the peer bounds
      instance.getScene().addEventListener( 'resize', this.boundsSyncListener );

      //Initial layout
      window.setTimeout( this.syncBounds.bind( this ), 30 );
    }
  };

  AccessibilityPeer.prototype = {
    constructor: AccessibilityPeer,
    
    dispose: function() {
      this.element.removeEventListener( 'click', this.clickListener );
      this.element.removeEventListener( 'focus', this.focusListener );
      this.element.removeEventListener( 'blur', this.blurListener );
      
      // don't leak memory
      if ( this.keepPeerBoundsInSync ) {
        this.instance.getNode().removeEventListener( 'bounds', this.boundsSyncListener );
        this.instance.getScene().removeEventListener( 'resize', this.boundsSyncListener );
      }
    },
    
    getGlobalBounds: function() {
      return this.trail.parentToGlobalBounds( this.trail.lastNode().getBounds() ).roundedOut();
    },
    
    syncBounds: function() {
      var globalBounds = this.getGlobalBounds();
      this.element.style.left = globalBounds.x + 'px';
      this.element.style.top = globalBounds.y + 'px';
      this.element.style.width = globalBounds.width + 'px';
      this.element.style.height = globalBounds.height + 'px';
    }
  };
  
  return AccessibilityPeer;
} );
