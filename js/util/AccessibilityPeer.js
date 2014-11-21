// Copyright 2002-2014, University of Colorado Boulder

/**
 * Accessibility peer, which is added to the dom for focus and keyboard navigation.
 *
 * @author Sam Reid
 */

define( function( require ) {
  'use strict';

  var scenery = require( 'SCENERY/scenery' );
  var inherit = require( 'PHET_CORE/inherit' );

  var AccessibilityPeer = scenery.AccessibilityPeer = function AccessibilityPeer( instance, element, options ) {
    var peer = this;

    this.options = options = options || {};

    this.id = 'peer-' + instance.trail.getUniqueId();

    //Defaulting to 0 would mean using the document order, which can easily be incorrect for a PhET simulation.
    //For any of the nodes to use a nonzero tabindex, they must all use a nonzero tabindex, see #40
    options.tabIndex = options.tabIndex || 1;

    // TODO: if element is a DOM element, verify that no other accessibility peer is using it! (add a flag, and remove on disposal)
    this.element = ( typeof element === 'string' ) ? $( element )[0] : element;

    if ( options.label ) {
      var labelId = this.id + '-label';
      this.element.id = labelId;
      this.peerElement = document.createElement( 'div' );
      var label = document.createElement( 'label' );
      label.appendChild( document.createTextNode( options.label ) );
      label.setAttribute( 'for', labelId );
      this.peerElement.appendChild( label );
      this.peerElement.appendChild( this.element );
    }
    else {
      this.peerElement = this.element;
    }
    this.peerElement.id = this.id;

    this.visible = true;

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

    this.element.addEventListener( 'focus', this.focusListener );
    this.element.addEventListener( 'blur', this.blurListener );

    // Handle key presses for buttons as well as <div> or <span> with role="button"
    // See https://developer.mozilla.org/en-US/docs/Web/Accessibility/ARIA/ARIA_Techniques/Using_the_button_role
    this.element.addEventListener( 'keyup', function handleBtnKeyUp( event ) {
      event = event || window.event;
      if ( event.keyCode === 32 || event.keyCode === 13 ) { // check for Space key or enter key (13)
        peer.clickListener();
      }
    } );
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

  return inherit( Object, AccessibilityPeer, {
    updateVisibility: function() {
      var newVisibility = this.trail.isVisible();
      if ( newVisibility !== this.visible ) {
        this.visible = newVisibility;
        if ( newVisibility ) {
          this.peerElement.style.display = 'inherit';
        }
        else {
          this.peerElement.style.display = 'none';
        }
      }
    },

    onAdded: function( peer ) {
      this.options.onAdded && this.options.onAdded( peer );
    },

    onRemoved: function( peer ) {
      this.options.onRemoved && this.options.onRemoved( peer );
    },

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
  } );
} );
