// Copyright 2015-2016, University of Colorado Boulder

/**
 * An accessible peer controls the appearance of an accessible Node's instance in the parallel DOM.
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 * @author Jesse Greenberg
 */

define( function( require ) {
  'use strict';

  var inherit = require( 'PHET_CORE/inherit' );
  var Poolable = require( 'PHET_CORE/Poolable' );
  var Events = require( 'AXON/Events' );
  var scenery = require( 'SCENERY/scenery' );
  var Focus = require( 'SCENERY/accessibility/Focus' );
  // so RequireJS doesn't complain about circular dependency
  // var Display = require( 'SCENERY/display/Display' );

  var globalId = 1;

  /**
   * Constructor.
   *
   * @param  {AccessibleInstance} accessibleInstance
   * @param  {HTMLElement} domElement - The main DOM element used for this peer.
   * @param  {Object} options
   * @constructor
   */
  function AccessiblePeer( accessibleInstance, domElement, options ) {
    this.initializeAccessiblePeer( accessibleInstance, domElement, options );
  }

  scenery.register( 'AccessiblePeer', AccessiblePeer );

  inherit( Events, AccessiblePeer, {

    /**
     * @param {AccessibleInstance} accessibleInstance
     * @param {HTMLElement} domElement - The main DOM element used for this peer.
     * @param {Object} [options]
     * @returns {AccessiblePeer} - Returns 'this' reference, for chaining
     */
    initializeAccessiblePeer: function( accessibleInstance, domElement, options ) {
      var self = this;

      options = _.extend( {
        parentContainerElement: null, // a parent container for this peer and potential siblings
        childContainerElement: null, // an child container element where nested elements can be placed
        labelElement: null, // the element containing this node's label content 
        descriptionElement: null // the element that will contain this node's description content
      }, options );

      Events.call( this ); // TODO: is Events worth mixing in by default? Will we need to listen to events?

      assert && assert( !this.id || this.disposed, 'If we previously existed, we need to have been disposed' );

      // unique ID
      this.id = this.id || globalId++;

      // @public
      this.accessibleInstance = accessibleInstance;
      this.display = accessibleInstance.display;
      this.trail = accessibleInstance.trail;

      // @public, the DOM elements associated with this peer
      this.domElement = domElement;
      this.labelElement = options.labelElement;
      this.descriptionElement = options.descriptionElement;

      // @private - descendent of domElement that can be used to hold nested children
      this.childContainerElement = options.childContainerElement ? options.childContainerElement : ( this.childContainerElement || null );

      // @private - a parent element that can contain this domElement and other siblings, usually label and description content
      this.parentContainerElement = options.parentContainerElement ? options.parentContainerElement : ( this.parentContainerElement || null );
      if ( this.parentContainerElement ) {

        // The first child of the parent container element should be the peer dom element
        // if undefined, the insertBefore method will insert the peerDOMElement as the first child
        var peerDOMElement = this.domElement;
        var firstChild = this.parentContainerElement.children[ 0 ];
        this.parentContainerElement.insertBefore( peerDOMElement, firstChild );
      }

      this.disposed = false;

      // @private - listener for the focus event, to be disposed
      var focusEventListener = function( event ) {
        if ( event.target === self.domElement ) {
          scenery.Display.focus = new Focus( accessibleInstance.display, accessibleInstance.trail );
        }
      };
      this.domElement.addEventListener( 'focus', focusEventListener );

      // @private - listener for the blur event, to be disposed
      var blurEventListener = function( event ) {
        if ( event.target === self.domElement ) {
          scenery.Display.focus = null;
        }
      };
      this.domElement.addEventListener( 'blur', blurEventListener );

      // make AccessiblePeer eligible for garabage collection
      this.disposeAccessiblePeer = function() {

        // remove focus if the disposed peer currently has a focus highlight
        if ( scenery.Display.focus &&
            scenery.Display.focus.trail &&
              scenery.Display.focus.trail.equals( self.trail ) ) {

            scenery.Display.focus = null;
        }

        self.domElement.removeEventListener( 'blur', blurEventListener );
        self.domElement.removeEventListener( 'focus', focusEventListener );
      };

      return this;
    },

    /**
     * Check to see if this peer is contained in a parent container.
     *
     * @returns {boolean}
     */
    hasParentContainer: function() {
      return !!this.parentContainerElement;
    },

    /**
     * Get the parent container or the peer's dom element direclty.  Used for sorting.
     *
     * @returns {type}  description
     */
    getParentContainerElement: function() {
      return this.parentContainerElement || this.domElement;
    },

    /**
     * Get the child container or the peer's DOM element, used for sorting.
     *
     * @returns {type}  description
     */
    getChildContainerElement: function() {
      return this.childContainerElement || this.domElement;
    },

    /**
     * Get an element on this node, looked up by the association flag passed in.
     * @public (scenery-internal)
     * 
     * @param  {string} association - see AccessibilityUtil for valid associations
     * @return {HTMLElement}
     */
    getElementByAssociation: function( association ) {
      var htmlElement = null;

      if ( association === AccessiblePeer.NODE ) {
        return this.domElement;
      }
      else if ( association === AccessiblePeer.LABEL ) {
        return this.labelElement;
      }
      else if ( association === AccessiblePeer.DESCRIPTION ) {
        return this.descriptionElement;
      }
      else if ( association === AccessiblePeer.LABEL ) {
        return this.parentContainerElement;
      }
      assert && assert( htmlElement, 'no HTMLELement found by association ' + association );
      return htmlElement;
    },

    dispose: function() {
      this.disposed = true;
      this.disposeAccessiblePeer();

      // for now
      this.freeToPool && this.freeToPool();
    }
  }, {

    // @static - specifies valid associations between related AccessiblePeers in the DOM
    NODE: 'NODE', // associate with all accessible content related to this peer
    LABEL: 'LABEL', // associate with just the label content of this peer
    DESCRIPTION: 'DESCRIPTION', // associate with just the description content of this peer
    PARENT_CONTAINER: 'PARENT_CONTAINER' // associate with everything under the parent container element of this peer
  } );

  // TODO: evaluate pooling, and is it OK to pool only some peers?
  AccessiblePeer.Poolable = {
    mixin: function( selfDrawableType ) {
      // for pooling, allow <AccessiblePeerType>.createFromPool( accessibleInstance ) and accessiblePeer.freeToPool().
      // Creation will initialize the peer to an initial state.
      Poolable.mixin( selfDrawableType, {
        defaultFactory: function() {
          return new selfDrawableType();
        },
        constructorDuplicateFactory: function( pool ) {
          return function( accessibleInstance ) {
            if ( pool.length ) {
              return pool.pop().initialize( accessibleInstance );
            }
            else {
              return new selfDrawableType( accessibleInstance );
            }
          };
        }
      } );
    }
  };

  return AccessiblePeer;
} );
