// Copyright 2015-2016, University of Colorado Boulder

/**
 * An accessible peer controls the appearance of an accessible Node's instance in the parallel DOM.
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 * @author Jesse Greenberg
 */

define( function( require ) {
  'use strict';

  var Events = require( 'AXON/Events' );
  var Focus = require( 'SCENERY/accessibility/Focus' );
  var inherit = require( 'PHET_CORE/inherit' );
  var scenery = require( 'SCENERY/scenery' );
  // so RequireJS doesn't complain about circular dependency
  // var Display = require( 'SCENERY/display/Display' );

  var globalId = 1;

  /**
   * Constructor.
   *
   * @param {AccessibleInstance} accessibleInstance
   * @param {HTMLElement} primarySibling - The main DOM element used for this peer.
   * @param {Object} [options]
   * @constructor
   */
  function AccessiblePeer( accessibleInstance, primarySibling, options ) {
    this.initializeAccessiblePeer( accessibleInstance, primarySibling, options );
  }

  scenery.register( 'AccessiblePeer', AccessiblePeer );

  inherit( Events, AccessiblePeer, {

    /**
     * @param {AccessibleInstance} accessibleInstance
     * @param {HTMLElement} primarySibling - The main DOM element used for this peer.
     * @param {Object} [options]
     * @returns {AccessiblePeer} - Returns 'this' reference, for chaining
     */
    initializeAccessiblePeer: function( accessibleInstance, primarySibling, options ) {
      var self = this;

      options = _.extend( {
        containerParent: null, // a container parent for this peer and potential siblings
        childContainerElement: null, // an child container element where nested elements can be placed
        labelSibling: null, // the element containing this node's label content
        descriptionSibling: null // the element that will contain this node's description content
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
      this.primarySibling = primarySibling;
      this.labelSibling = options.labelSibling;
      this.descriptionSibling = options.descriptionSibling;

      // @private - descendent of the primarySibling that can be used to hold nested children
      this.childContainerElement = options.childContainerElement ? options.childContainerElement : ( this.childContainerElement || null );

      // @private - a parent element that can contain this primarySibling and other siblings, usually label and description content
      this.containerParent = options.containerParent ? options.containerParent : ( this.containerParent || null );
      if ( this.containerParent ) {

        // The first child of the container parent element should be the peer dom element
        // if undefined, the insertBefore method will insert the primarySiblingDOMElement as the first child
        var primarySiblingDOMElement = this.primarySibling;
        var firstChild = this.containerParent.children[ 0 ];
        this.containerParent.insertBefore( primarySiblingDOMElement, firstChild );
      }

      this.disposed = false;

      // @private - listener for the focus event, update Display but only if node is focusable,  but , to be disposed
      var focusEventListener = function( event ) {
        if ( event.target === self.primarySibling ) {
          if ( self.accessibleInstance.node.focusable ) {
            scenery.Display.focus = new Focus( accessibleInstance.display, accessibleInstance.trail );
            self.display.pointerFocus = null;
          }
        }
      };
      this.primarySibling.addEventListener( 'focus', focusEventListener );

      // @private - listener for the blur event, to be disposed
      var blurEventListener = function( event ) {
        if ( event.target === self.primarySibling ) {
          scenery.Display.focus = null;
        }
      };
      this.primarySibling.addEventListener( 'blur', blurEventListener );

      // make AccessiblePeer eligible for garabage collection
      this.disposeAccessiblePeer = function() {

        // remove focus if the disposed peer currently has a focus highlight
        if ( scenery.Display.focus &&
            scenery.Display.focus.trail &&
              scenery.Display.focus.trail.equals( self.trail ) ) {

            scenery.Display.focus = null;
        }

        self.primarySibling.removeEventListener( 'blur', blurEventListener );
        self.primarySibling.removeEventListener( 'focus', focusEventListener );
      };

      return this;
    },

    /**
     * Check to see if this peer is contained in a container parent.
     *
     * @returns {boolean}
     */
    hasContainerParent: function() {
      return !!this.containerParent;
    },

    /**
     * Get the container parent or the peer's dom element direclty.  Used for sorting.
     *
     * @returns {type}  description
     */
    getContainerParent: function() {
      return this.containerParent || this.primarySibling;
    },

    /**
     * Get the child container or the peer's DOM element, used for sorting.
     *
     * @returns {type}  description
     */
    getChildContainerElement: function() {
      return this.childContainerElement || this.primarySibling;
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

      if ( association === AccessiblePeer.PRIMARY_SIBLING ) {
        htmlElement = this.primarySibling;
      }
      else if ( association === AccessiblePeer.LABEL_SIBLING ) {
        htmlElement = this.labelSibling;
      }
      else if ( association === AccessiblePeer.DESCRIPTION_SIBLING ) {
        htmlElement = this.descriptionSibling;
      }
      else if ( association === AccessiblePeer.CONTAINER_PARENT ) {
        htmlElement = this.containerParent;
      }

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
    PRIMARY_SIBLING: 'PRIMARY_SIBLING', // associate with all accessible content related to this peer
    LABEL_SIBLING: 'LABEL_SIBLING', // associate with just the label content of this peer
    DESCRIPTION_SIBLING: 'DESCRIPTION_SIBLING', // associate with just the description content of this peer
    CONTAINER_PARENT: 'CONTAINER_PARENT' // associate with everything under the container parent of this peer
  } );

  return AccessiblePeer;
} );
