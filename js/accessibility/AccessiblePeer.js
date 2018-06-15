// Copyright 2015-2016, University of Colorado Boulder

/**
 * An accessible peer controls the appearance of an accessible Node's instance in the parallel DOM.
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 * @author Jesse Greenberg
 */

define( function( require ) {
  'use strict';

  var AccessibilityUtil = require( 'SCENERY/accessibility/AccessibilityUtil' );
  var Events = require( 'AXON/Events' );
  var Focus = require( 'SCENERY/accessibility/Focus' );
  var inherit = require( 'PHET_CORE/inherit' );
  var Poolable = require( 'PHET_CORE/Poolable' );
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
     * Initializes the object (either from a freshly-created state, or from a "disposed" state brought back from a
     * pool)
     * @private
     *
     * @param {AccessibleInstance} accessibleInstance
     * @param {HTMLElement} primarySibling - The main DOM element used for this peer.
     * @param {Object} [options]
     * @returns {AccessiblePeer} - Returns 'this' reference, for chaining
     */
    initializeAccessiblePeer: function( accessibleInstance, primarySibling, options ) {
      options = _.extend( {
        containerParent: null, // a container parent for this peer and potential siblings
        labelSibling: null, // the element containing this node's label content
        descriptionSibling: null // the element that will contain this node's description content
      }, options );

      Events.call( this ); // TODO: is Events worth mixing in by default? Will we need to listen to events?

      assert && assert( !this.id || this.disposed, 'If we previously existed, we need to have been disposed' );

      // unique ID
      this.id = this.id || globalId++;

      // @public {AccessibleInstance}
      this.accessibleInstance = accessibleInstance;

      // @public {Display} - Each peer is associated with a specific Display.
      this.display = accessibleInstance.display;

      // @public {Trail} - NOTE: May have "gaps" due to accessibleOrder usage.
      this.trail = accessibleInstance.trail;

      // @public {HTMLElement} - The main element associated with this peer. If focusable, this is the element that gets
      // the focus. It also will contain any children.
      this.primarySibling = primarySibling;

      // @public {HTMLElement|null} - Optional label/description elements
      this.labelSibling = options.labelSibling;
      this.descriptionSibling = options.descriptionSibling;

      // @private {HTMLElement|null} - A parent element that can contain this primarySibling and other siblings, usually
      // the label and description content.
      this.containerParent = options.containerParent;

      if ( this.containerParent ) {
        // The first child of the container parent element should be the peer dom element
        // if undefined, the insertBefore method will insert the primarySiblingDOMElement as the first child
        var primarySiblingDOMElement = this.primarySibling;
        var firstChild = this.containerParent.children[ 0 ] || null;
        this.containerParent.insertBefore( primarySiblingDOMElement, firstChild  );
      }

      // @private {boolean} - Whether we are currently in a "disposed" (in the pool) state, or are available to be
      // interacted with.
      this.disposed = false;

      // @private {function} - Referenced for disposal
      this.focusEventListener = this.focusEventListener || this.onFocus.bind( this );
      this.blurEventListener = this.blurEventListener || this.onBlur.bind( this );

      // Hook up listeners for when our primary element is focused or blurred.
      this.primarySibling.addEventListener( 'blur', this.blurEventListener );
      this.primarySibling.addEventListener( 'focus', this.focusEventListener );

      return this;
    },

    /**
     * Called when our parallel DOM element gets focused.
     * @private
     *
     * @param {DOM Event} event
     */
    onFocus: function( event ) {
      if ( event.target === this.primarySibling ) {
        // NOTE: The "root" peer can't be focused (so it doesn't matter if it doesn't have a node).
        if ( this.accessibleInstance.node.focusable ) {
          scenery.Display.focus = new Focus( this.accessibleInstance.display, this.accessibleInstance.guessVisualTrail() );
          this.display.pointerFocus = null;
        }
      }
    },

    /**
     * Called when our parallel DOM element gets blurred (loses focus).
     * @private
     *
     * @param {DOM Event} event
     */
    onBlur: function( event ) {
      if ( event.target === this.primarySibling ) {
        scenery.Display.focus = null;
      }
    },

    /**
     * Get the container parent or the peer's dom element direclty.  Used for sorting.
     * @public (scenery-internal)
     *
     * @returns {HTMLElement}
     */
    getContainerParent: function() {
      return this.containerParent || this.primarySibling;
    },

    getTopLevelElements: function(){
    },

    /**
     * Get an element on this node, looked up by the association flag passed in.
     * @public (scenery-internal)
     *
     * @param {string} association - see AccessibilityUtil for valid associations
     * @return {HTMLElement}
     */
    getElementByName: function( association ) {
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

    /**
     * Removes external references from this peer, and places it in the pool.
     * @public (scenery-internal)
     */
    dispose: function() {
      this.disposed = true;

      // remove focus if the disposed peer currently has a focus highlight
      if ( scenery.Display.focus &&
           scenery.Display.focus.trail &&
           scenery.Display.focus.trail.equals( this.trail ) ) {

        scenery.Display.focus = null;
      }

      // remove listeners
      this.primarySibling.removeEventListener( 'blur', this.blurEventListener );
      this.primarySibling.removeEventListener( 'focus', this.focusEventListener );

      // zero-out references
      this.accessibleInstance = null;
      this.display = null;
      this.trail = null;
      this.primarySibling = null;
      this.labelSibling = null;
      this.descriptionSibling = null;
      this.containerParent = null;

      // for now
      this.freeToPool();
    },

    /**
     * Responsible for setting the content for the label sibling
     * @param {string} content - the content for the label sibling.
     * @param {boolean} isLabelTag - special treatment is given for a "label" tagName, as the "for" attribute is set to
     * the primary sibling.
     */
    setLabelSiblingContent: function( content, isLabelTag ) {
      assert && assert( this.labelSibling, 'must have a label sibling to set its content' );
      assert && assert( typeof content === 'string', 'incorrect label content type' );
      AccessibilityUtil.setTextContent( this.labelSibling, content );

      // if the label element happens to be a 'label', associate with 'for' attribute
      if ( isLabelTag ) {
        this.labelSibling.setAttribute( 'for', this.primarySibling.id );
      }
    },
    /**
     * Responsible for setting the content for the description sibling
     * @param {string} content - the content for the label sibling.
     * the primary sibling.
     */
    setDescriptionSiblingContent: function( content ) {
      assert && assert( this.descriptionSibling, 'must have a description sibling to set its content' );
      assert && assert( typeof content === 'string', 'incorrect description content type' );
      AccessibilityUtil.setTextContent( this.descriptionSibling, content );
    },

    /**
     * Responsible for setting the content for the primary sibling
     * @param {string} content - the content for the label sibling.
     * the primary sibling.
     */
    setPrimarySiblingContent: function( content ) {
      assert && assert( this.primarySibling, 'must have a primary sibling to set its content' );
      assert && assert( typeof content === 'string', 'incorrect inner content type' );
      assert && assert( this.accessibleInstance.children.length === 0, 'descendants exist with accessible content, innerContent cannot be used' );
      assert && assert( AccessibilityUtil.tagNameSupportsContent( this.primarySibling.tagName ),
        'tagName: ' + this._tagName + ' does not support inner content' );

      AccessibilityUtil.setTextContent( this.primarySibling, content );
    }

  }, {

    // @static - specifies valid associations between related AccessiblePeers in the DOM
    PRIMARY_SIBLING: 'PRIMARY_SIBLING', // associate with all accessible content related to this peer
    LABEL_SIBLING: 'LABEL_SIBLING', // associate with just the label content of this peer
    DESCRIPTION_SIBLING: 'DESCRIPTION_SIBLING', // associate with just the description content of this peer
    CONTAINER_PARENT: 'CONTAINER_PARENT' // associate with everything under the container parent of this peer
  } );

  // Set up pooling
  Poolable.mixInto( AccessiblePeer, {
    constructorDuplicateFactory: function( pool ) {
      return function( accessibleInstance, primarySibling, options ) {
        if ( pool.length ) {
          return pool.pop().initializeAccessiblePeer( accessibleInstance, primarySibling, options );
        }
        else {
          return new AccessiblePeer( accessibleInstance, primarySibling, options );
        }
      };
    }
  } );

  return AccessiblePeer;
} );
