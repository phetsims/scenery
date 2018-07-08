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

  // constants
  var PRIMARY_SIBLING = 'PRIMARY_SIBLING';
  var LABEL_SIBLING = 'LABEL_SIBLING';
  var DESCRIPTION_SIBLING = 'DESCRIPTION_SIBLING';
  var CONTAINER_PARENT = 'CONTAINER_PARENT';
  var LABEL_TAG = AccessibilityUtil.TAGS.LABEL;
  var INPUT_TAG = AccessibilityUtil.TAGS.INPUT;

  /**
   * Constructor.
   *
   * @param {AccessibleInstance} accessibleInstance
   * @param {Object} [options]
   * @constructor
   */
  function AccessiblePeer( accessibleInstance, options ) {
    this.initializeAccessiblePeer( accessibleInstance, options );
  }

  scenery.register( 'AccessiblePeer', AccessiblePeer );

  inherit( Events, AccessiblePeer, {

    /**
     * Initializes the object (either from a freshly-created state, or from a "disposed" state brought back from a
     * pool)
     * @private
     *
     * @param {AccessibleInstance} accessibleInstance
     * @param {Object} [options]
     * @returns {AccessiblePeer} - Returns 'this' reference, for chaining
     */
    initializeAccessiblePeer: function( accessibleInstance, options ) {
      options = _.extend( {
        primarySibling: null // {HTMLElement} primarySibling - The main DOM element used for this peer
        // containerParent: null, // a container parent for this peer and potential siblings
        // labelSibling: null, // the element containing this node's label content
        // descriptionSibling: null // the element that will contain this node's description content
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

      // @private {boolean|null} - whether or not this AccessiblePeer is visible in the PDOM
      // Only initialized to null, should not be set to it. isVisible() will return true if this.visible is null (because it hasn't been set yet).
      this.visible = null;

      //
      // // @public {HTMLElement|null} - Optional label/description elements
      this.labelSibling = null;
      this.descriptionSibling = null;
      //
      // // @private {HTMLElement|null} - A parent element that can contain this primarySibling and other siblings, usually
      // // the label and description content.
      this.containerParent = null;

      // @public {Array.<HTMLElement>} Rather than guarantee that a peer is a tree with a root DOMElement,
      // allow multiple HTMLElements at the top level of the peer. This is used for sorting the instance
      this.topLevelElements = [];

      // @private {boolean} - Whether we are currently in a "disposed" (in the pool) state, or are available to be
      // interacted with.
      this.disposed = false;

      // edge case for root accessibility
      if ( options.primarySibling ) {
        // // @public {HTMLElement} - The main element associated with this peer. If focusable, this is the element that gets
        // // the focus. It also will contain any children.
        this.primarySibling = options.primarySibling;
        return this;
      }

      // redraw view from the AccessibleInstance's Node data.
      this.updateEntirePeer();


      return this;
    },

    /**
     * Temporary function to use as a placeholder for invalidateAccessibleContent
     * @public
     */
    updateEntirePeer: function() {
      var node = this.accessibleInstance.node;

      // for each accessible peer, clear the container parent if it exists since we will be reinserting labels and
      // the dom element in createPeer
      while ( this.containerParent && this.containerParent.hasChildNodes() ) {
        this.containerParent.removeChild( this.containerParent.lastChild );
      }

      // if any parents are flagged as removed from the accessibility tree, set content to null
      var contentDisplayed = node._accessibleContentDisplayed;
      for ( var i = 0; i < node._parents.length; i++ ) {
        if ( !node._parents[ i ].accessibleContentDisplayed ) {
          contentDisplayed = false;
        }
      }


      if ( contentDisplayed && node._tagName ) {


        // clear out elements to be recreated below
        this.primarySibling = null;
        this.labelSibling = null;
        this.descriptionSibling = null;
        this.containerParent = null;


        // higher level api first, because it will effect the lower level setters.
        // if ( node.accessibleName ) {
        //   node.setAccessibleNameImplementation( node.accessibleName ); // set it again to support any option order
        // }
        //
        // if ( node.helpText ) {
        //   node.setHelpTextImplementation( node.helpText ); // set it again to support any option order
        // }

        var uniqueId = this.accessibleInstance.trail.getUniqueId();

        // create the base DOM element representing this accessible instance
        var primarySibling = AccessibilityUtil.createElement( node._tagName, node.focusable, {
          namespace: node._accessibleNamespace
        } );
        primarySibling.id = uniqueId;

        // create the container parent for the dom siblings
        var containerParent = null;
        if ( node._containerTagName ) {
          containerParent = AccessibilityUtil.createElement( node._containerTagName, false );
          containerParent.id = 'container-' + uniqueId;

          // provide the aria-role if it is specified
          if ( node._containerAriaRole ) {
            containerParent.setAttribute( 'role', node._containerAriaRole );
          }
        }

        // create the label DOM element representing this instance
        var labelSibling = null;
        if ( node._labelTagName ) {
          labelSibling = AccessibilityUtil.createElement( node._labelTagName, false );
          labelSibling.id = 'label-' + uniqueId;
        }

        // create the description DOM element representing this instance
        var descriptionSibling = null;
        if ( node._descriptionTagName ) {
          descriptionSibling = AccessibilityUtil.createElement( node._descriptionTagName, false );
          descriptionSibling.id = 'description-' + uniqueId;
        }


        this.primarySibling = primarySibling;
        this.labelSibling = labelSibling;
        this.descriptionSibling = descriptionSibling;
        this.containerParent = containerParent;

        if ( this.containerParent ) {
          // The first child of the container parent element should be the peer dom element
          // if undefined, the insertBefore method will insert the primarySiblingDOMElement as the first child
          var primarySiblingDOMElement = this.primarySibling;
          var firstChild = this.containerParent.children[ 0 ] || null;
          this.containerParent.insertBefore( primarySiblingDOMElement, firstChild );
          this.topLevelElements = [ this.containerParent ];
        }
        else {

          // Wean out any null siblings
          this.topLevelElements = [ this.labelSibling, this.descriptionSibling, this.primarySibling ].filter( function( i ) { return i; } );
        }

        // @private {function} - Referenced for disposal
        this.focusEventListener = this.focusEventListener || this.onFocus.bind( this );
        this.blurEventListener = this.blurEventListener || this.onBlur.bind( this );

        // Hook up listeners for when our primary element is focused or blurred.
        this.primarySibling.addEventListener( 'blur', this.blurEventListener );
        this.primarySibling.addEventListener( 'focus', this.focusEventListener );


        // set the accessible label now that the element has been recreated again, but not if the tagName
        // has been cleared out
        if ( node._labelContent && node._labelTagName !== null ) {
          this.setLabelSiblingContent( node._labelContent );
        }

        // restore the innerContent
        if ( node._innerContent && node._tagName !== null ) {
          this.setPrimarySiblingContent( node._innerContent );
        }

        // set the accessible description, but not if the tagName has been cleared out.
        if ( node._descriptionContent && node._descriptionTagName !== null ) {
          this.setDescriptionSiblingContent( node._descriptionContent );
        }

        // set if using aria-label
        node._ariaLabel && this.setAttributeToElement( 'aria-label', node._ariaLabel );

        // restore checked
        this.setAttributeToElement( 'checked', node._accessibleChecked );

        // restore input value
        node._inputValue && this.setAttributeToElement( 'value', node._inputValue );


        // set the accessible attributes, restoring from a defensive copy
        var defensiveAttributes = node.accessibleAttributes;
        for ( i = 0; i < defensiveAttributes.length; i++ ) {
          var attribute = defensiveAttributes[ i ].attribute;
          var value = defensiveAttributes[ i ].value;
          var namespace = defensiveAttributes[ i ].namespace;
          this.setAttributeToElement( attribute, value, {
            namespace: namespace
          } );
        }

        // if element is an input element, set input type
        if ( node._tagName.toUpperCase() === INPUT_TAG && node._inputType ) {
          this.setAttributeToElement( 'type', node._inputType );
        }

        // TODO: factor this out to be in peer instead of node
        // recompute and assign the association attributes that link two elements (like aria-labelledby)
        // node.updateLabelledbyDescribebyAssociations();

        // add all listeners to the dom element
        for ( i = 0; i < node._accessibleInputListeners.length; i++ ) {
          this.addDOMEventListeners( node._accessibleInputListeners[ i ] );
        }

        // insert the label and description elements in the correct location if they exist
        labelSibling && this.arrangeContentElement( labelSibling, node._appendLabel );
        descriptionSibling && this.arrangeContentElement( descriptionSibling, node._appendDescription );

        // Default the focus highlight in this special case to be invisible until selected.
        if ( node._focusHighlightLayerable ) {
          node._focusHighlight.visible = false;
        }

        assert && assert( this.primarySibling );

      }
      else {
        node.accessibleContent = false; // otherwise clear the accessibleContent
      }

    },

    onTagNameChange: function() {

      this.setHasAccessibleContent();
    },

    onLabelTagNameChange: function() {

      this.setHasAccessibleContent();
    },
    onDescriptionTagNameChange: function() {

      this.setHasAccessibleContent();
    },
    onAppendLabelChange: function() {

      this.setHasAccessibleContent();
    },
    onAppendDescriptionChange: function() {

      this.setHasAccessibleContent();
    },
    onContainerTagNameChange: function() {

      this.setHasAccessibleContent();
    },
    onAriaRoleChange: function() {

      this.setHasAccessibleContent();
    },
    onContainerAriaRoleChange: function() {

      this.setHasAccessibleContent();
    },
    onAccessibleNamespaceChange: function() {

      this.setHasAccessibleContent();
    },
    onFocusHighlightChange: function() {

      this.setHasAccessibleContent();
    },
    onFocusHighlightLayerableChange: function() {

      this.setHasAccessibleContent();
    },
    onAccessibleContentDisplayedChange: function() {

      this.setHasAccessibleContent();
    },

    setHasAccessibleContent: function() {
      var node = this.accessibleInstance.node;

      // if any parents are flagged as removed from the accessibility tree, set content to null
      var contentDisplayed = node._accessibleContentDisplayed;
      for ( var i = 0; i < node._parents.length; i++ ) {
        if ( !node._parents[ i ].accessibleContentDisplayed ) {
          contentDisplayed = false;
        }
      }


      if ( contentDisplayed && node._tagName ) {
        node.accessibleContent = true;
      }
      else {
        node.accessibleContent = false;
      }
    },

    /**
     * Called when our parallel DOM element gets focused.
     * @private
     *
     * @param {DOMEvent} event
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
     * @param {DOMEvent} event
     */
    onBlur: function( event ) {
      if ( event.target === this.primarySibling ) {
        scenery.Display.focus = null;
      }
    },

    /**
     * Get an element on this node, looked up by the elementName flag passed in.
     * @public (scenery-internal)
     *
     * @param {string} elementName - see AccessibilityUtil for valid associations
     * @return {HTMLElement}
     */
    getElementByName: function( elementName ) {
      if ( elementName === AccessiblePeer.PRIMARY_SIBLING ) {
        return this.primarySibling;
      }
      else if ( elementName === AccessiblePeer.LABEL_SIBLING ) {
        return this.labelSibling;
      }
      else if ( elementName === AccessiblePeer.DESCRIPTION_SIBLING ) {
        return this.descriptionSibling;
      }
      else if ( elementName === AccessiblePeer.CONTAINER_PARENT ) {
        return this.containerParent;
      }

      assert && assert( false, 'invalid elementName name: ' + elementName );
    },

    /**
     * Add DOM Event listeners to the peer's primary sibling.
     * @public (scenery-internal)
     *
     * @param {Object} accessibleInput - see Accessibility.addAccessibleInputListener
     */
    addDOMEventListeners: function( accessibleInput ) {
      AccessibilityUtil.addDOMEventListeners( accessibleInput, this.primarySibling );
    },
    /**
     * Remove DOM Event listeners from the peer's primary sibling.
     * @public (scenery-internal)
     * @param {Object} accessibleInput - see Accessibility.addAccessibleInputListener
     */
    removeDOMEventListeners: function( accessibleInput ) {
      AccessibilityUtil.removeDOMEventListeners( accessibleInput, this.primarySibling );
    },

    /**
     * Sets a attribute on one of the peer's HTMLElements.
     * @public (scenery-internal)
     * @param {string} attribute
     * @param {*} attributeValue
     * @param {Object} [options]
     */
    setAttributeToElement: function( attribute, attributeValue, options ) {

      options = _.extend( {
        // {string|null} - If non-null, will set the attribute with the specified namespace. This can be required
        // for setting certain attributes (e.g. MathML).
        namespace: null,

        elementName: PRIMARY_SIBLING // see this.getElementName() for valid values, default to the primary sibling
      }, options );

      var element = this.getElementByName( options.elementName );

      if ( options.namespace ) {
        element.setAttributeNS( options.namespace, attribute, attributeValue );
      }
      else {
        element.setAttribute( attribute, attributeValue );
      }
    },
    /**
     * Remove attribute from one of the peer's HTMLElements.
     * @public (scenery-internal)
     * @param {string} attribute
     * @param {Object} [options]
     */
    removeAttributeFromElement: function( attribute, options ) {

      options = _.extend( {
        // {string|null} - If non-null, will set the attribute with the specified namespace. This can be required
        // for setting certain attributes (e.g. MathML).
        namespace: null,

        elementName: PRIMARY_SIBLING // see this.getElementName() for valid values, default to the primary sibling
      }, options );

      var element = this.getElementByName( options.elementName );

      if ( options.namespace ) {
        element.removeAttributeNS( options.namespace, attribute );
      }
      else {
        element.removeAttribute( attribute );
      }
    },

    /**
     * Remove the given attribute from all peer elements
     * @public (scenery-internal)
     * @param {string} attribute
     */
    removeAttributeFromAllElements: function( attribute ) {
      assert && assert( typeof attribute === 'string' );
      this.primarySibling && this.primarySibling.removeAttribute( attribute );
      this.labelSibling && this.labelSibling.removeAttribute( attribute );
      this.descriptionSibling && this.descriptionSibling.removeAttribute( attribute );
      this.containerParent && this.containerParent.removeAttribute( attribute );
    },

    /**
     * Set either association attribute (aria-labelledby/describedby) on one of this peer's Elements
     * @public (scenery-internal)
     * @param {string} attribute - either aria-labelledby or aria-describedby
     * @param {Object} associationObject - see addAriaLabelledbyAssociation() for schema
     * @param {string} otherElementId - the id to be added to this peer element's attribute
     */
    setAssociationAttribute: function( attribute, associationObject, otherElementId ) {
      assert && assert( attribute === 'aria-labelledby' || attribute === 'aria-describedby',
        'unsupported attribute for setting with association object' );
      assert && assert( typeof otherElementId === 'string' );
      assert && AccessibilityUtil.validateAssociationObject( associationObject );

      var element = this.getElementByName( associationObject.thisElementName );

      // to support any option order, no-op if the peer element has not been created yet.
      if ( element ) {

        // only update associations if the requested peer element has been created
        // NOTE: in the future, we would like to verify that the association exists but can't do that yet because
        // we have to support cases where we set label association prior to setting the sibling/parent tagName
        var previousAttributeValue = element.getAttribute( attribute ) || '';
        assert && assert( typeof previousAttributeValue === 'string' );

        var newAttributeValue = [ previousAttributeValue.trim(), otherElementId ].join( ' ' ).trim();

        // add the id from the new association to the value of the HTMLElement's attribute.
        this.setAttributeToElement( attribute, newAttributeValue, {
          elementName: associationObject.thisElementName
        } );
      }
    },
    /**
     * Called by invalidateAccessibleContent. The contentElement will either be a
     * label or description element. The contentElement will be sorted relative to the primarySibling. Its placement
     * will also depend on whether or not this node wants to append this element,
     * see setAppendLabel() and setAppendDescription(). By default, the "content" element will be placed before the
     * primarySibling.
     *
     *
     * @param {HTMLElement} contentElement
     * @param {boolean} appendElement
     */
    arrangeContentElement: function( contentElement, appendElement ) {

      // if there is a containerParent
      if ( this.topLevelElements[ 0 ] === this.containerParent ) {
        assert && assert( this.topLevelElements.length === 1 );

        if ( appendElement ) {
          this.containerParent.appendChild( contentElement );
        }
        else {
          this.containerParent.insertBefore( contentElement, this.primarySibling );
        }
      }

      // If there are multiple top level nodes
      else {
        assert && assert( this.topLevelElements.indexOf( contentElement ) >= 0, 'element is not part of this peer, thus cannot be arranged' );

        // keep this.topLevelElements in sync
        this.topLevelElements.splice( this.topLevelElements.indexOf( contentElement ), 1 );

        var indexOffset = appendElement ? 1 : 0;
        var indexOfContentElement = this.topLevelElements.indexOf( this.primarySibling ) + indexOffset;
        indexOfContentElement = indexOfContentElement < 0 ? 0 : indexOfContentElement; //support primarySibling in the first position
        this.topLevelElements.splice( indexOfContentElement, 0, contentElement );
      }
    },


    /**
     * Is this peer hidden in the PDOM
     * @returns {boolean}
     */
    isVisible: function() {
      if ( assert ) {

        var visibleElements = 0;
        this.topLevelElements.forEach( function( element ) {
          if ( !element.hidden ) {
            visibleElements += 1;
          }
        } );
        assert( this.visible ? visibleElements === this.topLevelElements.length : visibleElements === 0,
          'some of the peer\'s elements are visible and some are not' );

      }
      return this.visible === null ? true : this.visible; // default to true if visibility hasn't been set yet.
    },

    /**
     * Set whether or not the peer is visible in the PDOM
     * @param {boolean} visible
     */
    setVisible: function( visible ) {
      assert && assert( typeof visible === 'boolean' );
      if ( this.visible !== visible ) {

        this.visible = visible;
        for ( var i = 0; i < this.topLevelElements.length; i++ ) {
          var element = this.topLevelElements[ i ];
          if ( visible ) {
            element.removeAttribute( 'hidden' );
          }
          else {
            element.setAttribute( 'hidden', '' );
          }
        }
      }
    },

    /**
     * Returns if this peer is focused. A peer is focused if its primarySibling is focused.
     * @public (scenery-internal)
     * @returns {boolean}
     */
    isFocused: function() {
      return document.activeElement === this.primarySibling;
    },

    /**
     * Focus the primary sibling of the peer.
     * @public (scenery-internal)
     */
    focus: function() {
      assert && assert( this.primarySibling, 'must have a primary sibling to focus' );
      this.primarySibling.focus();
    },

    /**
     * Blur the primary sibling of the peer.
     * @public (scenery-internal)
     */
    blur: function() {
      assert && assert( this.primarySibling, 'must have a primary sibling to blur' );
      this.primarySibling.blur();
    },


    /**
     * Responsible for setting the content for the label sibling
     * @public (scenery-internal)
     * @param {string} content - the content for the label sibling.
     * the primary sibling.
     */
    setLabelSiblingContent: function( content ) {
      assert && assert( typeof content === 'string', 'incorrect label content type' );

      // no-op to support any option order
      if ( !this.labelSibling ) {
        return;
      }

      AccessibilityUtil.setTextContent( this.labelSibling, content );

      // if the label element happens to be a 'label', associate with 'for' attribute
      if ( this.labelSibling.tagName.toUpperCase() === LABEL_TAG ) {
        this.labelSibling.setAttribute( 'for', this.primarySibling.id );
      }
    },
    /**
     * Responsible for setting the content for the description sibling
     * @public (scenery-internal)
     * @param {string} content - the content for the label sibling.
     * the primary sibling.
     */
    setDescriptionSiblingContent: function( content ) {
      assert && assert( typeof content === 'string', 'incorrect description content type' );

      // no-op to support any option order
      if ( !this.descriptionSibling ) {
        return;
      }
      AccessibilityUtil.setTextContent( this.descriptionSibling, content );
    },

    /**
     * Responsible for setting the content for the primary sibling
     * @public (scenery-internal)
     * @param {string} content - the content for the label sibling.
     * the primary sibling.
     */
    setPrimarySiblingContent: function( content ) {
      assert && assert( typeof content === 'string', 'incorrect inner content type' );
      assert && assert( this.accessibleInstance.children.length === 0, 'descendants exist with accessible content, innerContent cannot be used' );
      assert && assert( AccessibilityUtil.tagNameSupportsContent( this.primarySibling.tagName ),
        'tagName: ' + this._tagName + ' does not support inner content' );

      // no-op to support any option order
      if ( !this.primarySibling ) {
        return;
      }
      AccessibilityUtil.setTextContent( this.primarySibling, content );
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
    }
  }, {

    // @static - specifies valid associations between related AccessiblePeers in the DOM
    PRIMARY_SIBLING: PRIMARY_SIBLING, // associate with all accessible content related to this peer
    LABEL_SIBLING: LABEL_SIBLING, // associate with just the label content of this peer
    DESCRIPTION_SIBLING: DESCRIPTION_SIBLING, // associate with just the description content of this peer
    CONTAINER_PARENT: CONTAINER_PARENT // associate with everything under the container parent of this peer
  } );

  // Set up pooling
  Poolable.mixInto( AccessiblePeer, {
    initalize: AccessiblePeer.prototype.initializeAccessiblePeer
  } );

  return AccessiblePeer;
} );
