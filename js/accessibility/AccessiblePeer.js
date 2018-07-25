// Copyright 2015-2016, University of Colorado Boulder

/**
 * An accessible peer controls the appearance of an accessible Node's instance in the parallel DOM. An AccessiblePeer can
 * have up to four HTMLElements displayed in the PDOM, see constructor for details.
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
      }, options );

      Events.call( this ); // TODO: is Events worth mixing in by default? Will we need to listen to events?

      assert && assert( !this.id || this.disposed, 'If we previously existed, we need to have been disposed' );

      // unique ID
      this.id = this.id || globalId++;

      // @public {AccessibleInstance}
      this.accessibleInstance = accessibleInstance;

      // @public {Node}
      this.node = this.accessibleInstance.node;

      // @public {Display} - Each peer is associated with a specific Display.
      this.display = accessibleInstance.display;

      // @public {Trail} - NOTE: May have "gaps" due to accessibleOrder usage.
      this.trail = accessibleInstance.trail;

      // @private {boolean|null} - whether or not this AccessiblePeer is visible in the PDOM
      // Only initialized to null, should not be set to it. isVisible() will return true if this.visible is null (because it hasn't been set yet).
      this.visible = null;

      // @private {HTMLElement|null} - Optional label/description elements
      this._labelSibling = null;
      this._descriptionSibling = null;

      // @private {HTMLElement|null} - A parent element that can contain this primarySibling and other siblings, usually
      // the label and description content.
      this._containerParent = null;

      // @public {Array.<HTMLElement>} Rather than guarantee that a peer is a tree with a root DOMElement,
      // allow multiple HTMLElements at the top level of the peer. This is used for sorting the instance.
      // See this.orderElements for more info.
      this.topLevelElements = [];

      // @private {boolean} - Whether we are currently in a "disposed" (in the pool) state, or are available to be
      // interacted with.
      this.disposed = false;

      // edge case for root accessibility
      if ( options.primarySibling ) {

        // @private {HTMLElement} - The main element associated with this peer. If focusable, this is the element that gets
        // the focus. It also will contain any children.
        this._primarySibling = options.primarySibling;
        return this;
      }

      // for each accessible peer, clear the container parent if it exists since we will be reinserting labels and
      // the dom element in createPeer
      while ( this._containerParent && this._containerParent.hasChildNodes() ) {
        this._containerParent.removeChild( this._containerParent.lastChild );
      }

      // clear out elements to be recreated below
      this._primarySibling = null;
      this._labelSibling = null;
      this._descriptionSibling = null;
      this._containerParent = null;

      this.update();

      return this;
    },

    /**
     * Update the content of the peer
     * @private
     */
    update: function() {
      var i;
      var uniqueId = this.accessibleInstance.trail.getUniqueId();

      var options = this.node.getBaseOptions();

      if ( this.node.accessibleName !== null ) {
        options = this.node.accessibleNameBehavior( this.node, options, this.node.accessibleName );
      }

      if ( this.node.helpText !== null ) {
        options = this.node.helpTextBehavior( this.node, options, this.node.helpText );
      }


      // create the base DOM element representing this accessible instance
      var primarySibling = AccessibilityUtil.createElement( options.tagName, this.node.focusable, {
        namespace: options.accessibleNamespace
      } );
      primarySibling.id = uniqueId;

      // create the container parent for the dom siblings
      var containerParent = null;
      if ( options.containerTagName ) {
        containerParent = AccessibilityUtil.createElement( options.containerTagName, false );
        containerParent.id = 'container-' + uniqueId;

        // provide the aria-role if it is specified
        if ( options.containerAriaRole ) {
          containerParent.setAttribute( 'role', options.containerAriaRole );
        }
      }

      // create the label DOM element representing this instance
      var labelSibling = null;
      if ( options.labelTagName ) {
        labelSibling = AccessibilityUtil.createElement( options.labelTagName, false );
        labelSibling.id = 'label-' + uniqueId;
      }

      // create the description DOM element representing this instance
      var descriptionSibling = null;
      if ( options.descriptionTagName ) {
        descriptionSibling = AccessibilityUtil.createElement( options.descriptionTagName, false );
        descriptionSibling.id = 'description-' + uniqueId;
      }


      this._primarySibling = primarySibling;
      this._labelSibling = labelSibling;
      this._descriptionSibling = descriptionSibling;
      this._containerParent = containerParent;

      this.orderElements();

      // @private {function} - Referenced for disposal
      this.focusEventListener = this.focusEventListener || this.onFocus.bind( this );
      this.blurEventListener = this.blurEventListener || this.onBlur.bind( this );

      // Hook up listeners for when our primary element is focused or blurred.
      this._primarySibling.addEventListener( 'blur', this.blurEventListener );
      this._primarySibling.addEventListener( 'focus', this.focusEventListener );


      // set the accessible label now that the element has been recreated again, but not if the tagName
      // has been cleared out
      if ( options.labelContent && options.labelTagName !== null ) {
        this.setLabelSiblingContent( options.labelContent );
      }

      // restore the innerContent
      if ( options.innerContent && options.tagName !== null ) {
        this.setPrimarySiblingContent( options.innerContent );
      }

      // set the accessible description, but not if the tagName has been cleared out.
      if ( options.descriptionContent && options.descriptionTagName !== null ) {
        this.setDescriptionSiblingContent( options.descriptionContent );
      }

      // if element is an input element, set input type
      if ( options.tagName.toUpperCase() === INPUT_TAG && options.inputType ) {
        this.setAttributeToElement( 'type', options.inputType );
      }

      // recompute and assign the association attributes that link two elements (like aria-labelledby)
      this.onAriaLabelledbyAssociationChange();
      this.onAriaDescribedbyAssociationChange();


      // add all listeners to the dom element
      for ( i = 0; i < this.node.accessibleInputListeners.length; i++ ) {
        this.addDOMEventListeners( this.node.accessibleInputListeners[ i ] );
      }

      // update all attributes for the peer, should cover aria-label, role, input value and others
      this.onAttributeChange( options );

      // Default the focus highlight in this special case to be invisible until selected.
      if ( this.node.focusHighlightLayerable ) {

        // if focus highlight is layerable, it must be a node in the scene graph
        assert && assert( this.node.focusHighlight instanceof phet.scenery.Node );
        this.node.focusHighlight.visible = false;
      }

      // TODO: this is hacky, because updateOtherNodes functions could try to access this peer from their accessibleInstance.
      this.accessibleInstance.peer = this;
      this.node.updateOtherNodesAriaLabelledby();
      this.node.updateOtherNodesAriaDescribedby();
    },

    /**
     * Handle the internal ordering of the elements in the peer, this involves setting the proper value of
     * this.topLevelElements
     * @private
     */
    orderElements: function() {

      var truthySiblings = [ this._labelSibling, this._descriptionSibling, this._primarySibling ].filter( function( i ) { return i; } );

      if ( this._containerParent ) {
        // The first child of the container parent element should be the peer dom element
        // if undefined, the insertBefore method will insert the primarySiblingDOMElement as the first child
        var primarySiblingDOMElement = this._primarySibling;
        var firstChild = this._containerParent.children[ 0 ] || null;
        this._containerParent.insertBefore( primarySiblingDOMElement, firstChild );
        this.topLevelElements = [ this._containerParent ];
      }
      else {

        // Wean out any null siblings
        this.topLevelElements = truthySiblings;
      }

      // insert the label and description elements in the correct location if they exist
      this._labelSibling && this.arrangeContentElement( this._labelSibling, this.node.appendLabel );
      this._descriptionSibling && this.arrangeContentElement( this._descriptionSibling, this.node.appendDescription );

    },

    /**
     * Get the primary sibling element for the peer
     * @public
     * @returns {HTMLElement|null}
     */
    getPrimarySibling: function() {
      return this._primarySibling;
    },

    get primarySibling() { return this.getPrimarySibling(); },
    /**
     * Get the primary sibling element for the peer
     * @public
     * @returns {HTMLElement|null}
     */
    getLabelSibling: function() {
      return this._labelSibling;
    },
    get labelSibling() { return this.getLabelSibling(); },
    /**
     * Get the primary sibling element for the peer
     * @public
     * @returns {HTMLElement|null}
     */
    getDescriptionSibling: function() {
      return this._descriptionSibling;
    },
    get descriptionSibling() { return this.getDescriptionSibling(); },
    /**
     * Get the primary sibling element for the peer
     * @public
     * @returns {HTMLElement|null}
     */
    getContainerParent: function() {
      return this._containerParent;
    },
    get containerParent() { return this.getContainerParent(); },

    /**
     * Recompute the aria-labelledby attributes for all of the peer's elements
     * @public
     */
    onAriaLabelledbyAssociationChange: function() {
      this.removeAttributeFromAllElements( 'aria-labelledby' );

      for ( var i = 0; i < this.node.ariaLabelledbyAssociations.length; i++ ) {
        var associationObject = this.node.ariaLabelledbyAssociations[ i ];

        // Assert out if the model list is different than the data held in the associationObject
        assert && assert( associationObject.otherNode.nodesThatAreAriaLabelledbyThisNode.indexOf( this.node ) >= 0,
          'unexpected otherNode' );


        this.setAssociationAttribute( 'aria-labelledby', associationObject );
      }
    },

    /**
     * Recompute the aria-describedby attributes for all of the peer's elements
     * @public
     */
    onAriaDescribedbyAssociationChange: function() {
      this.removeAttributeFromAllElements( 'aria-describedby' );

      for ( var i = 0; i < this.node.ariaDescribedbyAssociations.length; i++ ) {
        var associationObject = this.node.ariaDescribedbyAssociations[ i ];

        // Assert out if the model list is different than the data held in the associationObject
        assert && assert( associationObject.otherNode.nodesThatAreAriaDescribedbyThisNode.indexOf( this.node ) >= 0,
          'unexpected otherNode' );


        this.setAssociationAttribute( 'aria-describedby', associationObject );
      }
    },

    /**
     * Set all accessible attributes onto the peer elements from the model's stored data objects
     * @param {Object} [a11yOptions] - these can override the values of the node, see this.update()
     */
    onAttributeChange: function( a11yOptions ) {

      for ( var i = 0; i < this.node.accessibleAttributes.length; i++ ) {
        var dataObject = this.node.accessibleAttributes[ i ];
        var attribute = dataObject.attribute;
        var value = dataObject.value;

        // allow overriding of aria-label for accessibleName setter
        // TODO: this is a specific workaround, it would be nice to sort out a general case for this, #795
        if ( attribute === 'aria-label' && a11yOptions && typeof a11yOptions.ariaLabel === 'string' && dataObject.options.elementName === PRIMARY_SIBLING ) {
          value = a11yOptions.ariaLabel;
        }
        this.setAttributeToElement( attribute, value, dataObject.options );
      }
    },

    /**
     * Called when our parallel DOM element gets focused.
     * @private
     *
     * @param {DOMEvent} event
     */
    onFocus: function( event ) {
      if ( event.target === this._primarySibling ) {
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
      if ( event.target === this._primarySibling ) {
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
        return this._primarySibling;
      }
      else if ( elementName === AccessiblePeer.LABEL_SIBLING ) {
        return this._labelSibling;
      }
      else if ( elementName === AccessiblePeer.DESCRIPTION_SIBLING ) {
        return this._descriptionSibling;
      }
      else if ( elementName === AccessiblePeer.CONTAINER_PARENT ) {
        return this._containerParent;
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
      AccessibilityUtil.addDOMEventListeners( accessibleInput, this._primarySibling );
    },
    /**
     * Remove DOM Event listeners from the peer's primary sibling.
     * @public (scenery-internal)
     * @param {Object} accessibleInput - see Accessibility.addAccessibleInputListener
     */
    removeDOMEventListeners: function( accessibleInput ) {
      AccessibilityUtil.removeDOMEventListeners( accessibleInput, this._primarySibling );
    },

    /**
     * Sets a attribute on one of the peer's HTMLElements.
     * NOTE: If the attributeValue is a boolean, then it will be set as a javascript property on the HTMLElement rather than an attribute
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

      // treat it like a property
      else if ( typeof attributeValue === 'boolean' ) {
        element[ attribute ] = attributeValue;
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
      this._primarySibling && this._primarySibling.removeAttribute( attribute );
      this._labelSibling && this._labelSibling.removeAttribute( attribute );
      this._descriptionSibling && this._descriptionSibling.removeAttribute( attribute );
      this._containerParent && this._containerParent.removeAttribute( attribute );
    },

    /**
     * Set either association attribute (aria-labelledby/describedby) on one of this peer's Elements
     * @public (scenery-internal)
     * @param {string} attribute - either aria-labelledby or aria-describedby
     * @param {Object} associationObject - see addAriaLabelledbyAssociation() for schema
     */
    setAssociationAttribute: function( attribute, associationObject ) {
      assert && assert( attribute === 'aria-labelledby' || attribute === 'aria-describedby',
        'unsupported attribute for setting with association object: ' + attribute );
      assert && AccessibilityUtil.validateAssociationObject( associationObject );

      var otherNodeAccessibleInstances = associationObject.otherNode.getAccessibleInstances();

      // If the other node hasn't been added to the scene graph yet, it won't have any accessible instances, so no op.
      // This will be recalculated when that node is added to the scene graph
      if ( otherNodeAccessibleInstances.length > 0 ) {

        // We are just using the first AccessibleInstance for simplicity, but it is OK because the accessible
        // content for all AccessibleInstances will be the same, so the Accessible Names (in the browser's
        // accessibility tree) of elements that are referenced by the attribute value id will all have the same content
        var firstAccessibleInstance = otherNodeAccessibleInstances[ 0 ];

        // Handle a case where you are associating to yourself, and the peer has not been constructed yet.
        if ( firstAccessibleInstance === this.accessibleInstance ) {
          firstAccessibleInstance.peer = this;
        }

        assert && assert( firstAccessibleInstance.peer, 'peer should exist' );

        // we can use the same element's id to update all of this Node's peers
        var otherPeerElement = firstAccessibleInstance.peer.getElementByName( associationObject.otherElementName );

        var element = this.getElementByName( associationObject.thisElementName );

        // to support any option order, no-op if the peer element has not been created yet.
        if ( element && otherPeerElement ) {

          // only update associations if the requested peer element has been created
          // NOTE: in the future, we would like to verify that the association exists but can't do that yet because
          // we have to support cases where we set label association prior to setting the sibling/parent tagName
          var previousAttributeValue = element.getAttribute( attribute ) || '';
          assert && assert( typeof previousAttributeValue === 'string' );

          var newAttributeValue = [ previousAttributeValue.trim(), otherPeerElement.id ].join( ' ' ).trim();

          // add the id from the new association to the value of the HTMLElement's attribute.
          this.setAttributeToElement( attribute, newAttributeValue, {
            elementName: associationObject.thisElementName
          } );
        }
      }
    },
    /**
     * The contentElement will either be a
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
      if ( this.topLevelElements[ 0 ] === this._containerParent ) {
        assert && assert( this.topLevelElements.length === 1 );

        if ( appendElement ) {
          this._containerParent.appendChild( contentElement );
        }
        else {
          this._containerParent.insertBefore( contentElement, this._primarySibling );
        }
      }

      // If there are multiple top level nodes
      else {
        assert && assert( this.topLevelElements.indexOf( contentElement ) >= 0, 'element is not part of this peer, thus cannot be arranged' );

        // keep this.topLevelElements in sync
        this.topLevelElements.splice( this.topLevelElements.indexOf( contentElement ), 1 );

        var indexOffset = appendElement ? 1 : 0;
        var indexOfContentElement = this.topLevelElements.indexOf( this._primarySibling ) + indexOffset;
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

          // support property or attribute
          if ( !element.hidden && !element.hasAttribute( 'hidden' ) ) {
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
      return document.activeElement === this._primarySibling;
    },

    /**
     * Focus the primary sibling of the peer.
     * @public (scenery-internal)
     */
    focus: function() {
      assert && assert( this._primarySibling, 'must have a primary sibling to focus' );
      this._primarySibling.focus();
    },

    /**
     * Blur the primary sibling of the peer.
     * @public (scenery-internal)
     */
    blur: function() {
      assert && assert( this._primarySibling, 'must have a primary sibling to blur' );
      this._primarySibling.blur();
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
      if ( !this._labelSibling ) {
        return;
      }

      AccessibilityUtil.setTextContent( this._labelSibling, content );

      // if the label element happens to be a 'label', associate with 'for' attribute
      if ( this._labelSibling.tagName.toUpperCase() === LABEL_TAG ) {
        this._labelSibling.setAttribute( 'for', this._primarySibling.id );
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
      if ( !this._descriptionSibling ) {
        return;
      }
      AccessibilityUtil.setTextContent( this._descriptionSibling, content );
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
      assert && assert( AccessibilityUtil.tagNameSupportsContent( this._primarySibling.tagName ),
        'tagName: ' + this._tagName + ' does not support inner content' );

      // no-op to support any option order
      if ( !this._primarySibling ) {
        return;
      }
      AccessibilityUtil.setTextContent( this._primarySibling, content );
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
      this._primarySibling.removeEventListener( 'blur', this.blurEventListener );
      this._primarySibling.removeEventListener( 'focus', this.focusEventListener );

      // zero-out references
      this.accessibleInstance = null;
      this.node = null;
      this.display = null;
      this.trail = null;
      this._primarySibling = null;
      this._labelSibling = null;
      this._descriptionSibling = null;
      this._containerParent = null;

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
