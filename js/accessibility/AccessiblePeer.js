// Copyright 2015-2016, University of Colorado Boulder

/**
 * An accessible peer controls the appearance of an accessible Node's instance in the parallel DOM. An AccessiblePeer can
 * have up to four HTMLElements displayed in the PDOM, see ftructor for details.
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 * @author Jesse Greenberg
 */

define( function( require ) {
  'use strict';

  var AccessibleSiblingStyle = require( 'SCENERY/accessibility/AccessibleSiblingStyle' );
  var AccessibilityUtil = require( 'SCENERY/accessibility/AccessibilityUtil' );
  var arrayRemove = require( 'PHET_CORE/arrayRemove' );
  var Bounds2 = require( 'DOT/Bounds2' );
  var FullScreen = require( 'SCENERY/util/FullScreen' );
  var inherit = require( 'PHET_CORE/inherit' );
  var Matrix3 = require( 'DOT/Matrix3' );
  var platform = require( 'PHET_CORE/platform' );
  var Poolable = require( 'PHET_CORE/Poolable' );
  var scenery = require( 'SCENERY/scenery' );
  // so RequireJS doesn't complain about circular dependency
  // var Display = require( 'SCENERY/display/Display' );

  // constants
  var PRIMARY_SIBLING = 'PRIMARY_SIBLING';
  var LABEL_SIBLING = 'LABEL_SIBLING';
  var DESCRIPTION_SIBLING = 'DESCRIPTION_SIBLING';
  var CONTAINER_PARENT = 'CONTAINER_PARENT';
  var LABEL_TAG = AccessibilityUtil.TAGS.LABEL;
  var INPUT_TAG = AccessibilityUtil.TAGS.INPUT;

  // DOM observers that apply new CSS transformations are triggered when children, or inner content change. Updating
  // style/positioning of the element will change attributes so we can't observe those changes since it would trigger
  // the MutationObserver infinitely.
  var OBSERVER_CONFIG = { attributes: false, childList: true, characterData: true };

  var globalId = 1;

  // mutables instances to avoid creating many in operations that occur frequently
  var scratchGlobalBounds = new Bounds2( 0, 0, 0, 0 );
  var scratchSiblingBounds = new Bounds2( 0, 0, 0, 0 );
  var globalNodeTranslationMatrix = new Matrix3();
  var globalToClientScaleMatrix = new Matrix3();
  var nodeScaleMagnitudeMatrix = new Matrix3();

  /**
   * @param {AccessibleInstance} accessibleInstance
   * @param {Object} [options]
   * @constructor
   * @mixes Poolable
   */
  function AccessiblePeer( accessibleInstance, options ) {
    this.initializeAccessiblePeer( accessibleInstance, options );
  }

  scenery.register( 'AccessiblePeer', AccessiblePeer );

  inherit( Object, AccessiblePeer, {

    /**
     * Initializes the object (either from a freshly-created state, or from a "disposed" state brought back from a
     * pool).
     *
     * NOTE: the AccessiblePeer is not fully constructed until calling AccessiblePeer.update() after creating from pool.
     * @private
     *
     * @param {AccessibleInstance} accessibleInstance
     * @param {Object} [options]
     * @returns {AccessiblePeer} - Returns 'this' reference, for chaining
     */
    initializeAccessiblePeer: function( accessibleInstance, options ) {
      options = _.extend( {
        primarySibling: null
      }, options );

      assert && assert( !this.id || this.isDisposed, 'If we previously existed, we need to have been disposed' );

      // @public {number} - unique ID
      this.id = this.id || globalId++;

      // @public {AccessibleInstance}
      this.accessibleInstance = accessibleInstance;

      // @public {Node|null} only null for the root accessibleInstance
      this.node = this.accessibleInstance.node;

      // @public {Display} - Each peer is associated with a specific Display.
      this.display = accessibleInstance.display;

      // @public {Trail} - NOTE: May have "gaps" due to accessibleOrder usage.
      this.trail = accessibleInstance.trail;

      // @private {boolean|null} - whether or not this AccessiblePeer is visible in the PDOM
      // Only initialized to null, should not be set to it. isVisible() will return true if this.visible is null
      // (because it hasn't been set yet).
      this.visible = null;

      // @private {boolean|null} - whether or not the primary sibling of this AccessiblePeer can receive focus.
      this.focusable = null;

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

      // @private {boolean} - flag that indicates that this peer has accessible content that changed, and so
      // the siblings need to be repositioned in the next Display.updateDisplay()
      this.positionDirty = false;

      // @private {boolean} - indicates that this peer's accessibleInstance has a descendant that is dirty. Used to
      // quickly find peers with positionDirty when we traverse the tree of AccessibleInstances
      this.childPositionDirty = false;

      // @private {MutationObserver} - An observer that will call back any time a property of the primary
      // sibling changes. Used to reposition the sibling elements if the bounding box resizes. No need to loop over
      // all of the mutations, any single mutation will require updating CSS positioning.
      // 
      // NOTE: Ideally, a single MutationObserver could be used to observe changes to all elements in the PDOM. But
      // MutationObserver makes it impossible to detach observers from a single element. MutationObserver.detach()
      // will remove listeners on all observed elements, so individual observers must be used on each element.
      // One alternative could be to put the MutationObserver on the root element and use "subtree: true" in
      // OBSERVER_CONFIG. This could reduce the number of MutationObservers, but there is no easy way to get the
      // peer from the mutation target element. If MutationObserver takes a lot of memory, this could be an
      // optimization that may come with a performance cost.
      // 
      // NOTE: ResizeObserver is a superior alternative to MutationObserver for this purpose because
      // it will only monitor changes we care about and prevent infinite callback loops if size is changed in
      // the callback function (we get around this now by not observing attribute changes). But it is not yet widely
      // supported, see https://developer.mozilla.org/en-US/docs/Web/API/ResizeObserver.
      //
      // TODO: Should we be watching "model" changes from Accessibility.js instead of using MutationObserver?
      // See https://github.com/phetsims/scenery/issues/852. This would be less fragile, and also less
      // memory intensive because we don't need an instance of MutationObserver on every AccessibleInstance.
      this.mutationObserver = this.mutationObserver || new MutationObserver( this.invalidateCSSPositioning.bind( this ) );

      // @private {function} - must be removed on disposal
      this.transformListener = this.transformListener || this.invalidateCSSPositioning.bind( this );
      this.accessibleInstance.transformTracker.addListener( this.transformListener );

      // @private {boolean} - Whether we are currently in a "disposed" (in the pool) state, or are available to be
      // interacted with.
      this.isDisposed = false;

      // edge case for root accessibility
      if ( this.accessibleInstance.isRootInstance ) {

        // @private {HTMLElement} - The main element associated with this peer. If focusable, this is the element that gets
        // the focus. It also will contain any children.
        this._primarySibling = options.primarySibling;
        this._primarySibling.className = AccessibleSiblingStyle.ROOT_CLASS_NAME;
      }

      return this;
    },

    /**
     * Update the content of the peer. This must be called after the AccessibePeer is constructed from pool.
     * @public (scenery-internal)
     */
    update: function() {
      var uniqueId = this.accessibleInstance.trail.getUniqueId();

      var options = this.node.getBaseOptions();

      if ( this.node.accessibleName !== null ) {
        options = this.node.accessibleNameBehavior( this.node, options, this.node.accessibleName );
      }

      if ( this.node.accessibleHeading !== null ) {
        options = this.node.accessibleHeadingBehavior( this.node, options, this.node.accessibleHeading );
      }

      if ( this.node.helpText !== null ) {
        options = this.node.helpTextBehavior( this.node, options, this.node.helpText );
      }

      // create the base DOM element representing this accessible instance
      // TODO: why not just options.focusable?
      this._primarySibling = createElement( options.tagName, this.node.focusable, uniqueId, {
        namespace: options.accessibleNamespace
      } );

      // create the container parent for the dom siblings
      if ( options.containerTagName ) {
        this._containerParent = createElement( options.containerTagName, false, uniqueId, {
          siblingName: 'container'
        } );
      }

      // create the label DOM element representing this instance
      if ( options.labelTagName ) {
        this._labelSibling = createElement( options.labelTagName, false, uniqueId, {
          siblingName: 'label'
        } );
      }

      // create the description DOM element representing this instance
      if ( options.descriptionTagName ) {
        this._descriptionSibling = createElement( options.descriptionTagName, false, uniqueId, {
          siblingName: 'description'
        } );
      }

      this.orderElements( options );

      // assign listeners (to be removed or disconnected during disposal)
      this.mutationObserver.disconnect(); // in case update() is called more than once on an instance of AccessiblePeer
      this.mutationObserver.observe( this._primarySibling, OBSERVER_CONFIG );

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

      this.setFocusable( this.node.focusable );

      // recompute and assign the association attributes that link two elements (like aria-labelledby)
      this.onAriaLabelledbyAssociationChange();
      this.onAriaDescribedbyAssociationChange();
      this.onActiveDescendantAssociationChange();

      // update all attributes for the peer, should cover aria-label, role, and others
      this.onAttributeChange( options );

      // update input value attribute for the peer
      this.onInputValueChange();

      this.node.updateOtherNodesAriaLabelledby();
      this.node.updateOtherNodesAriaDescribedby();
      this.node.updateOtherNodesActiveDescendant();
    },

    /**
     * Handle the internal ordering of the elements in the peer, this involves setting the proper value of
     * this.topLevelElements
     * @param {Object} options - the computed mixin options to be applied to the peer.
     * @private
     */
    orderElements: function( options ) {
      if ( this._containerParent ) {
        // The first child of the container parent element should be the peer dom element
        // if undefined, the insertBefore method will insert the this._primarySibling as the first child
        this._containerParent.insertBefore( this._primarySibling, this._containerParent.children[ 0 ] || null );
        this.topLevelElements = [ this._containerParent ];
      }
      else {

        // Wean out any null siblings
        this.topLevelElements = [ this._labelSibling, this._descriptionSibling, this._primarySibling ].filter( _.identity );
      }

      // insert the label and description elements in the correct location if they exist
      // NOTE: Important for arrangeContentElement to be called on the label sibling first for correct order
      this._labelSibling && this.arrangeContentElement( this._labelSibling, options.appendLabel );
      this._descriptionSibling && this.arrangeContentElement( this._descriptionSibling, options.appendDescription );

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
     * Recompute the aria-activedescendant attributes for all of the peer's elements
     * @public
     */
    onActiveDescendantAssociationChange: function() {
      this.removeAttributeFromAllElements( 'aria-activedescendant' );

      for ( var i = 0; i < this.node.activeDescendantAssociations.length; i++ ) {
        var associationObject = this.node.activeDescendantAssociations[ i ];

        // Assert out if the model list is different than the data held in the associationObject
        assert && assert( associationObject.otherNode.nodesThatAreActiveDescendantToThisNode.indexOf( this.node ) >= 0,
          'unexpected otherNode' );


        this.setAssociationAttribute( 'aria-activedescendant', associationObject );
      }
    },

    /**
     * Set all accessible attributes onto the peer elements from the model's stored data objects
     * @private
     *
     * @param {Object} [a11yOptions] - these can override the values of the node, see this.update()
     */
    onAttributeChange: function( a11yOptions ) {

      for ( var i = 0; i < this.node.accessibleAttributes.length; i++ ) {
        var dataObject = this.node.accessibleAttributes[ i ];
        var attribute = dataObject.attribute;
        var value = dataObject.value;

        // allow overriding of aria-label for accessibleName setter
        if ( attribute === 'aria-label' && a11yOptions && typeof a11yOptions.ariaLabel === 'string' && dataObject.options.elementName === PRIMARY_SIBLING ) {
          value = a11yOptions.ariaLabel;
        }
        this.setAttributeToElement( attribute, value, dataObject.options );
      }
    },

    /**
     * Set the input value on the peer's primary sibling element. The value attribute must be set as a Property to be
     * registered correctly by an assistive device. If null, the attribute is removed so that we don't clutter the DOM
     * with value="null" attributes.
     *
     * @public (scenery-internal)
     */
    onInputValueChange: function() {
      assert && assert( this.node.inputValue !== undefined, 'use null to remove input value attribute' );

      if ( this.node.inputValue === null ) {
        this.removeAttributeFromElement( 'value' );
      }
      else {

        // type conversion for DOM spec
        var valueString = this.node.inputValue + '';
        this.setAttributeToElement( 'value', valueString, { asProperty: true } );
      }
    },

    /**
     * Get an element on this node, looked up by the elementName flag passed in.
     * @public (scenery-internal)
     *
     * @param {string} elementName - see AccessibilityUtil for valid associations
     * @returns {HTMLElement}
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

        // set as a javascript property instead of an attribute on the DOM Element.
        asProperty: false,

        elementName: PRIMARY_SIBLING, // see this.getElementName() for valid values, default to the primary sibling

        // {HTMLElement|null} - element that will directly receive the input rather than looking up by name, if
        // provided, elementName option will have no effect
        element: null
      }, options );

      var element = options.element || this.getElementByName( options.elementName );

      if ( options.namespace ) {
        element.setAttributeNS( options.namespace, attribute, attributeValue );
      }
      else if ( options.asProperty ) {
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

        elementName: PRIMARY_SIBLING, // see this.getElementName() for valid values, default to the primary sibling

        // {HTMLElement|null} - element that will directly receive the input rather than looking up by name, if
        // provided, elementName option will have no effect
        element: null
      }, options );

      var element = options.element || this.getElementByName( options.elementName );

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
      assert && assert( AccessibilityUtil.ASSOCIATION_ATTRIBUTES.indexOf( attribute ) >= 0,
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
     * The contentElement will either be a label or description element. The contentElement will be sorted relative to
     * the primarySibling. Its placement will also depend on whether or not this node wants to append this element,
     * see setAppendLabel() and setAppendDescription(). By default, the "content" element will be placed before the
     * primarySibling.
     *
     * NOTE: This function assumes it is called on label sibling before description sibling for inserting elements
     * into the correct order.
     *
     * @private
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

        // keep this.topLevelElements in sync
        arrayRemove( this.topLevelElements, contentElement );
        var indexOfPrimarySibling = this.topLevelElements.indexOf( this._primarySibling );

        // if appending, just insert at at end of the top level elements
        var insertIndex = appendElement ? this.topLevelElements.length : indexOfPrimarySibling;
        this.topLevelElements.splice( insertIndex, 0, contentElement );
      }
    },

    /**
     * Is this peer hidden in the PDOM
     * @public
     *
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
     * @public
     *
     * @param {boolean} visible
     */
    setVisible: function( visible ) {
      assert && assert( typeof visible === 'boolean' );
      if ( this.visible !== visible ) {

        this.visible = visible;
        for ( var i = 0; i < this.topLevelElements.length; i++ ) {
          var element = this.topLevelElements[ i ];
          if ( visible ) {
            this.removeAttributeFromElement( 'hidden', { element: element } );
          }
          else {
            this.setAttributeToElement( 'hidden', '', { element: element } );
          }
        }

        // invalidate CSS transforms because when 'hidden' the content will have no dimensions in the viewport
        this.invalidateCSSPositioning();
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

      // no op if primary sibling does not have focus
      if ( document.activeElement === this._primarySibling ) {

        // Workaround for a bug in IE11 in Fullscreen mode where document.activeElement.blur() errors out with
        // "Invalid Function". A delay seems to be a common workaround for IE11, see
        // https://stackoverflow.com/questions/2600186/focus-doesnt-work-in-ie
        var self = this;
        if ( platform.ie11 && FullScreen.isFullScreen() ) {
          window.setTimeout( function() {

            // make sure that the primary sibling hasn't been removed from the document since the timeout was added
            self._primarySibling && self._primarySibling.blur();
          }, 0 );
        }
        else {
          this._primarySibling.blur();
        }
      }
    },

    /**
     * Make the peer focusable. Only the primary sibling is ever considered focusable.
     * @public
     * @param {boolean} focusable
     */
    setFocusable: function( focusable ) {
      assert && assert( typeof focusable === 'boolean' );

      const peerHadFocus = this.isFocused();
      if ( this.focusable !== focusable ) {
        this.focusable = focusable;
        AccessibilityUtil.overrideFocusWithTabIndex( this.primarySibling, focusable );

        // in Chrome, if tabindex is removed and the element is not focusable by default the element is blurred.
        // This behavior is reasonable and we want to enforce it in other browsers for consistency. See
        // https://github.com/phetsims/scenery/issues/967
        if ( peerHadFocus && !focusable ) {
          this.blur();
        }

        // reposition the sibling in the DOM, since non-focusable nodes are not positioned
        this.invalidateCSSPositioning();
      }
    },

    /**
     * Responsible for setting the content for the label sibling
     * @public (scenery-internal)
     * @param {string} content - the content for the label sibling.
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
        this.setAttributeToElement( 'for', this._primarySibling.id, {
          elementName: AccessiblePeer.LABEL_SIBLING
        } );
      }
    },

    /**
     * Responsible for setting the content for the description sibling
     * @public (scenery-internal)
     * @param {string} content - the content for the description sibling.
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
     * @param {string} content - the content for the primary sibling.
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
     * Mark that the siblings of this AccessiblePeer need to be updated in the next Display update. Possibly from a
     * change of accessible content or node transformation. Does nothing if already marked dirty.
     *
     * TODO: We shouldn't be marking all elements as dirty, just those that are focusable. Setting focusable
     * should therfore mark dirty.
     *
     * @private
     */
    invalidateCSSPositioning: function() {
      if ( !this.positionDirty && this.focusable ) {
        this.positionDirty = true;

        // mark all ancestors of this peer so that we can quickly find this dirty peer when we traverse
        // the AccessibleInstance tree
        var parent = this.accessibleInstance.parent;
        while ( parent ) {
          parent.peer.childPositionDirty = true;
          parent = parent.parent;
        }
      }
    },

    /**
     * Update the CSS positioning of the primary and label siblings. Required to support accessibility on mobile
     * devices. On activation of focusable elements, certain AT will send fake pointer events to the browser at
     * the center of the client bounding rectangle of the HTML element. By positioning elements over graphical display
     * objects we can capture those events. A transformation matrix is calculated that will transform the position
     * and dimension of the HTML element in pixels to the global coordinate frame. The matrix is used to transform
     * the bounds of the element prior to any other transformation so we can set the element's left, top, width, and
     * height with CSS attributes.
     *
     * For now we are only transforming the primary and label siblings if the primary sibling is focusable. If
     * focusable, the primary sibling needs to be transformed to receive user input. VoiceOver includes the label bounds
     * in its calculation for where to send the events, so it needs to be transformed as well. Descriptions are not
     * considered and do not need to be positioned.
     *
     * Initially, we tried to set the CSS transformations on elements directly through the transform attribute. While
     * this worked for basic input, it did not support other AT features like tapping the screen to focus elements.
     * With this strategy, the VoiceOver "touch area" was a small box around the top left corner of the element. It was
     * never clear why this was this case, but forced us to change our strategy to set the left, top, width, and height
     * attributes instead.
     *
     * This function assumes that elements have other style attributes so they can be positioned correctly and don't
     * interfere with scenery input, see SceneryStyle in AccessibilityUtil.
     *
     * Additional notes were taken in https://github.com/phetsims/scenery/issues/852, see that issue for more
     * information.
     *
     * Review: This function could be simplified by setting the element width/height a small arbitrary shape
     * at the center of the node's global bounds. There is a drawback in that the VO default highlight won't
     * surround the Node anymore. But it could be a performance enhancement and simplify this function.
     * Or maybe a big rectangle larger than the Display div still centered on the node so we never
     * see the VO highlight?
     *
     * @private
     */
    positionElements: function() {
      assert && assert( this._primarySibling, 'a primary sibling required to receive CSS positioning' );
      assert && assert( this.positionDirty, 'elements should only be repositioned if dirty' );

      // CSS transformation only needs to be applied if the node is focusable - otherwise the element will be found
      // by gesture navigation with the virtual cursor. Bounds for non-focusable elements in the ViewPort don't
      // need to be accurate because the AT doesn't need to send events to them.
      if ( this.node.focusable ) {

        scratchGlobalBounds.set( this.node.localBounds );
        if ( scratchGlobalBounds.isFinite() ) {

          scratchGlobalBounds.transform( this.accessibleInstance.transformTracker.getMatrix() );
          var scaleVector = this.node.getScaleVector(); // could be optimized to create less Vector2 instances

          var clientDimensions = getClientDimensions( this._primarySibling );
          var clientWidth = clientDimensions.width;
          var clientHeight = clientDimensions.height;

          if ( clientWidth > 0 && clientHeight > 0 ) {
            scratchSiblingBounds.setMinMax( 0, 0, clientWidth, clientHeight );
            scratchSiblingBounds.transform( getCSSMatrix( this._primarySibling, clientWidth, clientHeight, scratchGlobalBounds, scaleVector ) );
            setClientBounds( this._primarySibling, scratchSiblingBounds );
          }

          if ( this.labelSibling ) {
            clientDimensions = getClientDimensions( this._labelSibling );
            clientWidth = clientDimensions.width;
            clientHeight = clientDimensions.height;

            if ( clientHeight > 0 && clientWidth > 0 ) {
              scratchSiblingBounds.setMinMax( 0, 0, clientWidth, clientHeight );
              scratchSiblingBounds.transform( getCSSMatrix( this.labelSibling, clientWidth, clientHeight, scratchGlobalBounds, scaleVector ) );
              setClientBounds( this._labelSibling, scratchSiblingBounds );
            }
          }
        }
      }

      this.positionDirty = false;
    },

    /**
     * Update positioning of elements in the PDOM. Does a depth first search for all descendants of parentIntsance with
     * a peer that either has dirty positioning or as a descendant with dirty positioning.
     *
     * @public (scenery-internal)
     */
    updateSubtreePositioning: function() {
      this.childPositionDirty = false;

      if ( this.positionDirty ) {
        this.positionElements();
      }

      for ( var i = 0; i < this.accessibleInstance.children.length; i++ ) {
        var childPeer = this.accessibleInstance.children[ i ].peer;
        if ( childPeer.positionDirty || childPeer.childPositionDirty ) {
          this.accessibleInstance.children[ i ].peer.updateSubtreePositioning();
        }
      }
    },

    /**
     * Removes external references from this peer, and places it in the pool.
     * @public (scenery-internal)
     */
    dispose: function() {
      this.isDisposed = true;

      // remove focus if the disposed peer is the active element
      this.blur();

      // remove listeners
      this._primarySibling.removeEventListener( 'blur', this.blurEventListener );
      this._primarySibling.removeEventListener( 'focus', this.focusEventListener );
      this.accessibleInstance.transformTracker.removeListener( this.transformListener );
      this.mutationObserver.disconnect();

      // zero-out references
      this.accessibleInstance = null;
      this.node = null;
      this.display = null;
      this.trail = null;
      this._primarySibling = null;
      this._labelSibling = null;
      this._descriptionSibling = null;
      this._containerParent = null;
      this.focusable = null;

      // for now
      this.freeToPool();
    }
  }, {

    // @public {string} - specifies valid associations between related AccessiblePeers in the DOM
    PRIMARY_SIBLING: PRIMARY_SIBLING, // associate with all accessible content related to this peer
    LABEL_SIBLING: LABEL_SIBLING, // associate with just the label content of this peer
    DESCRIPTION_SIBLING: DESCRIPTION_SIBLING, // associate with just the description content of this peer
    CONTAINER_PARENT: CONTAINER_PARENT // associate with everything under the container parent of this peer
  } );

  // Set up pooling
  Poolable.mixInto( AccessiblePeer, {
    initalize: AccessiblePeer.prototype.initializeAccessiblePeer
  } );

  //--------------------------------------------------------------------------
  // Helper functions
  //--------------------------------------------------------------------------
  
  /**
   * Create a sibling element for the AccessiblePeer.
   *
   * @param {string} tagName
   * @param {boolean} focusable
   * @param {string} trailId - unique id that points to the instance of the node
   * @param {object} options - passed along to AccessibilityUtil.createElement
   * @returns {HTMLElement}
   */
  function createElement( tagName, focusable, trailId, options ) {
    options = _.extend( {

      // {string|null} - addition to the trailId, separated by a hyphen to identify the different siblings within
      // the document
      siblingName: null
    }, options );

    // add sibling name to unique ID generated from the trail to make the non-primary siblings unique in the DOM
    assert && assert( options.id === undefined, 'createElement will set optional id' );
    options.id = options.siblingName ? `${options.siblingName}-${trailId}` : trailId;

    assert && assert( options.trailId === undefined, 'createElement will set optional trailId' );
    options.trailId = trailId;

    return AccessibilityUtil.createElement( tagName, focusable, options );
  }

  /**
   * Get a matrix that can be used as the CSS transform for elements in the DOM. This matrix will an HTML element
   * dimensions in pixels to the global coordinate frame.
   *
   * @param  {HTMLElement} element - the element to receive the CSS transform
   * @param  {number} clientWidth - width of the element to transform in pixels
   * @param  {number} clientHeight - height of the element to transform in pixels
   * @param  {Bounds2} nodeGlobalBounds - Bounds of the AccessiblePeer's node in the global coordinate frame.
   * @param  {Vector2} scaleVector - the scale magnitude Vector for the Node.
   * @returns {Matrix3}
   */
  function getCSSMatrix( element, clientWidth, clientHeight, nodeGlobalBounds, scaleVector ) {

    // the translation matrix for the node's bounds in its local coordinate frame
    globalNodeTranslationMatrix.setToTranslation( nodeGlobalBounds.minX, nodeGlobalBounds.minY );

    // scale matrix for "client" HTML element, scale to make the HTML element's DOM bounds match the
    // local bounds of the node
    globalToClientScaleMatrix.setToScale( nodeGlobalBounds.width / clientWidth, nodeGlobalBounds.height / clientHeight );
    nodeScaleMagnitudeMatrix.setToScale( scaleVector.x, scaleVector.y );

    // combine these in a single transformation matrix
    return globalNodeTranslationMatrix.multiplyMatrix( globalToClientScaleMatrix ).multiplyMatrix( nodeScaleMagnitudeMatrix );
  }

  /**
   * Gets an object with the width and height of an HTML element in pixels, prior to any scaling. clientWidth and
   * clientHeight are zero for elements with inline layout and elements without CSS. For those elements we fall back
   * to the boundingClientRect, which at that point will describe the dimensions of the element prior to scaling.
   *
   * @param  {HTMLElement} siblingElement
   * @returns {Object} - Returns an object with two entries, { width: {number}, height: {number} }
   */
  function getClientDimensions( siblingElement ) {
    var clientWidth = siblingElement.clientWidth;
    var clientHeight = siblingElement.clientHeight;

    if ( clientWidth === 0 && clientHeight === 0 ) {
      var clientRect = siblingElement.getBoundingClientRect();
      clientWidth = clientRect.width;
      clientHeight = clientRect.height;
    }

    return { width: clientWidth, height: clientHeight };
  }

  /**
   * Set the bounds of the sibling element in the view port in pixels, using top, left, width, and height css.
   * The element must be styled with 'position: fixed', and an ancestor must have position: 'relative', so that
   * the dimensions of the sibling are relative to the parent.
   *
   * @param {HTMLElement} siblingElement - the element to position
   * @param {Bounds2} bounds - desired bounds, in pixels
   */
  function setClientBounds( siblingElement, bounds ) {
    siblingElement.style.top = bounds.top + 'px';
    siblingElement.style.left = bounds.left + 'px';
    siblingElement.style.width = bounds.width + 'px';
    siblingElement.style.height = bounds.height + 'px';
  }

  return AccessiblePeer;
} );
